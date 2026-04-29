import json
import os
import random
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

import gradio as gr

COMFYUI_URL = "http://localhost:8188"
INVOKEAI_DB = "/mnt/llm/hub/invokeai_data/databases/invokeai.db"
HUBMODELS_DB = "/mnt/llm/hub/hubmodels/hubrootv3.db"
HUBMODELS_DIR = "/mnt/llm/hub/hubmodels/models"
COMFYMDIR = os.environ.get("COMFYMDIR", "/home/dev/comfy/ComfyUI/models")
OUTPUT_DIR = Path(os.environ.get("COMFYUI_OUTPUT_ROOT", "/home/dev/comfy/ComfyUI/output"))

PRESETS = {
    "Z-IMAGE": (8, 1, "res_multistep", "simple", "lumina2", True),
    "FLUX": (20, 1, "euler", "simple", None, False),
    "SDXL": (20, 7, "euler", "normal", None, False),
    "SD1.5": (20, 7, "euler", "normal", None, False),
    "OTHER": (20, 7, "euler", "normal", None, False),
}

FAMILIES = {
    "ALL": None,
    "Z-IMAGE": lambda n: "zimage" in n or "z_image" in n or "z-image" in n or "z_img" in n,
    "FLUX": lambda n: "flux" in n,
    "SDXL": lambda n: any(x in n for x in ["sdxl", "xl", "juggernaut", "pony", "illustrious", "animagine"]),
    "SD1.5": lambda n: any(x in n for x in ["sd15", "sd_1", "v1-5", "v1_5", "dreamshaper", "realistic"]),
    "OTHER": lambda n: True,
}

TYPE_TO_SUBFOLDER = {
    "main": "checkpoints", "lora": "loras", "vae": "vae",
    "embedding": "embeddings", "controlnet": "controlnet",
    "clip_vision": "clip_vision", "ip_adapter": "controlnet",
    "spandrel_image_to_image": "upscale_models",
}


def classify_model(name):
    n = name.lower()
    for family, test in FAMILIES.items():
        if family in ("ALL", "OTHER"):
            continue
        if test(n):
            return family
    return "OTHER"


# ── ComfyUI API ──

def api_get(endpoint):
    try:
        return json.loads(urllib.request.urlopen(f"{COMFYUI_URL}{endpoint}", timeout=10).read())
    except Exception:
        return None


def api_post(endpoint, data):
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(f"{COMFYUI_URL}{endpoint}", data=payload, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def get_comfy_models(loader_type, key):
    data = api_get(f"/object_info/{loader_type}")
    if not data:
        return []
    try:
        return data[loader_type]["input"]["required"][key][0]
    except (KeyError, IndexError):
        return []


def get_all_model_names():
    return get_comfy_models("UNETLoader", "unet_name") + get_comfy_models("CheckpointLoaderSimple", "ckpt_name")


# ── Hub Symlink ──

def symlink_from_hub(blake3_hash, filename, subfolder):
    hex_hash = blake3_hash.replace("blake3:", "")
    source = Path(HUBMODELS_DIR) / hex_hash / filename
    if not source.exists():
        return False
    target_dir = Path(COMFYMDIR) / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if target.exists() or target.is_symlink():
        return True
    target.symlink_to(source)
    return True


def find_and_link_from_hub(invoke_name, invoke_hash, invoke_type):
    subfolder = TYPE_TO_SUBFOLDER.get(invoke_type, "checkpoints")
    hex_hash = invoke_hash.replace("blake3:", "") if invoke_hash else ""
    try:
        with sqlite3.connect(HUBMODELS_DB) as conn:
            cursor = conn.cursor()
            if hex_hash:
                cursor.execute("SELECT hash_blake3, filename FROM models WHERE hash_blake3 = ? AND deleted = 0", (hex_hash,))
                row = cursor.fetchone()
                if row and symlink_from_hub(row[0], row[1], subfolder):
                    return row[1], subfolder
            cursor.execute("SELECT hash_blake3, filename FROM models WHERE filename LIKE ? AND deleted = 0 LIMIT 5", (f"%{invoke_name}%",))
            for h, fn in cursor.fetchall():
                if symlink_from_hub(h, fn, subfolder):
                    return fn, subfolder
    except Exception:
        pass
    return None


def map_invoke_model_to_comfy(invoke_model_name, invoke_base, invoke_hash=None, invoke_type="main"):
    unet_models = get_comfy_models("UNETLoader", "unet_name")
    ckpt_models = get_comfy_models("CheckpointLoaderSimple", "ckpt_name")
    name_lower = invoke_model_name.lower().replace("-", "").replace("_", "")
    for m in unet_models + ckpt_models:
        m_lower = m.lower().replace("-", "").replace("_", "").replace(".safetensors", "")
        if name_lower in m_lower or m_lower in name_lower:
            return m, "unet" if m in unet_models else "ckpt"
    result = find_and_link_from_hub(invoke_model_name, invoke_hash, invoke_type)
    if result:
        return result[0], "ckpt" if result[1] == "checkpoints" else "unet"
    return None, None


def map_invoke_lora_to_comfy(invoke_lora_name, invoke_hash=None):
    comfy_loras = get_comfy_models("LoraLoader", "lora_name")
    name_lower = invoke_lora_name.lower().replace("-", "").replace("_", "")
    for l in comfy_loras:
        l_lower = l.lower().replace("-", "").replace("_", "").replace(".safetensors", "")
        if name_lower in l_lower or l_lower in name_lower:
            return l
    result = find_and_link_from_hub(invoke_lora_name, invoke_hash, "lora")
    if result:
        return result[0]
    return None


# ── InvokeAI State ──

def read_invoke_state():
    try:
        with sqlite3.connect(INVOKEAI_DB) as conn:
            row = conn.execute("SELECT value FROM client_state WHERE user_id='system' AND key='params'").fetchone()
            params = json.loads(row[0]) if row else {}
            row = conn.execute("SELECT value FROM client_state WHERE user_id='system' AND key='loras'").fetchone()
            loras_state = json.loads(row[0]) if row else {}
        return params, loras_state
    except Exception:
        return {}, {}


def pull_from_invoke():
    params, loras_state = read_invoke_state()
    if not params:
        return [gr.update()] * 15 + ["Failed to read InvokeAI state"]

    prompt = params.get("positivePrompt", "")
    neg_prompt = params.get("negativePrompt", "")
    width = params.get("dimensions", {}).get("width", 1024)
    height = params.get("dimensions", {}).get("height", 1024)
    steps = params.get("steps", 20)
    cfg = params.get("cfgScale", 7)
    seed = params.get("seed", 42)
    random_seed = params.get("shouldRandomizeSeed", True)

    invoke_scheduler = params.get("scheduler", "euler")
    scheduler_map = {"euler": "normal", "euler_k": "karras", "euler_a": "normal",
        "dpmpp_2m": "normal", "dpmpp_2m_k": "karras", "dpmpp_2m_sde": "karras",
        "dpmpp_sde": "karras", "ddim": "ddim_uniform", "lcm": "simple", "unipc": "normal",
        "kdpm_2": "karras", "heun": "normal"}
    scheduler = scheduler_map.get(invoke_scheduler, "normal")
    sampler_map = {"euler": "euler", "euler_k": "euler", "euler_a": "euler_ancestral",
        "dpmpp_2m": "dpmpp_2m", "dpmpp_2m_k": "dpmpp_2m", "dpmpp_2m_sde": "dpmpp_2m_sde",
        "dpmpp_sde": "dpmpp_sde", "ddim": "ddim", "lcm": "lcm", "unipc": "uni_pc",
        "kdpm_2": "dpm_2", "heun": "heun"}
    sampler = sampler_map.get(invoke_scheduler, "euler")

    invoke_model = params.get("model", {})
    comfy_model, model_type = map_invoke_model_to_comfy(
        invoke_model.get("name", ""), invoke_model.get("base", ""),
        invoke_model.get("hash", ""), invoke_model.get("type", "main"))

    lora_lines = []
    for entry in loras_state.get("loras", []):
        if not entry.get("isEnabled", False):
            continue
        lm = entry.get("model", {})
        cl = map_invoke_lora_to_comfy(lm.get("name", ""), lm.get("hash", ""))
        w = entry.get("weight", 1.0)
        lora_lines.append(f"{cl}:{w}" if cl else f"?{lm.get('name','')}:{w}")

    all_models = get_all_model_names()
    clips = get_comfy_models("CLIPLoader", "clip_name")
    vaes = get_comfy_models("VAELoader", "vae_name")

    status = f"Pulled: {invoke_model.get('name','')} -> {comfy_model or '?'}"
    if lora_lines:
        status += f" + {sum(1 for l in lora_lines if not l.startswith('?'))} LoRA(s)"

    return [
        gr.update(value=prompt), gr.update(value=neg_prompt),
        gr.update(value=width), gr.update(value=height),
        gr.update(value=steps), gr.update(value=cfg),
        gr.update(value=seed), gr.update(value=random_seed),
        gr.update(value=sampler), gr.update(value=scheduler),
        gr.update(choices=all_models, value=comfy_model),
        gr.update(value="\n".join(lora_lines) if lora_lines else ""),
        gr.update(choices=clips, value=clips[0] if clips else None),
        gr.update(choices=vaes, value=vaes[0] if vaes else None),
        status,
    ]


# ── Gallery ──

def get_gallery_images(count=80):
    if not OUTPUT_DIR.exists():
        return []
    images = []
    for f in OUTPUT_DIR.iterdir():
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and f.is_file():
            images.append(f)
    for subdir in OUTPUT_DIR.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and f.is_file():
                    images.append(f)
    images.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return [str(f) for f in images[:count]]


GALLERY_SCRIPT = """<script>
if (!window._cgInit) {
    window._cgSelected = new Set();
    window._cgInit = true;
    var _si = function(id, val) {
        var el = document.querySelector('#' + id + ' textarea');
        if (!el) return;
        var set = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
        set.call(el, val);
        el.dispatchEvent(new Event('input', {bubbles: true}));
    };
    var _cb = function(id) {
        setTimeout(function() { var b = document.querySelector('#' + id + ' button'); if (b) b.click(); }, 50);
    };
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            e.stopPropagation();
            var gb = document.querySelector('#gen-btn button');
            if (gb) gb.click();
        }
    }, true);
    document.addEventListener('click', function(e) {
        var item = e.target.closest('.cg-item');
        if (!item) return;
        var path = item.dataset.path;
        var sel = window._cgSelected;
        if (e.ctrlKey || e.metaKey) {
            if (sel.has(path)) { sel.delete(path); item.classList.remove('selected'); }
            else { sel.add(path); item.classList.add('selected'); }
        } else {
            document.querySelectorAll('.cg-item.selected').forEach(function(i) { i.classList.remove('selected'); });
            sel.clear();
            sel.add(path);
            item.classList.add('selected');
        }
        _si('selected-paths', JSON.stringify(Array.from(sel)));
        _si('viewed-path', path);
        _cb('view-btn');
        var cnt = document.querySelector('#sel-count');
        if (cnt) cnt.textContent = sel.size > 1 ? sel.size + ' selected' : '';
    });
    document.addEventListener('dblclick', function(e) {
        var item = e.target.closest('.cg-item');
        if (!item) return;
        _si('viewed-path', item.dataset.path);
        _cb('load-btn');
    });
    document.addEventListener('keydown', function(e) {
        var wrap = document.querySelector('.cg-wrap');
        if (!wrap) return;
        var sel = window._cgSelected;
        if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
            if (!wrap.contains(document.activeElement) && document.activeElement !== wrap) return;
            e.preventDefault();
            sel.clear();
            document.querySelectorAll('.cg-item').forEach(function(i) { i.classList.add('selected'); sel.add(i.dataset.path); });
            _si('selected-paths', JSON.stringify(Array.from(sel)));
            var cnt = document.querySelector('#sel-count');
            if (cnt) cnt.textContent = sel.size + ' selected';
        }
        if (e.key === 'Delete') {
            if (sel.size > 0) { e.preventDefault(); _cb('del-btn'); }
        }
    });
}
</script>"""


def render_gallery():
    images = get_gallery_images()
    if not images:
        return '<div class="cg-wrap" tabindex="0"><div class="cg-empty">No images</div></div>' + GALLERY_SCRIPT
    items = []
    for path in images:
        fname = Path(path).name
        url = "/gradio_api/file=" + urllib.parse.quote(path)
        items.append(
            f'<div class="cg-item" data-path="{path}" title="{fname}">'
            f'<img src="{url}" loading="lazy" draggable="false">'
            f'<div class="cg-badge"></div></div>')
    return f'<div class="cg-wrap" tabindex="0">{"".join(items)}</div>{GALLERY_SCRIPT}'


def load_settings_from_image(image_path):
    if not image_path or not Path(image_path).exists():
        return [gr.update()] * 13 + ["No image selected"]
    try:
        from PIL import Image
        img = Image.open(image_path)
        if "prompt" not in img.info:
            return [gr.update()] * 13 + [f"No metadata in {Path(image_path).name}"]
        wf = json.loads(img.info["prompt"])
        prompt = neg_prompt = model_name = sampler = scheduler = ""
        lora_lines = []
        steps = cfg = seed = width = height = 0
        clip_nodes = []

        for nid, node in wf.items():
            ct = node.get("class_type", "")
            inp = node.get("inputs", {})
            if ct == "CLIPTextEncode":
                clip_nodes.append((nid, inp.get("text", "")))
            elif ct == "CheckpointLoaderSimple":
                model_name = inp.get("ckpt_name", "")
            elif ct == "UNETLoader":
                model_name = inp.get("unet_name", "")
            elif ct == "LoraLoader":
                ln = inp.get("lora_name", "")
                if ln:
                    lora_lines.append(f"{ln}:{inp.get('strength_model', 1.0)}")
            elif ct == "KSampler":
                steps, cfg, seed = inp.get("steps", 20), inp.get("cfg", 7), inp.get("seed", 42)
                sampler, scheduler = inp.get("sampler_name", "euler"), inp.get("scheduler", "normal")
            elif ct in ("EmptyLatentImage", "EmptySD3LatentImage"):
                width, height = inp.get("width", 1024), inp.get("height", 1024)

        # First CLIPTextEncode is positive, second is negative (or ConditioningZeroOut)
        if clip_nodes:
            prompt = clip_nodes[0][1]
            if len(clip_nodes) > 1:
                neg_prompt = clip_nodes[1][1]

        all_models = get_all_model_names()
        return [
            gr.update(value=prompt), gr.update(value=neg_prompt),
            gr.update(value=int(width)), gr.update(value=int(height)),
            gr.update(value=int(steps)), gr.update(value=float(cfg)),
            gr.update(value=int(seed)), gr.update(value=False),
            gr.update(value=sampler), gr.update(value=scheduler),
            gr.update(choices=all_models, value=model_name),
            gr.update(value="\n".join(lora_lines)),
            image_path,
            f"Loaded: {Path(image_path).name}",
        ]
    except Exception as e:
        return [gr.update()] * 13 + [f"Error: {e}"]


def delete_images(selected_json):
    if not selected_json:
        return render_gallery(), "No images selected"
    try:
        paths = json.loads(selected_json)
        deleted = sum(1 for p in paths if Path(p).exists() and not Path(p).unlink() and True or True)
        return render_gallery(), f"Deleted {len(paths)} image(s)"
    except Exception as e:
        return render_gallery(), f"Delete error: {e}"


# ── Workflow Builders ──

def chain_loras(wf, lora_text, model_out, clip_out, start_id):
    if not lora_text or not lora_text.strip():
        return model_out, clip_out, start_id
    nid = start_id
    for line in lora_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("?"):
            continue
        parts = line.rsplit(":", 1)
        name = parts[0].strip()
        weight = float(parts[1]) if len(parts) > 1 else 1.0
        wf[str(nid)] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": name, "strength_model": weight,
            "strength_clip": weight, "model": model_out, "clip": clip_out}}
        model_out, clip_out = [str(nid), 0], [str(nid), 1]
        nid += 1
    return model_out, clip_out, nid


def build_unet_workflow(model, clip, vae, prompt, neg, loras, w, h, steps, cfg, sampler, sched, seed, prefix, clip_type="lumina2"):
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": clip_type, "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"shift": 3, "model": ["1", 0]}},
    }
    mo, co = ["4", 0], ["2", 0]
    mo, co, _ = chain_loras(wf, loras, mo, co, 20)
    wf.update({
        "50": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": co}},
        "51": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["50", 0]}},
        "52": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "53": {"class_type": "KSampler", "inputs": {"seed": seed, "control_after_generate": "fixed", "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": sched, "denoise": 1, "model": mo, "positive": ["50", 0], "negative": ["51", 0], "latent_image": ["52", 0]}},
        "54": {"class_type": "VAEDecode", "inputs": {"samples": ["53", 0], "vae": ["3", 0]}},
        "55": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["54", 0]}},
    })
    return wf


def build_ckpt_workflow(model, prompt, neg, loras, w, h, steps, cfg, sampler, sched, seed, prefix, vae_override=None):
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}}}
    mo, co, vo = ["1", 0], ["1", 1], ["1", 2]
    mo, co, _ = chain_loras(wf, loras, mo, co, 20)
    if vae_override:
        wf["2"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae_override}}
        vo = ["2", 0]
    wf.update({
        "50": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": co}},
        "51": {"class_type": "CLIPTextEncode", "inputs": {"text": neg or "", "clip": co}},
        "52": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "53": {"class_type": "KSampler", "inputs": {"seed": seed, "control_after_generate": "fixed", "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": sched, "denoise": 1, "model": mo, "positive": ["50", 0], "negative": ["51", 0], "latent_image": ["52", 0]}},
        "54": {"class_type": "VAEDecode", "inputs": {"samples": ["53", 0], "vae": vo}},
        "55": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["54", 0]}},
    })
    return wf


# ── Generate ──

def generate(model_name, clip_model, vae_model, prompt, neg, loras, w, h, steps, cfg, sampler, sched, seed_input, rand_seed):
    try:
        return _generate(model_name, clip_model, vae_model, prompt, neg, loras, w, h, steps, cfg, sampler, sched, seed_input, rand_seed)
    except Exception as e:
        import traceback
        return None, f"Error: {e}\n{traceback.format_exc()}", render_gallery()


def _generate(model_name, clip_model, vae_model, prompt, neg, loras, w, h, steps, cfg, sampler, sched, seed_input, rand_seed):
    if not prompt or not prompt.strip():
        return None, "Enter a prompt.", render_gallery()
    if not model_name:
        return None, "Select a model.", render_gallery()

    w, h, steps = int(w), int(h), int(steps)
    cfg = float(cfg)
    try:
        seed = random.randint(0, 2**32 - 1) if rand_seed else int(seed_input)
    except (ValueError, TypeError):
        seed = random.randint(0, 2**32 - 1)

    slug = re.sub(r'[^a-zA-Z0-9-]', '-', prompt[:40]).strip('-')
    prefix = f"{time.strftime('%Y%m%d')}_{slug}"
    is_unet = model_name in get_comfy_models("UNETLoader", "unet_name")
    family = classify_model(model_name)
    preset = PRESETS.get(family, PRESETS["OTHER"])

    if is_unet:
        clip = clip_model or "qwen_3_4b.safetensors"
        vae = vae_model or "ae.safetensors"
        wf = build_unet_workflow(model_name, clip, vae, prompt, neg, loras, w, h, steps, cfg, sampler, sched, seed, prefix, preset[4] or "lumina2")
    else:
        wf = build_ckpt_workflow(model_name, prompt, neg, loras, w, h, steps, cfg, sampler, sched, seed, prefix, vae_model or None)

    try:
        result = api_post("/prompt", {"prompt": wf})
    except Exception as e:
        return None, f"Submit failed: {e}", render_gallery()

    pid = result.get("prompt_id")
    if not pid:
        return None, f"Error: {result.get('error', result)}", render_gallery()

    for _ in range(120):
        time.sleep(2)
        hist = api_get(f"/history/{pid}")
        if hist and pid in hist:
            for out in hist[pid].get("outputs", {}).values():
                if "images" in out:
                    for img in out["images"]:
                        url = f"{COMFYUI_URL}/view?filename={img['filename']}&type=output"
                        if img.get("subfolder"):
                            url += f"&subfolder={img['subfolder']}"
                        data = urllib.request.urlopen(url).read()
                        p = Path(f"/tmp/comfygen_{img['filename']}")
                        p.write_bytes(data)
                        return str(p), f"{img['filename']} | seed: {seed}", render_gallery()
            return None, "No images in output.", render_gallery()

    return None, "Timeout.", render_gallery()


# ── UI ──

css = """
#viewer { min-height: 600px; }
#gen-btn { height: 50px; font-size: 18px; }
#pull-btn { height: 42px; font-size: 16px; }
#prompt textarea, #neg textarea { resize: vertical !important; min-height: 40px; }
#lora-box textarea { resize: vertical !important; min-height: 30px; font-family: monospace; font-size: 13px; }
.cg-wrap { display:flex; flex-wrap:wrap; gap:4px; padding:4px; outline:none;
           overflow-y:auto; max-height:720px; align-content:flex-start; }
.cg-item { position:relative; cursor:pointer; border:2px solid transparent;
           border-radius:4px; overflow:hidden; width:calc(25% - 4px); aspect-ratio:1; }
.cg-item img { width:100%; height:100%; object-fit:cover; display:block; }
.cg-item:hover { border-color:#666; }
.cg-item.selected { border-color:#f97316; }
.cg-item.selected .cg-badge { display:block; }
.cg-badge { display:none; position:absolute; top:4px; right:4px; width:16px; height:16px;
            background:#f97316; border-radius:50%; border:2px solid #fff; }
.cg-empty { color:#888; text-align:center; padding:40px; width:100%; }
.cg-bar { display:flex; gap:8px; align-items:center; }
#hidden-area { display:none !important; }
"""


with gr.Blocks(title="ComfyGen") as app:
    with gr.Row():
        gr.Markdown("## ComfyGen")
        status_box = gr.Textbox(show_label=False, interactive=False, scale=3, max_lines=1)
        refresh_btn = gr.Button("Refresh", size="sm", scale=0)

    with gr.Row(equal_height=True):
        with gr.Column(scale=2, min_width=320):
            pull_btn = gr.Button("Pull from InvokeAI", variant="secondary", elem_id="pull-btn")
            prompt_box = gr.Textbox(label="Prompt", lines=3, placeholder="Describe your image...", elem_id="prompt")
            neg_box = gr.Textbox(label="Negative Prompt", lines=1, value="text, watermark, blurry", elem_id="neg")
            gr.Markdown("### Image")
            with gr.Row():
                width_sl = gr.Slider(label="Width", minimum=256, maximum=2048, value=1024, step=64)
                height_sl = gr.Slider(label="Height", minimum=256, maximum=2048, value=1024, step=64)
            with gr.Row():
                seed_num = gr.Number(label="Seed", value=42, precision=0, scale=2)
                random_seed = gr.Checkbox(label="Random", value=True, scale=1)
            gr.Markdown("### Generation")
            family_filter = gr.Radio(choices=list(FAMILIES.keys()), value="ALL", label="Model Family", interactive=True)
            model_dd = gr.Dropdown(label="Model", choices=[], interactive=True)
            lora_box = gr.Textbox(label="LoRAs (one per line: filename.safetensors:weight)", lines=2,
                                  placeholder="add-detail-xl.safetensors:0.75\nDetailedEyes_V3.safetensors:1.0", elem_id="lora-box")
            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    sampler_dd = gr.Dropdown(label="Sampler", choices=["euler", "euler_cfg_pp", "euler_ancestral", "dpmpp_2m",
                        "dpmpp_2m_sde", "dpmpp_sde", "res_multistep", "ddim", "uni_pc", "lcm"], value="res_multistep", interactive=True)
                    scheduler_dd = gr.Dropdown(label="Scheduler", choices=["simple", "normal", "karras", "exponential",
                        "sgm_uniform", "ddim_uniform"], value="simple", interactive=True)
                with gr.Row():
                    steps_sl = gr.Slider(label="Steps", minimum=1, maximum=100, value=8, step=1)
                    cfg_sl = gr.Slider(label="CFG", minimum=0, maximum=30, value=1, step=0.5)
                clip_dd = gr.Dropdown(label="CLIP", choices=[], interactive=True)
                vae_dd = gr.Dropdown(label="VAE", choices=[], interactive=True)
            gen_btn = gr.Button("Generate", variant="primary", elem_id="gen-btn")

        with gr.Column(scale=4, min_width=400):
            output_img = gr.Image(label="Viewer", type="filepath", elem_id="viewer", height=700)

        with gr.Column(scale=2, min_width=250):
            with gr.Row(elem_classes="cg-bar"):
                gr.Markdown("**Gallery**")
                sel_count = gr.HTML("<span id='sel-count'></span>")
            gallery_html = gr.HTML(value=render_gallery, elem_id="gallery-panel")

    with gr.Column(elem_id="hidden-area"):
        viewed_path = gr.Textbox(elem_id="viewed-path")
        selected_paths = gr.Textbox(elem_id="selected-paths")
        view_btn = gr.Button("v", elem_id="view-btn")
        load_btn = gr.Button("l", elem_id="load-btn")
        del_btn = gr.Button("d", elem_id="del-btn")


    view_btn.click(fn=lambda p: p if p and Path(p).exists() else gr.update(), inputs=[viewed_path], outputs=[output_img])

    load_btn.click(fn=load_settings_from_image, inputs=[viewed_path],
        outputs=[prompt_box, neg_box, width_sl, height_sl, steps_sl, cfg_sl, seed_num, random_seed,
                 sampler_dd, scheduler_dd, model_dd, lora_box, output_img, status_box])

    del_btn.click(fn=delete_images, inputs=[selected_paths], outputs=[gallery_html, status_box])

    pull_btn.click(fn=pull_from_invoke,
        outputs=[prompt_box, neg_box, width_sl, height_sl, steps_sl, cfg_sl, seed_num, random_seed,
                 sampler_dd, scheduler_dd, model_dd, lora_box, clip_dd, vae_dd, status_box])

    gen_btn.click(fn=generate,
        inputs=[model_dd, clip_dd, vae_dd, prompt_box, neg_box, lora_box, width_sl, height_sl,
                steps_sl, cfg_sl, sampler_dd, scheduler_dd, seed_num, random_seed],
        outputs=[output_img, status_box, gallery_html])

    def on_family_change(family):
        models = get_all_model_names() if family == "ALL" else [m for m in get_all_model_names() if classify_model(m) == family]
        preset = PRESETS.get(family, PRESETS["OTHER"])
        return (gr.update(choices=models, value=models[0] if models else None),
                gr.update(value=preset[0]), gr.update(value=preset[1]), gr.update(value=preset[2]), gr.update(value=preset[3]))

    family_filter.change(fn=on_family_change, inputs=[family_filter], outputs=[model_dd, steps_sl, cfg_sl, sampler_dd, scheduler_dd])

    def on_refresh():
        models = get_all_model_names()
        clips = get_comfy_models("CLIPLoader", "clip_name")
        vaes = get_comfy_models("VAELoader", "vae_name")
        return (gr.update(choices=models, value=models[0] if models else None),
                gr.update(choices=clips, value=clips[0] if clips else None),
                gr.update(choices=vaes, value=vaes[0] if vaes else None),
                f"{len(models)} models loaded")

    refresh_btn.click(fn=on_refresh, outputs=[model_dd, clip_dd, vae_dd, status_box])

    def on_load():
        for _ in range(15):
            if api_get("/system_stats"):
                break
            time.sleep(2)
        models = get_all_model_names() or []
        clips = get_comfy_models("CLIPLoader", "clip_name") or []
        vaes = get_comfy_models("VAELoader", "vae_name") or []
        return (gr.update(choices=models, value=models[0] if models else None),
                gr.update(choices=clips, value=clips[0] if clips else None),
                gr.update(choices=vaes, value=vaes[0] if vaes else None),
                f"{len(models)} models loaded", render_gallery())

    app.load(fn=on_load, outputs=[model_dd, clip_dd, vae_dd, status_box, gallery_html])


if __name__ == "__main__":
    app.launch(server_port=9200, server_name="0.0.0.0",
               theme=gr.themes.Base(primary_hue="orange"), css=css,
               allowed_paths=[str(OUTPUT_DIR), "/tmp"])
