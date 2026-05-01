"""ComfyGen — Flask app with REST API"""

import random
import re
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from comfygen.comfyui import (
    api_get, api_post, download_output, get_all_model_names,
    get_comfy_models, is_online,
)
from comfygen.gallery import (
    OUTPUT_DIR, delete_files, get_gallery_items, load_settings_from_image,
)
from comfygen.models import (
    PRESETS, classify_model, get_filtered_loras, get_models_by_family,
    pick_default, pull_from_invoke,
)
from comfygen.workflows import build_ckpt_workflow, build_unet_workflow

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


# ── Model/LoRA endpoints ──

@app.route("/api/models")
def api_models():
    family = request.args.get("family", "ALL")
    models = get_models_by_family(family)
    preset = PRESETS.get(family, PRESETS["OTHER"])
    return jsonify({"models": models, "preset": preset})


@app.route("/api/loras")
def api_loras():
    family = request.args.get("family", "ALL")
    return jsonify({"loras": get_filtered_loras(family)})


@app.route("/api/clips")
def api_clips():
    clips = get_comfy_models("CLIPLoader", "clip_name")
    default = pick_default(clips, ["qwen_3_4b.safetensors", "qwen_3_4b_fp8_mixed.safetensors"])
    return jsonify({"clips": clips, "default": default})


@app.route("/api/vaes")
def api_vaes():
    vaes = get_comfy_models("VAELoader", "vae_name")
    default = pick_default(vaes, ["ae.safetensors"])
    return jsonify({"vaes": vaes, "default": default})


@app.route("/api/samplers")
def api_samplers():
    return jsonify({
        "samplers": [
            "euler", "euler_cfg_pp", "euler_ancestral", "dpmpp_2m",
            "dpmpp_2m_sde", "dpmpp_sde", "res_multistep", "ddim", "uni_pc", "lcm",
        ],
        "schedulers": [
            "simple", "normal", "karras", "exponential", "sgm_uniform", "ddim_uniform",
        ],
    })


@app.route("/api/families")
def api_families():
    return jsonify({"families": ["ALL", "Z-IMAGE", "FLUX", "SDXL", "SD1.5", "OTHER"]})


# ── Gallery endpoints ──

@app.route("/api/gallery")
def api_gallery():
    count = request.args.get("count", 80, type=int)
    return jsonify({"items": get_gallery_items(count)})


@app.route("/api/gallery/delete", methods=["POST"])
def api_gallery_delete():
    paths = request.json.get("paths", [])
    return jsonify(delete_files(paths))


@app.route("/api/gallery/load-settings", methods=["POST"])
def api_gallery_load_settings():
    path = request.json.get("path", "")
    return jsonify(load_settings_from_image(path))


@app.route("/api/media")
def api_media():
    path = request.args.get("path", "")
    if not path:
        return "Missing path", 400
    p = Path(path).resolve()
    if not p.exists():
        return "Not found", 404
    out_resolved = OUTPUT_DIR.resolve()
    tmp_path = Path("/tmp")
    if not (str(p).startswith(str(out_resolved)) or str(p).startswith(str(tmp_path))):
        return "Forbidden", 403
    return send_file(p)


# ── InvokeAI ──

@app.route("/api/pull-invoke", methods=["POST"])
def api_pull_invoke():
    return jsonify(pull_from_invoke())


# ── Generate ──

@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        data = request.json
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Enter a prompt."})
        model_name = data.get("model", "")
        if not model_name:
            return jsonify({"error": "Select a model."})

        neg = data.get("neg_prompt", "")
        loras = data.get("loras", "")
        w = int(data.get("width", 1024))
        h = int(data.get("height", 1024))
        steps = int(data.get("steps", 20))
        cfg = float(data.get("cfg", 7))
        sampler = data.get("sampler", "euler")
        sched = data.get("scheduler", "normal")
        clip_model = data.get("clip", "")
        vae_model = data.get("vae", "")

        rand_seed = data.get("random_seed", True)
        try:
            seed = random.randint(0, 2**32 - 1) if rand_seed else int(data.get("seed", 42))
        except (ValueError, TypeError):
            seed = random.randint(0, 2**32 - 1)

        slug = re.sub(r"[^a-zA-Z0-9-]", "-", prompt[:40]).strip("-")
        prefix = f"{time.strftime('%Y%m%d')}_{slug}"
        is_unet = model_name in get_comfy_models("UNETLoader", "unet_name")
        family = classify_model(model_name)
        preset = PRESETS.get(family, PRESETS["OTHER"])

        if is_unet:
            valid_clips = get_comfy_models("CLIPLoader", "clip_name")
            valid_vaes = get_comfy_models("VAELoader", "vae_name")
            preferred_clips = ["qwen_3_4b.safetensors", "qwen_3_4b_fp8_mixed.safetensors"]
            preferred_vaes = ["ae.safetensors"]
            clip = next((c for c in preferred_clips if c in valid_clips), clip_model or "qwen_3_4b.safetensors")
            vae = next((v for v in preferred_vaes if v in valid_vaes), vae_model or "ae.safetensors")
            clip_type = preset.get("clip_type", "lumina2")
            wf = build_unet_workflow(model_name, clip, vae, prompt, neg, loras, w, h, steps, cfg, sampler, sched, seed, prefix, clip_type)
        else:
            wf = build_ckpt_workflow(model_name, prompt, neg, loras, w, h, steps, cfg, sampler, sched, seed, prefix, vae_model or None)

        result = api_post("/prompt", {"prompt": wf})
        pid = result.get("prompt_id")
        if not pid:
            return jsonify({"error": f"ComfyUI error: {result.get('error', result)}"})

        for _ in range(120):
            time.sleep(2)
            hist = api_get(f"/history/{pid}")
            if hist and pid in hist:
                for out in hist[pid].get("outputs", {}).values():
                    if "images" in out:
                        for img in out["images"]:
                            img_data = download_output(img["filename"], img.get("subfolder"))
                            tmp = Path(f"/tmp/comfygen_{img['filename']}")
                            tmp.write_bytes(img_data)
                            return jsonify({
                                "path": str(tmp),
                                "filename": img["filename"],
                                "seed": seed,
                            })
                return jsonify({"error": "No images in output."})

        return jsonify({"error": "Timeout waiting for generation."})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})


# ── Status ──

@app.route("/api/status")
def api_status():
    online = is_online()
    model_count = len(get_all_model_names()) if online else 0
    return jsonify({"online": online, "model_count": model_count})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9200, debug=False)
