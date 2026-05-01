"""Model, LoRA, hub DB, and InvokeAI state management"""

import json
import os
import sqlite3
from pathlib import Path

from comfygen.comfyui import get_all_model_names, get_comfy_models

INVOKEAI_DB = "/mnt/llm/hub/invokeai_data/databases/invokeai.db"
HUBMODELS_DB = "/mnt/llm/hub/hubmodels/hubrootv3.db"
HUBMODELS_DIR = "/mnt/llm/hub/hubmodels/models"
COMFYMDIR = os.environ.get("COMFYMDIR", "/home/dev/comfy/ComfyUI/models")

PRESETS = {
    "Z-IMAGE": {"steps": 8, "cfg": 1, "sampler": "res_multistep", "scheduler": "simple", "clip_type": "lumina2", "is_unet": True},
    "FLUX": {"steps": 20, "cfg": 1, "sampler": "euler", "scheduler": "simple"},
    "SDXL": {"steps": 20, "cfg": 7, "sampler": "euler", "scheduler": "normal"},
    "SD1.5": {"steps": 20, "cfg": 7, "sampler": "euler", "scheduler": "normal"},
    "OTHER": {"steps": 20, "cfg": 7, "sampler": "euler", "scheduler": "normal"},
}

FAMILIES = {
    "ALL": None,
    "Z-IMAGE": lambda n: "zimage" in n or "z_image" in n or "z-image" in n or "z_img" in n,
    "FLUX": lambda n: "flux" in n,
    "SDXL": lambda n: any(x in n for x in ["sdxl", "xl", "juggernaut", "pony", "illustrious", "animagine"]),
    "SD1.5": lambda n: any(x in n for x in ["sd15", "sd_1", "v1-5", "v1_5", "dreamshaper", "realistic"]),
    "OTHER": lambda n: True,
}

FAMILY_TO_HUB_BASES = {
    "Z-IMAGE": ["zimageturbo", "z-image", "zimagebase", "qwen", "qwen3"],
    "FLUX": ["flux.1 d", "flux.1 s", "flux.2 d", "flux.2 klein 4b", "flux.2 klein 9b"],
    "SDXL": ["sdxl 1.0", "pony", "illustrious", "noobai"],
    "SD1.5": ["sd 1.5"],
    "ALL": None,
    "OTHER": None,
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


def get_models_by_family(family="ALL"):
    all_models = get_all_model_names()
    if family == "ALL":
        return all_models
    return [m for m in all_models if classify_model(m) == family]


def get_filtered_loras(family="ALL"):
    comfy_loras = get_comfy_models("LoraLoader", "lora_name")
    if not comfy_loras:
        return []
    if family == "ALL" or family not in FAMILY_TO_HUB_BASES or FAMILY_TO_HUB_BASES[family] is None:
        return sorted(comfy_loras)
    hub_bases = FAMILY_TO_HUB_BASES[family]
    try:
        with sqlite3.connect(HUBMODELS_DB) as conn:
            placeholders = ",".join("?" * len(hub_bases))
            rows = conn.execute(
                f"SELECT filename FROM models WHERE model_type='lora' AND base_model IN ({placeholders}) AND deleted=0",
                hub_bases,
            ).fetchall()
            hub_filenames = {row[0] for row in rows}
        return sorted([l for l in comfy_loras if l in hub_filenames])
    except Exception:
        return sorted(comfy_loras)


def pick_default(items, preferred):
    for p in preferred:
        if p in items:
            return p
    return items[0] if items else None


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
                cursor.execute(
                    "SELECT hash_blake3, filename FROM models WHERE hash_blake3 = ? AND deleted = 0",
                    (hex_hash,),
                )
                row = cursor.fetchone()
                if row and symlink_from_hub(row[0], row[1], subfolder):
                    return row[1], subfolder
            cursor.execute(
                "SELECT hash_blake3, filename FROM models WHERE filename LIKE ? AND deleted = 0 LIMIT 5",
                (f"%{invoke_name}%",),
            )
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

SCHEDULER_MAP = {
    "euler": "normal", "euler_k": "karras", "euler_a": "normal",
    "dpmpp_2m": "normal", "dpmpp_2m_k": "karras", "dpmpp_2m_sde": "karras",
    "dpmpp_sde": "karras", "ddim": "ddim_uniform", "lcm": "simple",
    "unipc": "normal", "kdpm_2": "karras", "heun": "normal",
}

SAMPLER_MAP = {
    "euler": "euler", "euler_k": "euler", "euler_a": "euler_ancestral",
    "dpmpp_2m": "dpmpp_2m", "dpmpp_2m_k": "dpmpp_2m", "dpmpp_2m_sde": "dpmpp_2m_sde",
    "dpmpp_sde": "dpmpp_sde", "ddim": "ddim", "lcm": "lcm", "unipc": "uni_pc",
    "kdpm_2": "dpm_2", "heun": "heun",
}


def pull_from_invoke():
    try:
        with sqlite3.connect(INVOKEAI_DB) as conn:
            row = conn.execute("SELECT value FROM client_state WHERE user_id='system' AND key='params'").fetchone()
            params = json.loads(row[0]) if row else {}
            row = conn.execute("SELECT value FROM client_state WHERE user_id='system' AND key='loras'").fetchone()
            loras_state = json.loads(row[0]) if row else {}
    except Exception:
        return {"error": "Failed to read InvokeAI state"}

    if not params:
        return {"error": "No params found in InvokeAI state"}

    invoke_scheduler = params.get("scheduler", "euler")
    invoke_model = params.get("model", {})
    comfy_model, model_type = map_invoke_model_to_comfy(
        invoke_model.get("name", ""), invoke_model.get("base", ""),
        invoke_model.get("hash", ""), invoke_model.get("type", "main"),
    )

    lora_lines = []
    for entry in loras_state.get("loras", []):
        if not entry.get("isEnabled", False):
            continue
        lm = entry.get("model", {})
        cl = map_invoke_lora_to_comfy(lm.get("name", ""), lm.get("hash", ""))
        w = entry.get("weight", 1.0)
        lora_lines.append(f"{cl}:{w}" if cl else f"?{lm.get('name', '')}:{w}")

    return {
        "prompt": params.get("positivePrompt", ""),
        "neg_prompt": params.get("negativePrompt", ""),
        "width": params.get("dimensions", {}).get("width", 1024),
        "height": params.get("dimensions", {}).get("height", 1024),
        "steps": params.get("steps", 20),
        "cfg": params.get("cfgScale", 7),
        "seed": params.get("seed", 42),
        "random_seed": params.get("shouldRandomizeSeed", True),
        "sampler": SAMPLER_MAP.get(invoke_scheduler, "euler"),
        "scheduler": SCHEDULER_MAP.get(invoke_scheduler, "normal"),
        "model": comfy_model,
        "model_type": model_type,
        "loras": "\n".join(lora_lines) if lora_lines else "",
        "status": f"Pulled: {invoke_model.get('name', '')} -> {comfy_model or '?'}",
    }
