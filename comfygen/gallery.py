"""Gallery file management and PNG metadata reading"""

import json
import os
from pathlib import Path

from comfygen.comfyui import get_all_model_names

OUTPUT_DIR = Path(os.environ.get("COMFYUI_OUTPUT_ROOT", "/home/dev/comfy/ComfyUI/output"))

MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def get_gallery_items(count=80):
    if not OUTPUT_DIR.exists():
        return []
    items = []
    for f in OUTPUT_DIR.iterdir():
        if f.suffix.lower() in MEDIA_EXTENSIONS and f.is_file():
            items.append(f)
    for subdir in OUTPUT_DIR.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.suffix.lower() in MEDIA_EXTENSIONS and f.is_file():
                    items.append(f)
    items.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    result = []
    for f in items[:count]:
        is_video = f.suffix.lower() in {".mp4", ".webm"}
        result.append({
            "path": str(f),
            "name": f.name,
            "is_video": is_video,
            "mtime": f.stat().st_mtime,
        })
    return result


def delete_files(paths):
    deleted = 0
    errors = []
    for p in paths:
        try:
            fp = Path(p)
            if fp.exists() and str(fp).startswith(str(OUTPUT_DIR)):
                fp.unlink()
                deleted += 1
        except Exception as e:
            errors.append(str(e))
    return {"deleted": deleted, "errors": errors}


def load_settings_from_image(image_path):
    if not image_path or not Path(image_path).exists():
        return {"error": "No image selected"}
    try:
        from PIL import Image
        img = Image.open(image_path)
        if "prompt" not in img.info:
            return {"error": f"No metadata in {Path(image_path).name}"}
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
                steps = inp.get("steps", 20)
                cfg = inp.get("cfg", 7)
                seed = inp.get("seed", 42)
                sampler = inp.get("sampler_name", "euler")
                scheduler = inp.get("scheduler", "normal")
            elif ct in ("EmptyLatentImage", "EmptySD3LatentImage"):
                width = inp.get("width", 1024)
                height = inp.get("height", 1024)

        if clip_nodes:
            prompt = clip_nodes[0][1]
            if len(clip_nodes) > 1:
                neg_prompt = clip_nodes[1][1]

        return {
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "cfg": float(cfg),
            "seed": int(seed),
            "sampler": sampler,
            "scheduler": scheduler,
            "model": model_name,
            "loras": "\n".join(lora_lines),
            "filename": Path(image_path).name,
        }
    except Exception as e:
        return {"error": str(e)}
