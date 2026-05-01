"""ComfyUI API client"""

import json
import urllib.request

COMFYUI_URL = "http://localhost:8188"


def api_get(endpoint):
    try:
        return json.loads(
            urllib.request.urlopen(f"{COMFYUI_URL}{endpoint}", timeout=10).read()
        )
    except Exception:
        return None


def api_post(endpoint, data):
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}{endpoint}",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
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
    return get_comfy_models("UNETLoader", "unet_name") + get_comfy_models(
        "CheckpointLoaderSimple", "ckpt_name"
    )


def is_online():
    return api_get("/system_stats") is not None


def download_output(filename, subfolder=None):
    url = f"{COMFYUI_URL}/view?filename={filename}&type=output"
    if subfolder:
        url += f"&subfolder={subfolder}"
    return urllib.request.urlopen(url).read()
