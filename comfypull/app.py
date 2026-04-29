import json
import os
import re
import sqlite3
from pathlib import Path

import requests
import streamlit as st

# ── Config ──
HUB_DB = "/mnt/llm/hub/hubmodels/hubrootv3.db"
HUB_MODELS = "/mnt/llm/hub/hubmodels/models"
FREE_AI_URL = "http://127.0.0.1:6500"
VALID_SUBFOLDERS = [
    "checkpoints", "loras", "vae", "text_encoders", "clip", "clip_vision",
    "controlnet", "diffusion_models", "unet", "upscale_models", "embeddings",
    "style_models", "hypernetworks", "audio_encoders", "latent_upscale_models",
    "model_patches", "photomaker", "diffusers", "configs", "gligen",
]


def get_comfyui_models_path():
    models_path = os.environ.get("COMFYMDIR")
    if not models_path:
        return None
    p = Path(models_path)
    return p if p.exists() else None


def extract_models_with_ai(text):
    """Send text to free-ai /structured endpoint to extract model filenames."""
    prompt = f"""Extract all AI model filenames from the text below. For each model, determine which ComfyUI models subfolder it belongs in.

Return ONLY a JSON array. Each element must have "filename" and "subfolder" keys.
Valid subfolders: {', '.join(VALID_SUBFOLDERS)}

Rules:
- Only extract filenames that end in .safetensors or .gguf
- If the subfolder is obvious from context (e.g. "diffusion_models" section header, or "lora_name:" field), use that
- If unclear, infer from the model name (e.g. names with "lora" go in "loras", names with "vae" go in "vae", names with "clip" or "t5" go in "text_encoders", names with "controlnet" go in "controlnet")
- If you truly cannot determine the subfolder, use "checkpoints" as default
- Do not invent filenames that aren't in the text

Text:
{text}"""

    try:
        resp = requests.post(
            f"{FREE_AI_URL}/structured",
            json={"prompt": prompt, "system": "Respond with ONLY valid JSON. No markdown, no explanation."},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("data") or data.get("content", "")
        if isinstance(content, str):
            content = content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = re.sub(r"^```\w*\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            content = json.loads(content)

        if isinstance(content, list):
            return content
        if isinstance(content, dict) and "models" in content:
            return content["models"]
        return []
    except Exception as e:
        st.error(f"AI extraction failed: {e}")
        return []


def search_hub_for_model(filename):
    """Search hub database for a model. Returns list of (model_id, full_path, db_filename) tuples."""
    try:
        with sqlite3.connect(HUB_DB) as conn:
            cursor = conn.cursor()

            # Tier 1: Exact match
            cursor.execute(
                "SELECT id, hash_blake3, filename FROM models WHERE filename = ? AND deleted = 0",
                (filename,),
            )
            results = cursor.fetchall()
            if results:
                matches = []
                for model_id, hash_blake3, db_filename in results:
                    model_path = Path(HUB_MODELS) / hash_blake3 / db_filename
                    if model_path.exists():
                        matches.append((model_id, model_path, db_filename))
                if matches:
                    return matches

            # Tier 2: Fuzzy match
            stem = Path(filename).stem
            ext = Path(filename).suffix
            parts = re.split(r"[-_]+", stem)
            parts = [p for p in parts if len(p) > 2]
            if not parts:
                return []

            like_pattern = "%" + "%".join(parts) + "%" + ext
            cursor.execute(
                "SELECT id, hash_blake3, filename FROM models WHERE filename LIKE ? AND deleted = 0",
                (like_pattern,),
            )
            results = cursor.fetchall()
            if results:
                matches = []
                for model_id, hash_blake3, db_filename in results:
                    model_path = Path(HUB_MODELS) / hash_blake3 / db_filename
                    if model_path.exists():
                        matches.append((model_id, model_path, db_filename))
                if matches:
                    return matches

            # Tier 3: Broad fuzzy
            longest_part = max(parts, key=len)
            broad_pattern = f"%{longest_part}%{ext}"
            cursor.execute(
                "SELECT id, hash_blake3, filename FROM models WHERE filename LIKE ? AND deleted = 0",
                (broad_pattern,),
            )
            results = cursor.fetchall()
            if results:
                matches = []
                for model_id, hash_blake3, db_filename in results:
                    model_path = Path(HUB_MODELS) / hash_blake3 / db_filename
                    if model_path.exists():
                        matches.append((model_id, model_path, db_filename))
                if matches:
                    return matches

            return []
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return []


def create_symlink(source, target_dir, filename):
    """Create symlink from source to target_dir/filename. Returns (success, message)."""
    target = target_dir / filename

    if target.exists() or target.is_symlink():
        if target.is_symlink():
            existing_target = target.resolve()
            if existing_target == source:
                return True, "Already linked"
            else:
                return False, "Different source already linked"
        else:
            return False, "File exists (not a symlink)"

    try:
        target.symlink_to(source)
        return True, "Linked"
    except OSError as e:
        return False, f"Symlink failed: {e}"


# ── Streamlit App ──
st.set_page_config(page_title="ComfyPull", layout="wide")
st.title("ComfyPull")

comfyui_models = get_comfyui_models_path()
if not comfyui_models:
    st.error("COMFYMDIR environment variable not set or directory doesn't exist.")
    st.stop()

st.caption(f"Hub: {HUB_DB}  |  Models: {comfyui_models}")

text_input = st.text_area(
    "Paste text containing model names",
    height=200,
    placeholder="Paste ComfyUI missing models dialog, error messages, or any text containing model filenames...",
)

if st.button("Pull", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Paste some text first.")
        st.stop()

    # Step 1: AI extraction
    with st.spinner("AI extracting model names..."):
        models = extract_models_with_ai(text_input)

    if not models:
        st.warning("No models found in the text.")
        st.stop()

    st.info(f"Found {len(models)} model(s)")

    # Step 2: Hub lookup and symlink
    results = []
    for model in models:
        filename = model.get("filename", "")
        subfolder = model.get("subfolder", "checkpoints")

        if not filename:
            continue

        # Search hub
        matches = search_hub_for_model(filename)

        if not matches:
            results.append({
                "Model": filename,
                "Subfolder": subfolder,
                "Hub Match": "",
                "Status": "Not in hub",
            })
            continue

        for model_id, source_path, db_filename in matches:
            is_fuzzy = db_filename != filename
            target_dir = comfyui_models / subfolder
            target_dir.mkdir(parents=True, exist_ok=True)

            success, message = create_symlink(source_path, target_dir, db_filename)

            results.append({
                "Model": filename,
                "Subfolder": subfolder,
                "Hub Match": db_filename if is_fuzzy else filename,
                "Status": message,
            })

    # Step 3: Display results
    if results:
        linked = sum(1 for r in results if r["Status"] in ("Linked", "Already linked"))
        missing = sum(1 for r in results if r["Status"] == "Not in hub")
        errors = sum(1 for r in results if r["Status"] not in ("Linked", "Already linked", "Not in hub"))

        col1, col2, col3 = st.columns(3)
        col1.metric("Linked", linked)
        col2.metric("Missing", missing)
        col3.metric("Errors", errors)

        st.dataframe(results, use_container_width=True, hide_index=True, height=800)
