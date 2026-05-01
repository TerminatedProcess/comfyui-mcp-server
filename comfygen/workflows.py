"""ComfyUI workflow builders"""


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
        wf[str(nid)] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": name,
                "strength_model": weight,
                "strength_clip": weight,
                "model": model_out,
                "clip": clip_out,
            },
        }
        model_out, clip_out = [str(nid), 0], [str(nid), 1]
        nid += 1
    return model_out, clip_out, nid


def build_unet_workflow(
    model, clip, vae, prompt, neg, loras, w, h, steps, cfg,
    sampler, sched, seed, prefix, clip_type="lumina2",
):
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
        "53": {"class_type": "KSampler", "inputs": {
            "seed": seed, "control_after_generate": "fixed", "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": sched, "denoise": 1,
            "model": mo, "positive": ["50", 0], "negative": ["51", 0], "latent_image": ["52", 0],
        }},
        "54": {"class_type": "VAEDecode", "inputs": {"samples": ["53", 0], "vae": ["3", 0]}},
        "55": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["54", 0]}},
    })
    return wf


def build_ckpt_workflow(
    model, prompt, neg, loras, w, h, steps, cfg,
    sampler, sched, seed, prefix, vae_override=None,
):
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
        "53": {"class_type": "KSampler", "inputs": {
            "seed": seed, "control_after_generate": "fixed", "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": sched, "denoise": 1,
            "model": mo, "positive": ["50", 0], "negative": ["51", 0], "latent_image": ["52", 0],
        }},
        "54": {"class_type": "VAEDecode", "inputs": {"samples": ["53", 0], "vae": vo}},
        "55": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["54", 0]}},
    })
    return wf
