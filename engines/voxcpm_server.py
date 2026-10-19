#!/usr/bin/env python3
"""Moteur VoxCPM2 (OpenBMB, Apache-2.0, 2B, 30 langues dont le francais, 48 kHz). Tourne dans /venv/voxcpm.

Voix : "clone"  (ref_wav ; + ref_text -> clonage "ultime" en continuation, plus fidele)
       "design" (description en langage naturel, sans audio de reference).
Jeu  : "style" (champ de l'item, ex. "voix basse, amere, au bord des larmes") est place entre
       parentheses en tete du texte, comme le prevoit VoxCPM2 : il dirige le ton sans changer
       le timbre clone. Avec un style, le clonage passe en mode "controlable" (reference seule),
       car la continuation (prompt_wav + prompt_text) prolonge le ton de l'extrait de reference.
"""
import os, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # poids bakes dans l'image

import soundfile as sf
from voxcpm import VoxCPM

from common import seed_everything, serve

MODEL_ID = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")
PARAMS = {"cfg_value": float, "inference_timesteps": int}

# optimize=False : pas de torch.compile (compilation longue au cold start, triton requis)
model = VoxCPM.from_pretrained(MODEL_ID, load_denoiser=False, optimize=False, local_files_only=True)
SR = model.tts_model.sample_rate


def process(req):
    t = time.time()
    voice = req.get("voice") or {"mode": "design"}
    mode = voice.get("mode", "design")
    params = {k: PARAMS[k](v) for k, v in (req.get("params") or {}).items() if k in PARAMS}
    style = (req.get("style") or "").strip()
    text = req["text"]
    kw = {}

    if mode == "clone":
        kw["reference_wav_path"] = voice["ref_wav_path"]
        ref_text = (voice.get("ref_text") or "").strip()
        if ref_text and not style:
            kw.update(prompt_wav_path=voice["ref_wav_path"], prompt_text=ref_text)
        if style:
            text = f"({style}){text}"
    elif mode == "design":
        desc = ", ".join(s for s in ((voice.get("instruct") or "").strip(), style) if s)
        if desc:
            text = f"({desc}){text}"
    else:
        raise ValueError(f"voxcpm : mode de voix non gere : {mode} (clone | design)")

    seed_everything(int(req.get("seed", 42)))
    wav = model.generate(text=text, **kw, **params)
    sf.write(req["out_path"], wav, SR)
    variant = ("clone+style" if style else "clone-ultime" if "prompt_text" in kw else "clone") \
        if mode == "clone" else "design"
    return {"sr": SR, "duration_s": round(len(wav) / SR, 2), "seconds": round(time.time() - t, 2),
            "model": f"VoxCPM2 ({variant})", "text_sent": text}


if __name__ == "__main__":
    serve(process)
