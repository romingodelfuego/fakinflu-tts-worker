#!/usr/bin/env python3
"""Moteur Chatterbox Multilingual V3 (Resemble AI, MIT). Tourne dans /venv/chatterbox.

Voix : "clone" (ref_wav) ou "default" (voix integree conds.pt). Pas de "design".
Note : toute sortie porte le filigrane audio Perth de Resemble AI (inaudible).
"""
import copy, os, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # poids bakes dans l'image

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from common import seed_everything, serve

T3_MODEL = os.environ.get("CHATTERBOX_T3", "v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE, t3_model=T3_MODEL)
DEFAULT_CONDS = copy.deepcopy(model.conds)  # generate() ecrase model.conds a chaque clonage
_cache = {"ref": None}  # derniere reference clonee : evite de la re-encoder a chaque ligne

PARAMS = ("exaggeration", "cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p")


def process(req):
    t = time.time()
    voice = req.get("voice") or {"mode": "default"}
    mode = voice.get("mode", "default")
    params = {k: float(v) for k, v in (req.get("params") or {}).items() if k in PARAMS}
    exaggeration = params.get("exaggeration", 0.5)

    if mode == "clone":
        ref = voice["ref_wav_path"]
        if _cache["ref"] != ref:
            model.prepare_conditionals(ref, exaggeration=exaggeration)
            _cache["ref"] = ref
    elif mode == "default":
        model.conds = copy.deepcopy(DEFAULT_CONDS)
        _cache["ref"] = None
    else:
        raise ValueError(f"chatterbox : mode de voix non gere : {mode} (clone | default)")

    seed_everything(int(req.get("seed", 42)))
    wav = model.generate(req["text"], language_id=req["lang"], **params)
    ta.save(req["out_path"], wav, model.sr)
    return {"sr": model.sr, "duration_s": round(wav.shape[-1] / model.sr, 2),
            "seconds": round(time.time() - t, 2), "model": f"chatterbox-mtl-{T3_MODEL}"}


if __name__ == "__main__":
    serve(process)
