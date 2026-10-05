#!/usr/bin/env python3
"""Moteur Qwen3-TTS 12Hz 1.7B (Apache-2.0). Tourne dans /venv/qwen.

Voix : "clone" (Base : ref_wav + ref_text, ou x-vector seul sans ref_text)
       "design" (VoiceDesign : description en langage naturel, ex. "jeune femme de 25 ans").
Chaque modele est charge a la premiere requete qui en a besoin.
"""
import glob, os, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # poids bakes dans l'image

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from common import seed_everything, serve

MODELS = {
    "clone": os.environ.get("QWEN_TTS_BASE", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    "design": os.environ.get("QWEN_TTS_DESIGN", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
}
LANGS = {"fr": "French", "en": "English", "de": "German", "es": "Spanish", "it": "Italian",
         "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}
GEN_PARAMS = ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")

_models = {}
_prompts = {}  # (ref_wav, ref_text) -> voice_clone_prompt reutilisable


def _local(repo):
    """Dossier du snapshot bake. qwen_tts interroge l'API HF quand on lui passe un id de
    depot, meme hors ligne ; un chemin local court-circuite tout appel reseau."""
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snaps = sorted(glob.glob(os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", "*")))
    return snaps[-1] if snaps else repo


def _model(kind):
    if kind not in _models:
        _models[kind] = Qwen3TTSModel.from_pretrained(
            _local(MODELS[kind]), device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16, attn_implementation="sdpa")
    return _models[kind]


def process(req):
    t = time.time()
    voice = req.get("voice") or {}
    mode = voice.get("mode", "clone")
    lang = LANGS.get(req["lang"], req["lang"])
    gen = {k: v for k, v in (req.get("params") or {}).items() if k in GEN_PARAMS}
    seed_everything(int(req.get("seed", 42)))

    if mode == "clone":
        m = _model("clone")
        key = (voice["ref_wav_path"], voice.get("ref_text") or "")
        if key not in _prompts:
            _prompts[key] = m.create_voice_clone_prompt(
                ref_audio=key[0], ref_text=key[1] or None, x_vector_only_mode=not key[1])
        wavs, sr = m.generate_voice_clone(text=req["text"], language=lang,
                                          voice_clone_prompt=_prompts[key], **gen)
    elif mode == "design":
        m = _model("design")
        wavs, sr = m.generate_voice_design(text=req["text"], language=lang,
                                           instruct=voice["instruct"], **gen)
    else:
        raise ValueError(f"qwen : mode de voix non gere : {mode} (clone | design)")

    sf.write(req["out_path"], wavs[0], sr)
    return {"sr": sr, "duration_s": round(len(wavs[0]) / sr, 2),
            "seconds": round(time.time() - t, 2), "model": os.path.basename(MODELS[mode])}


if __name__ == "__main__":
    serve(process)
