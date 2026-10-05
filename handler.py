#!/usr/bin/env python3
"""fakinflu — worker RunPod : tts (Chatterbox Multilingual V3 + Qwen3-TTS 1.7B).

Entree :
  {"voices": {"<voice_id>": {"ref_audio": "<b64 wav>", "ref_text": "..."}},   # optionnel
   "items": [{"id": "...", "engine": "chatterbox|qwen", "text": "...", "lang": "fr|en|...",
              "voice": {"mode": "clone", "ref": "<voice_id>"}         # les deux moteurs
                     | {"mode": "design", "instruct": "..."}          # qwen seulement
                     | {"mode": "default"},                           # chatterbox seulement
              "params": {...}, "seed": 42}]}
Sortie : {"results": [{"id", "audio": "<b64 wav>", "sr", "duration_s", "seconds", "model"} | {"id", "error"}]}

Les deux moteurs exigent des versions de transformers incompatibles (5.2 vs 4.57) :
chacun vit dans son venv et tourne comme sous-process persistant (engines/*_server.py),
lance a la premiere requete puis garde chaud tant que le worker vit. Une voix de
reference est envoyee UNE fois par job (bloc "voices") et partagee par tous les items.
"""
import base64, json, os, subprocess, tempfile, time, traceback

import runpod

ENGINES = {
    "chatterbox": ("/venv/chatterbox/bin/python", "/app/engines/chatterbox_server.py"),
    "qwen": ("/venv/qwen/bin/python", "/app/engines/qwen_server.py"),
}
RESULT_TAG = "@@TTS@@ "
_procs = {}


def _engine(name):
    p = _procs.get(name)
    if p is not None and p.poll() is None:
        return p
    if name not in ENGINES:
        raise ValueError(f"moteur inconnu : {name} (connus : {', '.join(ENGINES)})")
    py, script = ENGINES[name]
    t = time.time()
    p = subprocess.Popen([py, "-u", script], cwd=os.path.dirname(script), text=True,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    _read(p, name)  # attend {"ready": true} : modele charge
    print(f"[tts] moteur {name} pret en {time.time() - t:.1f}s", flush=True)
    _procs[name] = p
    return p


def _read(p, name):
    for line in p.stdout:
        if line.startswith(RESULT_TAG):
            return json.loads(line[len(RESULT_TAG):])
        print(line, end="", flush=True)  # sortie non taguee : log
    raise RuntimeError(f"moteur {name} arrete (code {p.wait()}) : voir les logs du worker")


def _call(name, req):
    p = _engine(name)
    p.stdin.write(json.dumps(req) + "\n")
    p.stdin.flush()
    return _read(p, name)


def _b64_to_file(b64, path):
    if b64.strip().startswith("data:") and "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def _process(item, voices, tmp):
    engine = item.get("engine", "chatterbox")
    if not (item.get("text") or "").strip():
        raise ValueError("champ 'text' vide")
    voice = dict(item.get("voice") or {"mode": "default" if engine == "chatterbox" else "design"})
    if voice.get("mode") == "clone":
        ref = voice.get("ref")
        if ref not in voices:
            raise ValueError(f"voix de reference inconnue : {ref!r} (bloc 'voices' du job)")
        voice["ref_wav_path"] = voices[ref]["path"]
        voice.setdefault("ref_text", voices[ref].get("ref_text") or "")
    out_path = os.path.join(tmp, f"out_{len(os.listdir(tmp))}.wav")
    res = _call(engine, {"text": item["text"], "lang": item.get("lang", "fr"), "voice": voice,
                         "params": item.get("params") or {}, "seed": item.get("seed", 42),
                         "out_path": out_path})
    if "error" in res:
        return res
    with open(out_path, "rb") as f:
        res["audio"] = base64.b64encode(f.read()).decode()
    res["engine"] = engine
    return res


def handler(job):
    inp = job.get("input", {}) or {}
    if inp.get("ping"):
        return {"pong": True, "engines": list(ENGINES)}
    items = inp.get("items") or [inp]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        voices = {}
        for vid, v in (inp.get("voices") or {}).items():
            path = os.path.join(tmp, f"ref_{len(voices)}.wav")
            _b64_to_file(v["ref_audio"], path)
            voices[vid] = {"path": path, "ref_text": v.get("ref_text") or ""}
        for item in items:
            res = {"id": item.get("id")}
            try:
                res.update(_process(item, voices, tmp))
            except Exception as e:
                traceback.print_exc()
                res["error"] = f"{type(e).__name__}: {e}"
            results.append(res)
    return {"results": results}


runpod.serverless.start({"handler": handler})
