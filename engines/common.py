"""Protocole commun aux serveurs de moteurs TTS (un process par venv).

Le handler RunPod (python systeme) parle a chaque moteur par stdin/stdout, une
requete JSON par ligne. Les reponses sont prefixees par RESULT_TAG pour ne pas
se melanger aux logs des bibliotheques, qui partent sur stderr ou stdout sans tag.
"""
import json, random, sys, traceback

RESULT_TAG = "@@TTS@@ "


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serve(process):
    """Boucle : lit une requete, appelle process(req) -> dict, ecrit la reponse."""
    out = sys.stdout
    sys.stdout = sys.stderr  # tout print() des bibliotheques part dans les logs
    out.write(RESULT_TAG + json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            res = process(json.loads(line))
        except Exception as e:
            traceback.print_exc()
            res = {"error": f"{type(e).__name__}: {e}"}
        out.write(RESULT_TAG + json.dumps(res) + "\n")
        out.flush()
