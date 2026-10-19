# Worker RunPod : tts

Texte → voix, multilingue (français obligatoire), clonage de voix optionnel. Trois moteurs dans
la même image, pour le bench :

| Moteur | Modèle | Licence | Modes de voix |
|---|---|---|---|
| `chatterbox` | Chatterbox Multilingual **V3** (Resemble AI), 23 langues | MIT | `clone`, `default` |
| `qwen` | Qwen3-TTS 12Hz **1.7B** Base + VoiceDesign, 10 langues | Apache-2.0 | `clone`, `design` |
| `voxcpm` | **VoxCPM2** (OpenBMB, 2B, 48 kHz), 30 langues | Apache-2.0 | `clone`, `design` |

**Direction de jeu** : le champ `style` d'un item (« voix basse, amère, au bord des larmes ») est
appliqué par `voxcpm` (instruction entre parenthèses en tête du texte, timbre cloné conservé) et par
`qwen` en mode `design` (ajouté à la description). Chatterbox l'ignore.

Higgs TTS 3 (balises d'émotion en ligne) vit dans un worker séparé (pile vLLM-Omni) :
`tts-higgs/serverless/`.

Dépôt publié : https://github.com/romingodelfuego/fakinflu-tts-worker. Ce dépôt est un
**miroir** : la source fait foi dans le dossier `tts/serverless/` de fakinflu.

## Architecture

Les bibliothèques exigent des versions incompatibles de `transformers` (5.2 contre 4.57…).
Chaque moteur a donc son venv (`/venv/chatterbox`, `/venv/qwen`, `/venv/voxcpm`) et tourne comme
sous-process persistant (`engines/*_server.py`, une requête JSON par ligne). Le handler
(python système + `runpod`) lance un moteur à sa première requête et le garde chaud
tant que le worker vit. **Un seul moteur occupe le GPU à la fois** : en lancer un arrête les
autres. Regrouper les items par moteur dans un job évite les rechargements.

Chatterbox ajoute à toute sortie le filigrane audio inaudible **Perth** de Resemble AI.

## Contrat d'API

```json
{"input": {
  "voices": {"fr_f": {"ref_audio": "<b64 wav, 10-20 s>", "ref_text": "transcription exacte"}},
  "items": [
    {"id": "1", "engine": "qwen", "lang": "fr", "text": "...", "voice": {"mode": "clone", "ref": "fr_f"}},
    {"id": "2", "engine": "qwen", "lang": "fr", "text": "...", "voice": {"mode": "design", "instruct": "jeune femme de 25 ans, voix posée"}},
    {"id": "3", "engine": "chatterbox", "lang": "fr", "text": "...", "voice": {"mode": "clone", "ref": "fr_f"},
     "params": {"exaggeration": 0.5, "cfg_weight": 0.5}, "seed": 42},
    {"id": "4", "engine": "voxcpm", "lang": "fr", "text": "...", "voice": {"mode": "clone", "ref": "fr_f"},
     "style": "voix basse, amère, qui tremble"}
  ]}}
→ {"results": [{"id": "1", "audio": "<b64 wav>", "sr": 24000, "duration_s": 3.2, "seconds": 2.1, "model": "...", "engine": "qwen"},
               {"id": "2", "error": "..."}]}
```

- Les voix de référence sont envoyées **une fois par job** dans `voices` et partagées par les items.
- `ref_text` vide : Qwen clone sur l'empreinte vocale seule (x-vector), avec une qualité moindre.
- `params` Chatterbox : `exaggeration`, `cfg_weight`, `temperature`, `repetition_penalty`, `min_p`, `top_p`.
  `params` Qwen : `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_new_tokens`.
  `params` VoxCPM2 : `cfg_value` (2.0), `inference_timesteps` (10).
- VoxCPM2 en `clone` : avec `ref_text` et sans `style`, clonage « ultime » (continuation de
  l'extrait) ; avec un `style`, clonage contrôlable (référence seule + instruction).
- `{"input": {"ping": true}}` répond sans charger de modèle.

## Cycle de vie

```bash
make worker-status W=tts              # local vs dernière release (0 crédit)
make worker-release W=tts DRY=1       # ce qui partirait, sans rien publier
make worker-release W=tts BUMP=minor NOTES="..."   # commit + tag vX.Y.Z + GitHub Release → rebuild RunPod
```

**Première release seulement** : il faut lier le dépôt dans la console RunPod
(**Serverless → New Endpoint → Import Git Repository**). Appliquer ensuite les réglages
de `worker.json` (GPU, workers, timeouts) et noter l'ID dans `.env` → `RUNPOD_TTS_ENDPOINT_ID`.
Claude peut faire tout ça à ta place (skill `deploy-worker`).

Test local du handler (sur une machine GPU) : `python handler.py --test_input "$(cat test_input.json)"`.
