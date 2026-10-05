# fakinflu — worker RunPod : tts (Chatterbox Multilingual V3 + Qwen3-TTS 1.7B, poids BAKES)
# Build par RunPod (integration GitHub) a chaque GitHub Release : make worker-release W=tts
#
# Deux venvs : chatterbox-tts epingle transformers==5.2.0, qwen-tts transformers==4.57.3.
# Le handler (python systeme + runpod) pilote chaque moteur comme sous-process de son venv.
ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv git ca-certificates ffmpeg sox libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && pip3 install "runpod>=1.7,<2"

# --- Chatterbox Multilingual V3 : depot GitHub epingle (la V3 n'est pas encore sur PyPI) ---
ARG CHATTERBOX_REF=5de7a54aa4e5e2baadb0182dde554908b48b85c2
RUN python3 -m venv /venv/chatterbox \
 && /venv/chatterbox/bin/pip install --upgrade pip setuptools wheel \
 && /venv/chatterbox/bin/pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
 && /venv/chatterbox/bin/pip install "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@${CHATTERBOX_REF}"

# --- Qwen3-TTS ---
RUN python3 -m venv /venv/qwen \
 && /venv/qwen/bin/pip install --upgrade pip setuptools wheel \
 && /venv/qwen/bin/pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
 && /venv/qwen/bin/pip install qwen-tts==0.1.1 soundfile

# --- Poids BAKES (~12 Go) : aucun telechargement au cold start ---
RUN /venv/chatterbox/bin/python -c "from huggingface_hub import snapshot_download as d; \
d('ResembleAI/chatterbox', allow_patterns=['ve.pt', 't3_mtl23ls_v3.safetensors', 's3gen.pt', \
'grapheme_mtl_merged_expanded_v1.json', 'conds.pt', 'Cangjie5_TC.json'])"
# Segmenteur chinois de Chatterbox : sinon telecharge (~35 Mo) a chaque cold start
RUN /venv/chatterbox/bin/python -c "from spacy_pkuseg import pkuseg; pkuseg()"
RUN /venv/qwen/bin/python -c "from huggingface_hub import snapshot_download as d; \
[d(m) for m in ('Qwen/Qwen3-TTS-12Hz-1.7B-Base', 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign')]"

COPY engines/ /app/engines/
COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

CMD ["python3", "-u", "handler.py"]
