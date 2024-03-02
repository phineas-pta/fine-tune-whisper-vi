FROM huggingface/peft-gpu

RUN apt-get update && pip install --no-cache-dir 'datasets[audio]'

WORKDIR /workspace

ENV XDG_CACHE_HOME=/workspace/cache

COPY train.py /workspace/
