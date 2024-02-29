FROM huggingface/peft-gpu

RUN apt-get update && pip install --no-cache-dir 'datasets[audio]'

WORKDIR /workspace

COPY train.py /workspace/

CMD ["python", "train.py"]
