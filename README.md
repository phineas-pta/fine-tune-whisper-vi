# fine-tune whisper vi

jupyter notebooks to fine tune whisper models on vietnamese using colab and/or kaggle

*N.B.* import any trainer or pipeline class from `transformers` crash kaggle TPU session so better use GPU

evaluate accuracy (WER): [evaluate-whisper.ipynb](evaluate-whisper.ipynb)

fine-tune whisper tiny with traditional approach:
- script: [whisper-tiny-traditional.ipynb](whisper-tiny-traditional.ipynb)
- model with evaluated WER: https://huggingface.co/doof-ferb/whisper-tiny-vi

fine-tine whisper large with PEFT-LoRA + int8: [whisper-large-lora.ipynb](whisper-large-lora.ipynb)

fine-tune wav2vec v2 bert: *not yet*

convert to `openai-whisper`, `whisper.cpp`, `faster-whisper`, ONNX, TensorRT: *not yet*
