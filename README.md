# fine-tune whisper vi

jupyter notebooks to fine tune whisper models on vietnamese using kaggle (should also work on colab but not throughly tested)

using my collection of vietnamese speech datasets: https://huggingface.co/collections/doof-ferb/vietnamese-speech-dataset-65c6af8c15c9950537862fa6

*N.B.1* import any trainer or pipeline class from `transformers` crash kaggle TPU session (see huggingface/transformers#28609) so better use GPU

*N.B.2* ~~trainer class from `transformers` can auto use multi-GPU like kaggle free T4×2 without code change~~ by default trainer use naive model parallelism which cannot fully use all gpu in same time, so better use distributed data parallelism

*N.B.3* use default greedy search, because beam search trigger a spike in VRAM usage which may cause out-of-memory (original whisper use num beams = 5, something like `do_sample=True, num_beams=5`)

*N.B.4* if use kaggle + resume training, remember to enable files persistency before launching

## scripts

evaluate accuracy (WER) with batched inference:
- on whisper models: [evaluate-whisper.ipynb](eval/evaluate-whisper.ipynb)
- on whisper with PEFT LoRA: [evaluate-whisper-lora.ipynb](eval/evaluate-whisper-lora.ipynb)
- on wav2vec BERT v2 models: [evaluate-w2vBERT.ipynb](eval/evaluate-w2vBERT.ipynb)

fine-tune whisper tiny with traditional approach:
- script: [whisper-tiny-traditional.ipynb](train/whisper-tiny-traditional.ipynb)
- model with evaluated WER: https://huggingface.co/doof-ferb/whisper-tiny-vi

fine-tine whisper large with PEFT-LoRA + int8:
- script for 1 GPU: [whisper-large-lora.ipynb](train/whisper-large-lora.ipynb)
- script for multi-GPU using distributed data parallelism: [whisper-large-lora-DDP.ipynb](train/whisper-large-lora-DDP.ipynb)
- model with evaluated WER: https://huggingface.co/doof-ferb/whisper-large-peft-lora-vi

(testing - not working) fine-tune wav2vec v2 bert: [w2v-bert-v2.ipynb](train/w2v-bert-v2.ipynb)

docker image to run on AWS EC2: [Dockerfile](docker/Dockerfile), comes with standalone scripts

convert to `openai-whisper`, `whisper.cpp`, `faster-whisper`, ONNX, TensorRT: *not yet*

miscellaneous: convert to huggingface audio datasets format

## resources

- https://huggingface.co/blog/fine-tune-whisper
- https://huggingface.co/blog/fine-tune-w2v2-bert
- https://github.com/openai/whisper/discussions/988
- https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb
- https://github.com/vasistalodagala/whisper-finetune
- https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event
- https://github.com/krylm/whisper-event-tuning
- https://www.kaggle.com/code/leonidkulyk/train-infer-mega-pack-wav2vec2-whisper-qlora
- https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py
