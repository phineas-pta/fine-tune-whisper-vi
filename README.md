# fine-tune whisper vi

jupyter notebooks to fine tune whisper models on vietnamese using kaggle (should also work on colab but not throughly tested)

*N.B.1* import any trainer or pipeline class from `transformers` crash kaggle TPU session so better use GPU

*N.B.2* trainer class from `transformers` can auto use multi-GPU like kaggle free T4×2 without code change

*N.B.3* use default greedy search, because beam search trigger a spike in VRAM usage which may cause out-of-memory (original whisper use num beams = 5, something like `do_sample=True, num_beams=5`)

## scripts

evaluate accuracy (WER):
- on whisper models: [evaluate-whisper.ipynb](eval/evaluate-whisper.ipynb)
- on whisper with PEFT LoRA: [evaluate-whisper-lora.ipynb](eval/evaluate-whisper-lora.ipynb)
- on wav2vec BERT v2 models: [evaluate-w2vBERT.ipynb](eval/evaluate-w2vBERT.ipynb)

fine-tune whisper tiny with traditional approach:
- script: [whisper-tiny-traditional.ipynb](train/whisper-tiny-traditional.ipynb)
- model with evaluated WER: https://huggingface.co/doof-ferb/whisper-tiny-vi

fine-tine whisper large with PEFT-LoRA + int8:
- script: [whisper-large-lora.ipynb](train/whisper-large-lora.ipynb)
- model with evaluated WER: https://huggingface.co/doof-ferb/whisper-large-peft-lora-vi

fine-tune wav2vec v2 bert: [w2v-bert-v2.ipynb](train/w2v-bert-v2.ipynb)

docker image to fine-tune on AWS: [Dockerfile](docker/Dockerfile)

convert to `openai-whisper`, `whisper.cpp`, `faster-whisper`, ONNX, TensorRT: *not yet*

miscellaneous: convert to huggingface audio datasets format

## datasets

my collection of vietnamese speech datasets: https://huggingface.co/collections/doof-ferb/vietnamese-speech-dataset-65c6af8c15c9950537862fa6

other datasets but cannot find download links:
- Broadcasting Speech Corpus VOV: https://www.isca-archive.org/iscslp_2006/luong06_iscslp.html
- VNSpeechCorpus: https://aclanthology.org/L04-1364/
- VAIS-1000: https://ieee-dataport.org/documents/vais-1000-vietnamese-speech-synthesis-corpus
- https://arxiv.org/abs/1904.05569
- ViASR: https://aclanthology.org/2023.paclic-1.38/
- NIST OpenKWS13 Evaluation Workshop
- Viettel call center data: https://vlsp.org.vn/sites/default/files/2019-10/VLSP2019-ASR-MaiVanTuan.pdf
- https://catalog.elra.info/en-us/repository/browse/ELRA-S0322/
- https://catalog.elra.info/en-us/repository/browse/ELRA-S0432/
