# OpenAI Whisper

## metadata

Transformer sequence-to-sequence model

pre-trained on 680k hours of audio, a lot of which came from youtube

text not cleaned properly → hallucinations on non-speech segments,<br />
e.g. “hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn”<br />
see https://github.com/openai/whisper/discussions/928

training data: vietnamese part: only 691h audio for transcription task (+1.7k hours for translation task)

## fine-tuning

- traditional fine-tuning: https://huggingface.co/blog/fine-tune-whisper
- PEFT-LoRA int8:
  - https://github.com/openai/whisper/discussions/988
  - https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb

## audio pre-processing techniques

remove non-speech segments: SileroVAD: https://github.com/snakers4/silero-vad

separate voice from background music/noise:
- https://github.com/facebookresearch/demucs
- https://github.com/deezer/spleeter

need to further testing on vietnamese audio

large-v3 maybe worse than large-v2 because of hallucination: https://deepgram.com/learn/whisper-v3-results

## inference acceleration

native quantization: see my post: https://github.com/openai/whisper/discussions/1990

alternative inference backend: https://github.com/SYSTRAN/faster-whisper

run on embedded systems: https://github.com/ggerganov/whisper.cpp

huggingface transformers framework:
- https://huggingface.co/docs/transformers/en/perf_infer_gpu_one
- https://huggingface.co/blog/whisper-speculative-decoding
- https://huggingface.co/docs/optimum/main/en/quicktour

convert to ONNX:
- https://github.com/zhuzilin/whisper-openvino
- https://github.com/k2-fsa/sherpa-onnx

convert to TensorRT: partially successful: https://github.com/openai/whisper/discussions/169
