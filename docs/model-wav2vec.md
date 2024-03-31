# Facebook/Meta Wav2vec

## metadata

wav2vec v2 - BERT:
- https://huggingface.co/facebook/w2v-bert-2.0
- https://huggingface.co/docs/transformers/main/en/model_doc/wav2vec2-bert

output word timestamp more correctly

cannot handle long audio (out-of-memory error)

smaller community / less resources than Whisper

## fine-tuning

https://huggingface.co/blog/fine-tune-w2v2-bert

found vietnamese fine-tuned model: https://huggingface.co/trick4kid/w2v-bert-2.0-vietnamese-CV16.0 <br />
WER: 35.5% on CommonVoice v16.0

LoRA maybe possible but not yet: https://github.com/huggingface/peft/issues/128
