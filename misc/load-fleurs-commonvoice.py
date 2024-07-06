#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""extract vietnamese subset of the FLEURS dataset & CommonVoice dataset"""

# %pip install -qU 'datasets[audio]'
import datasets as hugDS

(hugDS
	.load_dataset("google/fleurs", name="vi_vn", trust_remote_code=True)
	.select_columns(["id", "audio", "transcription", "gender"])
	.push_to_hub("doof-ferb/FLEURS_vie", token="███")
)

(hugDS
	.load_dataset("mozilla-foundation/common_voice_17_0", name="vi", trust_remote_code=True, token="███")
	.select_columns(["audio", "sentence", "age", "gender"])
	.push_to_hub("doof-ferb/CommonVoice_v17.0_vie", token="███")
)
