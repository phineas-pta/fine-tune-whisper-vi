#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract vietnamese subset of the FLEURS dataset & CommonVoice dataset
"""

# %pip install -q 'datasets[audio]'
import datasets as hugDS

ds = hugDS.load_dataset("google/fleurs", name="vi_vn", trust_remote_code=True).select_columns(["id", "audio", "transcription", "gender"])
ds.push_to_hub("doof-ferb/FLEURS_vie", token="███")

ds = hugDS.load_dataset("mozilla-foundation/common_voice_17_0", name="vi", trust_remote_code=True).select_columns(["audio", "sentence", "age", "gender"])
ds.push_to_hub("doof-ferb/CommonVoice_v17.0_vie", token="███")
