#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""convert VietMed unlabeled set to huggingface audio datasets format"""

###############################################################################
# commands to download files in colab

# those zip files contain quite weird file path inside
# in case error with gdown, see https://stackoverflow.com/a/67550427/10805680

# !gdown --folder 1Wrofm2FkjYngpxR1ysmkm0XvFiGEvyFM  # folder with 10 zip files in official google drive download link
# !for file in unlabeled_medical_data/*.zip; do unzip -q "$file" && rm "$file"; done
# !mkdir data
# !for file in content/drive/MyDrive/Colab_Notebooks/VietMed/VietMed_unlabeled/*/*.wav; do mv "$file" data/; done
# %pip install -qU 'datasets[audio]'
# %cd data

###############################################################################

import os, datasets as hugDS
yolo = os.listdir()

(hugDS
	.Dataset.from_dict({"audio": yolo, "Metadata ID": [x[:14] for x in yolo]})
	.cast_column("audio", hugDS.Audio())
	.push_to_hub("doof-ferb/VietMed_unlabeled", token="███")
)
