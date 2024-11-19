# [Preprint] Number it: Temporal Grounding Videos like Flipping Manga
This repository contains the PyTorch implementation for the [paper Number it: Temporal Grounding Videos like Flipping Manga](https://arxiv.org/abs/2411.10332).

If you have any questions on this repository or the related paper, feel free to create an issue.

## Introduction
Video Large Language Models (Vid-LLMs) excel in video comprehension but struggle with precise temporal localization. Introducing Number-Prompt (NumPro): a novel method that adds unique numerical identifiers to video frames, transforming Video Temporal Grounding (VTG) into an intuitive process similar to flipping through manga panels. This technique significantly enhances VTG performance without additional computational cost, achieving up to 6.9% improvement in mIoU for moment retrieval and 8.5% in mAP for highlight detection.

## Get Started

## Data
Please download the video data from [Charades-STA](http://vuchallenge.org/charades.html), [DiDeMo](https://github.com/LisaAnne/TemporalLanguageRelease) and [ActivityNet](http://activity-net.org/download.html).

The instruction dataset for training can be found at [Google Drive](https://drive.google.com/drive/folders/13NYRDC87Uc4AqaT5FBHA7QkHV5OMl-v8?usp=sharing).

Note: We also use the instruction dataset from [VTimeLLM](https://github.com/huangb23/VTimeLLM) stage2, due to the copyright issue of youtube, we cannot provide source video here. You can download through [yt-dlp](https://github.com/yt-dlp/yt-dlp) by yourself.
## Train

## Inference
The checkpoints and results can be found at [Google Drive](https://drive.google.com/drive/folders/13NYRDC87Uc4AqaT5FBHA7QkHV5OMl-v8?usp=sharing).
```bash
DATA_PATH="charades/videos"
LORA_PATH="longva_7b_dpo_NumPro_FT"

python eval_vtg.py \
    --test_path testset/charades_test.json \
    --data_path $DATA_PATH --save_path results/${LORA_PATH}_charades.json 
```
## Acknowledgement
Our implementation is based on the following repositories:

- [VTimeLLM](https://github.com/huangb23/VTimeLLM). 
- [TimeChat](https://github.com/RenShuhuai-Andy/TimeChat)
- [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA)
- [Open-LLaVA-NeXT](https://github.com/xiaoachen98/Open-LLaVA-NeXT)
- [OpenLongVA](https://github.com/LaBaZh/OpenLongVA)

We thank the authors for their excellent work.
