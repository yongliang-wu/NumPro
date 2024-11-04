# NumPro
To be achieved.

## Get Started

## Data

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
