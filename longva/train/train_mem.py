from train import train
import torch
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
