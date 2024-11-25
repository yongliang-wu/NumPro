from train import train
import torch
if __name__ == "__main__":
    if(torch.cuda.is_available()):
        train(attn_implementation="flash_attention_2")
    else:
        train()
