import logging
import random
import argparse
import glob
import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from llama import Tokenizer, Transformer, LLaMA, ModelArgs
from data_utils import PileDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def inference(args, device):
    tokenizer = Tokenizer(args.tokenizer_path)
    with open(f"{args.model_dir}/training_args.txt") as f:
        training_args = json.load(f)
    model_args = ModelArgs(dim=training_args["hidden_dim"], n_layers=training_args["n_layers"], 
                           n_heads=training_args["n_heads"], vocab_size=tokenizer.n_words,
                           norm_eps=training_args["vnorm_eps"], max_batch_size=training_args["batch_size"], 
                           max_seq_len=training_args["max_seq_len"])
    model = Transformer(model_args)

    if not args.load_epoch:
        # Load the latest model checkpoint, in the form of llama_{i}.pth with largest i
        best_path = max(glob.glob(f"{args.model_dir}/llama_*.pth"), key=lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]))
        model.load_state_dict(torch.load(best_path))
    else:
        model.load_state_dict(torch.load(args.model_dir / f"llama_{args.load_epoch}.pth"))
    logger.info(f"Loaded model state dict from {args.model_dir}")

    llama = LLaMA(model, tokenizer)
    llama.to_device(device)
    logger.info(f"Loaded model")

    prompts = args.prompt.split("###")
    start_time = time.time()
    output = llama.generate(prompts, args.max_gen_len, args.temperature, args.top_p)
    end_time = time.time() - start_time
    logger.info(f"Generated {len(output)} prompts in {end_time} seconds")
    for i in range(len(output)):
        print(f"Prompt: {prompts[i]}")
        print(f"Generated: {output[i]}")
        print("-----------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--load_epoch", type=int, default=None, help="Load model and optimizer from epoch i")
    parser.add_argument("--prompt", type=str, required=True, help="### separated prompts")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=int, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_gen_len", type=int, default=100)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
        
        
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    inference(args, device)
