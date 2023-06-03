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

def evaluate_on_val(args, device):
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
        checkpoint_path = max(glob.glob(f"{args.model_dir}/llama_*.pth"), key=lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]))
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        checkpoint_path = f"{args.model_dir}/llama_{args.load_epoch}.pth"
        model.load_state_dict(torch.load(checkpoint_path))
    logger.info(f"Loaded model state dict from {checkpoint_path}")

    llama = LLaMA(model, tokenizer)
    llama.to_device(device)
    logger.info(f"Loaded model")

    val_dataset = PileDataset(args.valset_path, tokenizer, training_args["max_seq_len"], args.valset_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    logger.info(f"Loaded validation dataset")

    avg_loss = compute_val_loss(val_dataloader, llama)
    print("Average validation loss: ", avg_loss)

def compute_val_loss(val_dataloader, llama_model):
    llama_model.model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in val_dataloader:
            batch = batch.to(llama_model.device)
            loss = llama_model.forward(batch)
            total_loss += loss.item()
    return total_loss / len(val_dataloader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--load_epoch", type=int, default=None, help="Load model and optimizer from epoch i")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    
    parser.add_argument("--valset_path", type=str, required=True)
    parser.add_argument("--valset_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    evaluate_on_val(args, device)
