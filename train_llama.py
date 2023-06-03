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
from evaluate import compute_val_loss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_model(args, device):
    tokenizer = Tokenizer(args.tokenizer_path)
    model_args = ModelArgs(dim=args.hidden_dim, n_layers=args.n_layers, n_heads=args.n_heads, vocab_size=tokenizer.n_words,
                           norm_eps=args.vnorm_eps, max_batch_size=args.batch_size, max_seq_len=args.max_seq_len)
    model = Transformer(model_args)

    trained_sequences = 0
    if args.model_dir and args.load_epoch != -1:
        if not args.load_epoch:
            # Load the latest model checkpoint, in the form of llama_{i}.pth with largest i
            checkpoint_path = max(glob.glob(f"{args.model_dir}/llama_*.pth"), key=lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]))
            model.load_state_dict(torch.load(checkpoint_path))
            trained_sequences = int(checkpoint_path.split("/")[-1].split("_")[1].split(".")[0])
        else:
            checkpoint_path = f"{args.model_dir}/llama_{args.load_epoch}.pth"
            model.load_state_dict(torch.load(checkpoint_path))
            trained_sequences = int(args.load_epoch)
        logger.info(f"Loaded model state dict from {checkpoint_path}")
    llama = LLaMA(model, tokenizer)
    llama.to_device(device)
    logger.info(f"Loaded model")

    cur_time = time.time()
    dataset = PileDataset(args.dataset_path, tokenizer, args.max_seq_len, args.dataset_size, args.dataset_start)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    finish_time = time.time()
    logger.info("Loaded dataset in %.2f seconds", finish_time - cur_time)
    logger.info("Num batches: %d, batch size: %d", len(dataloader), args.batch_size)
    if args.validation_period is not None:
        assert args.valset_path is not None, "Validation set is not provided"
        val_dataset = PileDataset(args.valset_path, tokenizer, args.max_seq_len, args.valset_size)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(llama.model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps, weight_decay=args.weight_decay)
    if args.load_optimizer and args.load_epoch != -1:
        if not args.load_epoch:
            optimizer_path = max(glob.glob(f"{args.model_dir}/optimizer_*.pth"), key=lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]))
            optimizer.load_state_dict(torch.load(optimizer_path))
        else:
            optimizer_path = f"{args.model_dir}/optimizer_{args.load_epoch}.pth"
            optimizer.load_state_dict(torch.load(optimizer_path))
        logger.info(f"Loaded optimizer state from {optimizer_path}")
    logger.info(f"Loaded optimizer")
        
    llama.model.to(device)
    cumulative_loss = 0
    batch_counter = 0
    start_time = time.time()
    for i in range(args.epochs):
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = llama.forward(batch)
            loss.backward()
            nn.utils.clip_grad_norm(llama.model.parameters(), args.clip_grad_norm)
            optimizer.step()
            
            batch_counter += 1
            cumulative_loss += loss.item()
            trained_sequences += args.batch_size
            if (batch_counter) % args.save_freq == 0:
                torch.save(llama.model.state_dict(), f"{args.model_dir}/llama_{trained_sequences}.pth")
                torch.save(optimizer.state_dict(), f"{args.model_dir}/optimizer_{trained_sequences}.pth")
                logger.info(f"Saved model at {trained_sequences} trained sequences")
            if args.validation_period is not None and (batch_counter) % args.validation_period == 0:
                val_loss = compute_val_loss(llama, val_dataloader)
                logger.info(f"Epoch {i} trained sequences {trained_sequences} loss: {val_loss}")
            if (batch_counter) % args.log_freq == 0:
                logger.info(f"Epoch {i}, trained sequence {trained_sequences}: loss {cumulative_loss/args.log_freq}")
                logger.info(f"Time elapsed: {time.time() - start_time}")
                cumulative_loss = 0
            torch.cuda.empty_cache()
            
            
    torch.save(llama.model.state_dict(), f"{args.model_dir}/llama_{trained_sequences}.pth")
    torch.save(optimizer.state_dict(), f"{args.model_dir}/optimizer_{trained_sequences}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--load_epoch", type=int, default=None, help="Load model and optimizer from epoch i, pass -1 to not train from scratch")
    parser.add_argument("--load_optimizer", default='store_false')
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--vnorm_eps", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum context length")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_size", type=int, default=None, help="Number of samples to use from the dataset")
    parser.add_argument("--dataset_start", type=int, default=0, help="Index of the first sample to use from the dataset")
    parser.add_argument("--validation_period", type=int, default=None, help="Number of batches between validation checks")
    parser.add_argument("--valset_path", type=str)
    parser.add_argument("--valset_size", type=int, default=10000)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=100, help="save per every n batches")
    parser.add_argument("--log_freq", type=int, default=10, help="log per every n batches")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
        
        
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    with open(f'{args.model_dir}/training_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_model(args, device)