import logging
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.util.data import DataLoader, Dataset

from llama import Tokenizer, Transformer, LLaMA, ModelArgs
from data_utils import PileDataset

logger = logging.getLogger(__name__)

def train_model(args):
    tokenizer = Tokenizer(args.tokenizer_path)
    model_args = ModelArgs(dim=args.hidden_dim, n_layers=args.n_layers, n_heads=args.n_heads, norm_eps=args.vnorm_eps, max_batch_size=args.batch_size, max_seq_len=args.max_seq_len)
    model = Transformer(model_args)
    if args.model_path:
        if not args.load_epoch:
            # Load the latest model checkpoint, in the form of llama_{i}.pth with largest i
            best_path = max(args.model_path.glob("llama_*.pth"), key=lambda x: int(x.stem.split("_")[1]))
            model.load_state_dict(torch.load(best_path))
        else:
            model.load_state_dict(torch.load(args.model_path / f"llama_{args.load_epoch}.pth"))
        logger.info(f"Loaded model from {args.model_path}")
    llama = LLaMA(model, tokenizer)

    dataset = PileDataset(args.dataset_path, tokenizer, args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    logger.info("Loaded dataset")

    optimizer = torch.optim.AdamW(llama.parameters(), lr=args.lr)
    if args.load_optimizer:
        if not args.load_epoch:
            best_path = max(args.model_path.glob("optimizer_*.pth"), key=lambda x: int(x.stem.split("_")[1]))
            optimizer.load_state_dict(torch.load(best_path))
        else:
            optimizer.load_state_dict(torch.load(args.model_path / f"optimizer_{args.load_epoch}.pth"))
        logger.info(f"Loaded optimizer from {args.model_path}")
        

    batch_counter = 0
    cumulative_loss = 0
    for i in range(args.epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            loss = llama(batch)
            loss.backward()
            optimizer.step()
            
            batch_counter += 1
            cumulative_loss += loss.item()
            if i % args.save_freq == 0:
                torch.save(llama.state_dict(), f"model_dir/llama_{i*args.batch_size}.pth")
                torch.save(optimizer.state_dict(), f"model_dir/optimizer_{i*args.batch_size}.pth")
                logger.info(f"Saved model at {i} batches")
            if i % args.log_freq == 0:
                logger.info(f"Epoch {i}, batch {batch_counter}: loss {cumulative_loss/args.loss_freq}")
                cumulative_loss = 0
            
    torch.save(llama.state_dict(), f"model_dir/llama_{i*args.batch_size}.pth")
    torch.save(optimizer.state_dict(), f"model_dir/optimizer_{i*args.batch_size}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--load_epoch", type=int, default=None, help="Load model and optimizer from epoch i")
    parser.add_argument("--load_optimizer", default='store_false')
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--hidden_dim", type=str, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--vnorm_eps", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum context length")
    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=100, help="save per every n batches")
    parser.add_argument("--log_freq", type=int, default=10, help="log per every n batches")
    
    args = parser.parse_args()

    train_model(args)