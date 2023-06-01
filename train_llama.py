import logging
import random
import argparse
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from llama import Tokenizer, Transformer, LLaMA, ModelArgs
from data_utils import PileDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_model(args):
    tokenizer = Tokenizer(args.tokenizer_path)
    model_args = ModelArgs(dim=args.hidden_dim, n_layers=args.n_layers, n_heads=args.n_heads, vocab_size=tokenizer.n_words,
                           norm_eps=args.vnorm_eps, max_batch_size=args.batch_size, max_seq_len=args.max_seq_len)
    model = Transformer(model_args)
    if args.model_dir and args.load_epoch != -1:
        if not args.load_epoch:
            # Load the latest model checkpoint, in the form of llama_{i}.pth with largest i
            best_path = max(glob.glob(f"{args.model_dir}/llama_*.pth"), key=lambda x: int(x.stem.split("_")[1]))
            model.load_state_dict(torch.load(best_path))
        else:
            model.load_state_dict(torch.load(args.model_dir / f"llama_{args.load_epoch}.pth"))
        logger.info(f"Loaded model state dict from {args.model_dir}")
    llama = LLaMA(model, tokenizer)
    logger.info(f"Loaded model")

    dataset = PileDataset(args.dataset_path, tokenizer, args.max_seq_len, args.dataset_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    logger.info("Loaded dataset")

    optimizer = torch.optim.AdamW(llama.model.parameters(), lr=args.lr)
    if args.load_optimizer and args.load_epoch != -1:
        if not args.load_epoch:
            best_path = max(args.model_dir.glob("optimizer_*.pth"), key=lambda x: int(x.stem.split("_")[1]))
            optimizer.load_state_dict(torch.load(best_path))
        else:
            optimizer.load_state_dict(torch.load(f"{args.model_dir}/optimizer_{args.load_epoch}.pth"))
        logger.info(f"Loaded optimizer state from {args.model_dir}")
    logger.info(f"Loaded optimizer")
        

    batch_counter = 0
    cumulative_loss = 0
    for i in range(args.epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            loss = llama.forward(batch)
            loss.backward() # Will report backpropogate twice error if not set to True
            optimizer.step()
            
            batch_counter += 1
            cumulative_loss += loss.item()
            if (batch_counter) % args.save_freq == 0:
                torch.save(llama.model.state_dict(), f"f{args.model_dir}/llama_{batch_counter*args.batch_size}.pth")
                torch.save(optimizer.state_dict(), f"f{args.model_dir}/optimizer_{batch_counter*args.batch_size}.pth")
                logger.info(f"Saved model at {batch_counter} batches")
            if (batch_counter) % args.log_freq == 0:
                logger.info(f"Epoch {i}, batch {batch_counter}: loss {cumulative_loss/args.log_freq}")
                cumulative_loss = 0
            
    torch.save(llama.model.state_dict(), f"model_dir/llama_{batch_counter*args.batch_size}.pth")
    torch.save(optimizer.state_dict(), f"model_dir/optimizer_{batch_counter*args.batch_size}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--load_epoch", type=int, default=None, help="Load model and optimizer from epoch i, pass -1 to not train from scratch")
    parser.add_argument("--load_optimizer", default='store_false')
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--vnorm_eps", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum context length")

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_size", type=int, default=None, help="Number of samples to use from the dataset")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=100, help="save per every n batches")
    parser.add_argument("--log_freq", type=int, default=10, help="log per every n batches")
    
    args = parser.parse_args()

    train_model(args)