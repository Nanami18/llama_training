import json

from torch.utils.data import DataLoader, Dataset
import torch

class PileDataset(Dataset):

    def __init__(self, file_path, tokenizer, context_length, dataset_size=None, dataset_start=0):
        self.file_path = file_path
        self.data = []
        self.tokenizer = tokenizer
        self.context_length = context_length
        with open(file_path, 'r') as f:
            line_counter = 0
            for line in f:
                if line_counter < dataset_start:
                    line_counter += 1
                    continue
                if line == "\n":
                    continue
                text_line = json.loads(line)
                # If we do this, wouldn't this cause the model to generate eos only rarely?
                tokens = self.tokenizer.encode(text_line['text'], bos=True, eos=True)
                # segment text by context length
                for i in range(0, len(tokens), context_length):
                    self.data.append(tokens[i:i+context_length])
                line_counter += 1
                if dataset_size is not None and line_counter >= dataset_size:
                    break
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # pad the token to the context length, temporarily use bos token to get pass the embedding layer
        data = data + [self.tokenizer.bos_id] * (self.context_length - len(data))
        
        return torch.tensor(data).long()