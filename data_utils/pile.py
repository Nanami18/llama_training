import json
from torch.util.data import DataLoader, Dataset

class PileDataset(Dataset):

    def __init__(self, file_path, context_length, tokenizer):
        self.file_path = file_path
        self.data = []
        self.tokenizer = tokenizer
        with open(file_path, 'r') as f:
            for line in f:
                text_line = json.loads(line)
                # If we do this, wouldn't this cause the model to generate eos only rarely?
                tokens = self.tokenizer.encode(text_line, bos=True, eos=True)
                # segment text by context length
                for i in range(0, len(tokens), context_length):
                    self.data.append(tokens[i:i+context_length])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # pad the token to the context length
        data = data + [self.tokenizer.pad_id] * (self.context_length - len(data))
    