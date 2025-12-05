import json
import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len=128):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        self.samples = []
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

        self.pad_token = tokenizer.token_to_id("<pad>") or 0

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids

        if len(ids) > self.seq_len:
            ids = ids[:self.seq_len]

        if len(ids) < self.seq_len:
            ids += [self.pad_token] * (self.seq_len - len(ids))

        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]
        vec = item["label"]

        x = self.encode_text(text)
        y = torch.tensor(vec, dtype=torch.float32)

        return x, y
