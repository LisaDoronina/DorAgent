from tokenizers import Tokenizer
from data.prepare_data import train_conversations, val_conversations
import random
import numpy as np

tokenizer = Tokenizer.from_file("trained_tokenizer.json")

EOS = tokenizer.token_to_id("<|eos|>")
BLOCK = 1024

tokenized = []
conversations = train_conversations

for conv in conversations:
    text = conv["text"] + "<|eos|>"
    ids = tokenizer.encode(text).ids
    tokenized.append(ids)

train_blocks = []


for ids in tokenized:
    for i in range(0, len(ids) - BLOCK + 1, BLOCK):
        block = ids[i:i + BLOCK]
        train_blocks.append(block)

random.shuffle(train_blocks)

dataset = np.array(train_blocks, dtype=np.int32)
np.save("../data/conversations_tokenized/conversations.npy", dataset)
