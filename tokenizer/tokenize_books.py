import os
import glob

from tokenizers import Tokenizer
import numpy as np

tokenizer = Tokenizer.from_file("trained_tokenizer.json")
SEQ_LEN = 1024

def load_book_text(folder_path):
    texts = []
    for file in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned = raw_text.replace("\r", "").strip()
        texts.append(cleaned)

        print(f"Loaded {file} ({len(cleaned)} characters)")

    return texts

def tokenize_chunk_padding(text, seq_len = SEQ_LEN):
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids

    chunks = []

    pad_token_id = tokenizer.token_to_id("<pad>") or 0

    for i in range(0, len(token_ids), seq_len):
        chunk = token_ids[i:i + seq_len]

        if len(chunk) >= seq_len // 2:
            if len(chunk) < seq_len:
                padding_length = seq_len - len(chunk)
                chunk = chunk + [pad_token_id] * padding_length

            if len(chunk) == seq_len:
                chunks.append(chunk)

    print(f"Created {len(chunks)} chunks")
    return chunks

def save_chunks(chunks, filename):
    if not chunks:
        print("No chunks found...")
        return

    array = np.array(chunks, dtype=np.int32)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, array)
    print(f"Saved {array.shape} to {filename}.npy")

def process_books():
    texts = load_book_text("../data/books_raw")

    if not texts:
        print("No texts found...")
        return

    all_chunks = []

    for i, text in enumerate(texts):
        chunks = tokenize_chunk_padding(text)
        all_chunks.extend(chunks)

    save_chunks(all_chunks, "../data/books_tokenized/books")

if __name__ == "__main__":
    process_books()