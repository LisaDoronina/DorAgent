from tokenizers import Tokenizer
import os
import numpy as np

SEG_LEN = 1024

def tokenize_wiki(corpus_file, output_file, seg_len=SEG_LEN):
    tokenizer = Tokenizer.from_file("trained_tokenizer.json")

    with open(corpus_file, 'r', encoding='utf-8') as f:
        content = f.read()

    articles = content.split('\n\n')
    articles = [article.strip() for article in articles if article.strip()]

    all_chunks = []
    pad_token_id = tokenizer.token_to_id("<pad>") or 0

    for i, article in enumerate(articles):
        encoding = tokenizer.encode(article)
        token_ids = encoding.ids

        for j in range(0, len(token_ids), seg_len):
            chunk = token_ids[j:j+seg_len]

            if len(chunk) >= seg_len // 2:
                if len(chunk) < seg_len:
                    chunk = chunk + [pad_token_id] * (seg_len - len(chunk))

                if len(chunk) == seg_len:
                    all_chunks.append(chunk)

    if all_chunks:
        array = np.array(all_chunks, dtype=np.int32)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, array)
        print(f"Saved {array.shape} to {output_file}.npy")
        print(f"Created {len(all_chunks)} training sequences")
    else:
        print("No chunks were created!")

if __name__ == "__main__":
    # Tokenize the Wikipedia corpus you created
    tokenize_wiki(
        corpus_file="../data/wiki_corpus.txt",
        output_file="../data/wiki_tokenized/wiki_data"
    )