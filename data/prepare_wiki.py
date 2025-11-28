import os
import re
from datasets import load_dataset

def clean_wiki_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def prepare_wiki_corpus(output_file = "../data/wiki_corpus.txt"):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:5_000]")

    wiki_texts = []
    for i, article in enumerate(dataset):
        text = article["text"]
        cleaned = clean_wiki_text(text)
        if len(cleaned) > 100:
            wiki_texts.append(cleaned)

        if i % 1000 == 0:
            print(f"Processed {i} articles")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding='utf-8') as f:
        for text in wiki_texts:
            f.write(text + "\n\n")

    print(f"Saved {len(wiki_texts)} Wikipedia articles to {output_file}")
    return output_file

wiki_corpus_file = prepare_wiki_corpus()
