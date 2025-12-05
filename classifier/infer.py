import torch
from tokenizers import Tokenizer
from model import Transformer
from dataset import EmotionDataset

MODEL_PATH = "../classifier/models/emotions_classifier.pt"
TOKENIZER_PATH = "../tokenizer/trained_tokenizer.json"

def load_model():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    model = Transformer(vocab_size=vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model, tokenizer

def predict(text):
    model, tokenizer = load_model()

    dataset = EmotionDataset("../classifier/data/emotions.txt", tokenizer, seq_len=128)
    x = dataset.encode_text(text).unsqueeze(0)

    with torch.no_grad():
        vec = model(x).squeeze(0).tolist()

    return vec