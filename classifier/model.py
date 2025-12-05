import torch
import torch.nn as nn
from tokenizers import Tokenizer


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 32, kernel_size=k, padding=k // 2)
            for k in [1, 2, 3, 5]
        ])

        self.classifier = nn.Sequential(
            nn.Linear(32 * 4, 64),  # 4 conv outputs * 32 channels
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[3].weight, gain=0.1)

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.classifier[-1].weight, gain=0.1)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)

        conv_outs = []
        for conv in self.convs:
            h = torch.relu(conv(emb))
            h = torch.max_pool1d(h, h.size(-1)).squeeze(-1)  # Global max pooling
            conv_outs.append(h)

        combined = torch.cat(conv_outs, dim=1)
        return self.classifier(combined)

        return self.mlp(pooled)

def check_model():
    tokenizer = Tokenizer.from_file("../tokenizer/trained_tokenizer.json")
    vocab_size = tokenizer.vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=32,  # MUST be 32 (from checkpoint)
        hidden_dim=64,  # MUST be 64 (from checkpoint)
        num_classes=6  # MUST be 6 (from checkpoint)
    )

    checkpoint = torch.load("../classifier/models/emotions_classifier.pt",
                            map_location=torch.device('cpu'))

    print("Loading checkpoint dictionary...")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded successfully")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")

    model.eval()

    # 4. Test texts
    test_texts = [
        "I HATE YOU!",
        "I love you so much! ❤️",
        "This makes me really angry!",
        "I'm feeling depressed today...",
        "Wow! This is incredible!",
        "How does this thing work?"
    ]

    emotions = ["friendly/kind", "calm/serious", "angry/mean",
                "sad/depressed", "excited/fun", "curious/motivated"]

    print("\n" + "=" * 60)
    print("Testing model predictions (with sigmoid):")
    print("=" * 60)

    for text in test_texts:
        # Tokenize and prepare input
        encoding = tokenizer.encode(text)
        input_ids = encoding.ids[:128]  # Truncate to max 128 tokens

        # Pad to exactly 128 tokens (CRITICAL!)
        if len(input_ids) < 128:
            input_ids = input_ids + [0] * (128 - len(input_ids))

        # Convert to tensor
        input_tensor = torch.tensor(input_ids).unsqueeze(0)  # [1, 128]

        print(f"\nInput for '{text}':")
        print(f"  Token length: {len(encoding.ids)} (padded to 128)")

        with torch.no_grad():
            # Get raw output (logits)
            raw_output = model(input_tensor)
            print(f"  Raw output shape: {raw_output.shape}")
            print(f"  Raw values: {raw_output[0].tolist()}")

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(raw_output)[0]

        print(f"\nProbabilities for '{text}':")
        for i, emotion in enumerate(emotions):
            print(f"  {emotion:20} {probs[i]:.3f} ({probs[i]:.1%})")

        # Find strongest emotion
        strongest_idx = probs.argmax()
        print(f"  → Strongest: {emotions[strongest_idx]} ({probs[strongest_idx]:.1%})")

    print("\n" + "=" * 60)
    print("ADDITIONAL DEBUGGING:")
    print("=" * 60)

    # Test with random input to verify model works
    print("\nTesting with random input:")
    random_input = torch.randint(0, 100, (1, 128))
    with torch.no_grad():
        random_output = model(random_input)
        random_probs = torch.sigmoid(random_output)[0]
        print(f"Random input output: {random_probs.tolist()}")

    # Test model parameters
    print("\nModel parameter shapes:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")


if __name__ == "__main__":
    check_model()




