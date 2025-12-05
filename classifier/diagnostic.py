# debug_model.py
import torch
import torch.nn as nn
from tokenizers import Tokenizer
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)


# 1. Define the EXACT model architecture
class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_classes=6):
        super().__init__()
        print(f"Creating model with: vocab_size={vocab_size}, embed_dim={embed_dim}, hidden_dim={hidden_dim}")

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        # Print shapes for debugging
        print(f"  embedding.weight shape: {self.embedding.weight.shape}")
        print(f"  mlp.0.weight shape: {self.mlp[0].weight.shape}")
        print(f"  mlp.3.weight shape: {self.mlp[3].weight.shape}")

    def forward(self, x):
        print(f"\nForward pass:")
        print(f"  Input x shape: {x.shape}, dtype: {x.dtype}")
        print(f"  Input sample values: {x[0, :5].tolist()}...")

        emb = self.embedding(x)
        print(f"  Embedding output shape: {emb.shape}")

        pooled = emb.mean(dim=1)
        print(f"  Pooled shape: {pooled.shape}")

        output = self.mlp(pooled)
        print(f"  MLP output shape: {output.shape}")
        print(f"  Raw output values: {output[0].tolist()}")

        return output


# 2. Load tokenizer
print("\n" + "=" * 60)
print("Loading tokenizer...")
try:
    tokenizer = Tokenizer.from_file("../tokenizer/trained_tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    print(f"✓ Tokenizer loaded, vocab_size={vocab_size}")
except Exception as e:
    print(f"✗ Error loading tokenizer: {e}")
    sys.exit(1)

# 3. Create model
print("\n" + "=" * 60)
print("Creating model...")
model = SimpleClassifier(
    vocab_size=vocab_size,
    embed_dim=32,  # MUST match checkpoint
    hidden_dim=64,  # MUST match checkpoint
    num_classes=6  # MUST match checkpoint
)

# 4. Load checkpoint
print("\n" + "=" * 60)
print("Loading checkpoint...")
try:
    checkpoint = torch.load("../classifier/models/emotions_classifier.pt",
                            map_location=torch.device('cpu'))
    print(f"✓ Checkpoint loaded")
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")

    if 'model_state_dict' in checkpoint:
        print("\nLoading model state dict...")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model weights loaded successfully")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    else:
        print("✗ No 'model_state_dict' in checkpoint!")
        sys.exit(1)

except Exception as e:
    print(f"✗ Error loading checkpoint: {e}")
    sys.exit(1)

# 5. Test with a simple input FIRST
print("\n" + "=" * 60)
print("TEST 1: Simple constant input")
print("=" * 60)

model.eval()

# Test with token ID 100 (any valid token)
simple_tokens = torch.full((1, 128), 100, dtype=torch.long)
print(f"Simple input: all tokens = 100")
print(f"Input shape: {simple_tokens.shape}")

with torch.no_grad():
    output = model(simple_tokens)
    probs = torch.sigmoid(output)[0]
    print(f"\nOutput probabilities:")
    emotions = ["friendly", "calm", "angry", "sad", "excited", "curious"]
    for i, prob in enumerate(probs):
        print(f"  {emotions[i]:10} {prob:.3f} ({prob:.1%})")

# 6. Test with actual text
print("\n" + "=" * 60)
print("TEST 2: Actual text input")
print("=" * 60)

test_text = "I love you"
print(f"Testing text: '{test_text}'")

# Tokenize
encoding = tokenizer.encode(test_text)
print(f"Token IDs: {encoding.ids}")
print(f"Number of tokens: {len(encoding.ids)}")

# Prepare input tensor
input_ids = encoding.ids[:128]  # Truncate
if len(input_ids) < 128:
    input_ids = input_ids + [0] * (128 - len(input_ids))

print(f"After padding to 128: {len(input_ids)} tokens")
print(f"First 10 tokens: {input_ids[:10]}")

input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
print(f"Input tensor shape: {input_tensor.shape}")
print(f"Input tensor dtype: {input_tensor.dtype}")

with torch.no_grad():
    output = model(input_tensor)
    probs = torch.sigmoid(output)[0]
    print(f"\nOutput probabilities for '{test_text}':")
    for i, prob in enumerate(probs):
        print(f"  {emotions[i]:10} {prob:.3f} ({prob:.1%})")

# 7. Test model weights
print("\n" + "=" * 60)
print("TEST 3: Model weight verification")
print("=" * 60)

print("\nChecking loaded weights match checkpoint:")
for name, param in model.named_parameters():
    if name in checkpoint['model_state_dict']:
        checkpoint_weight = checkpoint['model_state_dict'][name]
        if torch.allclose(param, checkpoint_weight, rtol=1e-5):
            print(f"  ✓ {name}: shapes match")
        else:
            print(f"  ✗ {name}: WEIGHTS DIFFER!")
            print(f"    Model mean: {param.mean():.6f}, Checkpoint mean: {checkpoint_weight.mean():.6f}")
    else:
        print(f"  ✗ {name}: Not in checkpoint!")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)