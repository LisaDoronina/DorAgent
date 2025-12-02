import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

# Add path to your model
sys.path.append('data/model/')
import data.model.gpt as GPT


def test_model():
    print("🚀 Testing your trained AI model...")

    try:
        # Load your model
        checkpoint = torch.load("../tokenizer/best_model.pth")
        print(f"✅ Model loaded! Val loss: {checkpoint.get('val_loss', 'N/A')}")

        data = np.load("../data/final_training_data.npy")
        vocab_size = int(data.max()) + 1
        print(f"📊 Vocab size: {vocab_size}")

        # Create model
        model = GPT.GPT(
            vocab_size=vocab_size,
            **checkpoint['config']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("🤖 Model ready for chat!")

        # Simple generation test
        def generate(prompt_tokens, max_length=20):
            tokens = prompt_tokens.copy()
            for _ in range(max_length):
                input_tokens = tokens[-model.block_size:]
                x = torch.tensor([input_tokens], dtype=torch.long)

                with torch.no_grad():
                    logits = model(x)

                next_token = torch.argmax(logits[0, -1, :]).item()
                tokens.append(next_token)

            return tokens

        # Test with sample tokens
        print("\n🧪 Generation test:")
        test_prompt = [1, 2, 3]  # Replace with actual tokens from your data
        result = generate(test_prompt)
        print(f"Input: {test_prompt}")
        print(f"Output: {result}")
        print(f"Generated {len(result)} tokens")

        print("\n🎉 Your AI is working! Ready for proper chat interface!")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_model()