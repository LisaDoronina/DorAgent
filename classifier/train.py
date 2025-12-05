import torch
from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer
from dataset import EmotionDataset
from model import Transformer
import numpy as np

DATA_PATH = "../classifier/data/emotions.txt"
MODEL_SAVE = "../classifier/models/emotions_classifier.pt"


def smooth_labels(labels, alpha=0.2):
    """Apply label smoothing to prevent overconfident predictions"""
    num_classes = labels.shape[1]
    return (1 - alpha) * labels + alpha / num_classes


def train():
    tokenizer = Tokenizer.from_file("../tokenizer/trained_tokenizer.json")
    dataset = EmotionDataset(DATA_PATH, tokenizer, seq_len=128)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Smaller batch size for better gradient estimates
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    vocab_size = tokenizer.get_vocab_size()

    # TINY MODEL for tiny dataset
    model = Transformer(vocab_size=vocab_size, num_classes=6)

    device = next(model.parameters()).device

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    patience = 3
    best_val_loss = float("inf")
    patience_counter = 0

    # Optimizer with stronger regularization
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    pos_weight = torch.tensor([
        1.0,  # friendly: 34.6% → weight ~1.0
        2.0,  # calm: 14.9% → needs higher weight
        1.5,  # angry: 24.1%
        2.0,  # sad: 13.2%
        1.0,  # excited: 36.0%
        0.8  # curious: 47.2% → needs lower weight
    ])

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        patience=1,  # More aggressive reduction
        factor=0.5,
        min_lr=1e-6
    )

    print(f"Training on {device}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # After creating dataset, check it
    print("\n=== DATA CHECK ===")
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample labels distribution:")
    all_labels = []
    for i in range(min(100, len(dataset))):
        _, labels = dataset[i]
        all_labels.append(labels)

    all_labels = torch.stack(all_labels)
    print(f"Label means: {all_labels.mean(dim=0).tolist()}")
    print(f"Label stds: {all_labels.std(dim=0).tolist()}")

    # Check for duplicates
    print("\nChecking sample inputs...")
    sample_texts = []
    for i in range(min(5, len(dataset))):
        tokens, _ = dataset[i]
        sample_texts.append(tokens[:10].tolist())  # First 10 tokens
        print(f"Sample {i} first tokens: {tokens[:10].tolist()}")
    print("-" * 50)

    for epoch in range(50):  # More epochs but early stopping will kick in
        model.train()
        train_loss = 0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Apply label smoothing
            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()

            # Tighter gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            opt.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        # Validation
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Print epoch results
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {opt.param_groups[0]['lr']:.6f}")

        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'model_config': {  # Save all parameters
                    'num_classes': 6
                }
            }, MODEL_SAVE)
            print(f"  ✓ Saved best model (val loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

        # Optional: Print predictions for a few samples to see if learning
        if epoch == 0 or (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                x_sample, y_sample = next(iter(val_loader))
                x_sample, y_sample = x_sample.to(device), y_sample.to(device)
                preds = model(x_sample)

                print(f"  Sample predictions (first 3):")
                for i in range(min(3, len(preds))):
                    pred_str = ', '.join([f'{p:.2f}' for p in preds[i]])
                    true_str = ', '.join([f'{t:.2f}' for t in y_sample[i]])
                    print(f"    Pred: [{pred_str}]")
                    print(f"    True: [{true_str}]")

        print("-" * 50)

    checkpoint = torch.load(MODEL_SAVE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nTraining complete!")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print(f"Model saved to: {MODEL_SAVE}")


if __name__ == "__main__":
    train()