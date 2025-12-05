# check_dataset_bias.py
import json
from collections import Counter


def check_dataset():
    with open("emotions.txt", "r") as f:
        lines = f.readlines()

    emotion_names = ["friendly/kind", "calm/serious", "angry/mean",
                     "sad/depressed", "excited/fun", "curious/motivated"]

    # Count dominant emotions
    dominant_counts = [0] * 6
    total_samples = len(lines)

    for line in lines:
        data = json.loads(line)
        label = data["label"]

        # Find which emotion is strongest
        strongest_idx = max(range(6), key=lambda i: label[i])
        dominant_counts[strongest_idx] += 1

    print("Dataset analysis:")
    print(f"Total samples: {total_samples}")
    print("\nDominant emotion distribution:")
    for i in range(6):
        percentage = dominant_counts[i] / total_samples * 100
        print(f"  {emotion_names[i]}: {dominant_counts[i]} samples ({percentage:.1f}%)")

    # Check if friendly is overrepresented
    friendly_percentage = dominant_counts[0] / total_samples * 100
    if friendly_percentage > 40:  # Should be ~16.7% for balanced
        print(f"\n⚠️ PROBLEM: 'friendly/kind' is {friendly_percentage:.1f}% of dataset!")
        print("  Should be ~16.7% for perfect balance")

    # Also check average values
    print("\nAverage label values:")
    avg_values = [0.0] * 6
    for line in lines:
        data = json.loads(line)
        label = data["label"]
        for i in range(6):
            avg_values[i] += label[i]

    for i in range(6):
        avg_values[i] /= total_samples
        print(f"  {emotion_names[i]}: {avg_values[i]:.3f}")


check_dataset()