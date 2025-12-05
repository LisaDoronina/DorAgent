# create_proper_dataset.py
import json


def create_proper_dataset():
    """Create a properly balanced dataset with clear emotional signals"""

    samples = []

    # 1. CLEAR ANGRY examples (30)
    angry_examples = [
        ("I hate you so much!", [0.0, 0.1, 1.0, 0.2, 0.0, 0.0]),
        ("You're being completely unreasonable!", [0.0, 0.2, 0.95, 0.1, 0.0, 0.0]),
        ("This makes me furious!", [0.0, 0.1, 0.98, 0.3, 0.0, 0.0]),
        ("I can't stand your attitude!", [0.0, 0.3, 0.97, 0.2, 0.0, 0.0]),
        ("You're testing my patience!", [0.0, 0.4, 0.96, 0.1, 0.0, 0.0]),
        ("That was a terrible thing to do!", [0.0, 0.2, 0.95, 0.4, 0.0, 0.0]),
        ("I'm so angry right now!", [0.0, 0.1, 0.99, 0.3, 0.0, 0.0]),
        ("You're being disrespectful!", [0.0, 0.3, 0.96, 0.2, 0.0, 0.0]),
        ("This is completely unacceptable!", [0.0, 0.4, 0.97, 0.1, 0.0, 0.0]),
        ("I'm done with your excuses!", [0.0, 0.2, 0.98, 0.3, 0.0, 0.0]),
        ("You've crossed a line!", [0.0, 0.3, 0.95, 0.2, 0.0, 0.0]),
        ("I won't tolerate this behavior!", [0.0, 0.4, 0.96, 0.1, 0.0, 0.0]),
        ("You're impossible to work with!", [0.0, 0.2, 0.97, 0.3, 0.0, 0.0]),
        ("I'm boiling with rage!", [0.0, 0.1, 1.0, 0.4, 0.0, 0.0]),
        ("You're being selfish!", [0.0, 0.3, 0.95, 0.2, 0.0, 0.0]),
        ("This is infuriating!", [0.0, 0.2, 0.98, 0.3, 0.0, 0.0]),
        ("I'm seething with anger!", [0.0, 0.1, 0.99, 0.4, 0.0, 0.0]),
        ("You're creating unnecessary problems!", [0.0, 0.4, 0.96, 0.1, 0.0, 0.0]),
        ("I'm fed up with this!", [0.0, 0.3, 0.97, 0.3, 0.0, 0.0]),
        ("You're driving me crazy!", [0.0, 0.2, 0.98, 0.4, 0.0, 0.0]),
        ("This is ridiculous!", [0.0, 0.1, 0.95, 0.2, 0.0, 0.0]),
        ("I'm losing my patience!", [0.0, 0.3, 0.96, 0.3, 0.0, 0.0]),
        ("You're being difficult!", [0.0, 0.4, 0.95, 0.1, 0.0, 0.0]),
        ("I'm at my wit's end!", [0.0, 0.2, 0.97, 0.4, 0.0, 0.0]),
        ("You're ruining everything!", [0.0, 0.1, 0.98, 0.5, 0.0, 0.0]),
        ("This is beyond frustrating!", [0.0, 0.3, 0.96, 0.3, 0.0, 0.0]),
        ("I'm disgusted by this!", [0.0, 0.2, 0.97, 0.4, 0.0, 0.0]),
        ("You're not listening to me!", [0.0, 0.4, 0.95, 0.2, 0.0, 0.0]),
        ("I'm so disappointed!", [0.0, 0.3, 0.96, 0.5, 0.0, 0.0]),
        ("You're being thoughtless!", [0.0, 0.2, 0.95, 0.3, 0.0, 0.0]),
    ]

    for text, label in angry_examples:
        samples.append({"text": text, "label": label})

    # 2. CLEAR FRIENDLY examples (30) - but make them DIFFERENT from angry!
    friendly_examples = [
        ("I really appreciate your help!", [0.95, 0.3, 0.0, 0.0, 0.4, 0.2]),
        ("You're such a kind person!", [0.97, 0.2, 0.0, 0.0, 0.3, 0.1]),
        ("Thank you so much for everything!", [0.96, 0.4, 0.0, 0.0, 0.2, 0.1]),
        ("I'm so grateful to know you!", [0.98, 0.3, 0.0, 0.0, 0.5, 0.2]),
        ("You always make me smile!", [0.95, 0.1, 0.0, 0.0, 0.7, 0.3]),
        ("I value our friendship deeply!", [0.97, 0.4, 0.0, 0.0, 0.3, 0.2]),
        ("You're a true blessing!", [0.99, 0.2, 0.0, 0.0, 0.4, 0.3]),
        ("I admire your compassion!", [0.96, 0.5, 0.0, 0.0, 0.2, 0.4]),
        ("You're the best friend ever!", [0.98, 0.1, 0.0, 0.0, 0.8, 0.2]),
        ("Your support means everything!", [0.97, 0.3, 0.0, 0.0, 0.3, 0.1]),
        ("I'm lucky to have you in my life!", [0.99, 0.2, 0.0, 0.0, 0.6, 0.3]),
        ("You make the world better!", [0.96, 0.4, 0.0, 0.0, 0.5, 0.4]),
        ("I believe in you completely!", [0.95, 0.3, 0.0, 0.0, 0.4, 0.5]),
        ("You're stronger than you think!", [0.94, 0.5, 0.0, 0.0, 0.3, 0.6]),
        ("Your kindness is inspiring!", [0.97, 0.2, 0.0, 0.0, 0.2, 0.3]),
        ("I'm proud of your accomplishments!", [0.96, 0.3, 0.0, 0.0, 0.7, 0.4]),
        ("You always cheer people up!", [0.95, 0.1, 0.0, 0.0, 0.8, 0.3]),
        ("Your generosity is amazing!", [0.98, 0.4, 0.0, 0.0, 0.3, 0.5]),
        ("I appreciate you more than words!", [0.99, 0.3, 0.0, 0.0, 0.2, 0.2]),
        ("You have a heart of gold!", [0.97, 0.2, 0.0, 0.0, 0.5, 0.4]),
        ("You're making a difference!", [0.96, 0.5, 0.0, 0.0, 0.4, 0.6]),
        ("I'm always here for you!", [0.95, 0.4, 0.0, 0.0, 0.3, 0.3]),
        ("You deserve all happiness!", [0.98, 0.2, 0.0, 0.0, 0.7, 0.4]),
        ("You're an incredible person!", [0.99, 0.3, 0.0, 0.0, 0.4, 0.5]),
        ("Your friendship means the world!", [0.97, 0.4, 0.0, 0.0, 0.3, 0.3]),
        ("You handle everything gracefully!", [0.96, 0.6, 0.0, 0.0, 0.2, 0.4]),
        ("I'm glad we're friends!", [0.98, 0.2, 0.0, 0.0, 0.8, 0.3]),
        ("You're a ray of sunshine!", [0.95, 0.1, 0.0, 0.0, 0.9, 0.2]),
        ("You have a beautiful soul!", [0.97, 0.3, 0.0, 0.0, 0.3, 0.5]),
        ("Thank you for being you!", [0.99, 0.4, 0.0, 0.0, 0.5, 0.4]),
    ]

    for text, label in friendly_examples:
        samples.append({"text": text, "label": label})

    # 3. Add other emotions similarly...
    # [Add 30 examples each for calm, sad, excited, curious]

    # Save the dataset
    with open("emotions.txt", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created dataset with {len(samples)} samples")
    print("First sample:", samples[0])


create_proper_dataset()