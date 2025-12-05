import random

with open("emotions.txt", 'r') as f:
    lines = f.readlines()

random.shuffle(lines)

with open("emotions.txt", 'w') as f:
    f.writelines(lines)


print("dataset was shuffled successfully")