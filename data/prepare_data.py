from datasets import load_dataset
from collections import defaultdict
import re

def clean_text(text):
    text = text.strip()
    text.replace("\r", "")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def reconstruct_conversations(dataset):

    tree_messages = defaultdict(list)

    for example in dataset:
        tree_id = example["message_tree_id"]
        tree_messages[tree_id].append(example)

    conversations = []

    for tree_id, messages in tree_messages.items():
        sorted_messages = sorted(messages, key=lambda x: x["message_id"])

        conversation_text = ""
        for message in sorted_messages:
            role = message["role"]
            text = clean_text(message["text"])

            if role == "prompter":
                conversation_text += f'<|user|>: {text}\n'
            elif role == "assistant":
                conversation_text += f'<|assistant|>: {text}\n'

        if conversation_text.strip():
            conversations.append({
                "text": conversation_text.strip(),
                "tree_id": tree_id,
                "message_count": len(messages)
            })

    return conversations

ds = load_dataset('OpenAssistant/oasst1')
train_conversations = reconstruct_conversations(ds['train'])
val_conversations = reconstruct_conversations(ds['validation'])

print(f"Reconstructed {len(train_conversations)} training conversations")
print(f"Reconstructed {len(val_conversations)} validation conversations")

if train_conversations:
    print("\nSample conversation:")
    print(train_conversations[0]['text'])