from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=70_000,
    min_frequency=2,
    special_tokens=[
        "<pad>", "<unk>", "<bos>",
        "<|user|>", "<|assistant|>",
        "<eos>", "<personality>",
        "[WIKI]", "[BOOK]", "[MATH]", "[CODE]"
    ]
)

tokenizer.train(["../data/corpus.txt"], trainer)
tokenizer.save("trained_tokenizer.json")