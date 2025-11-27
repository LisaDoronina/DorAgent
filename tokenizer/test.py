from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("trained_tokenizer.json")

encoded = tokenizer.encode("Hello world! I am a trained tokenizer! 🥰❤️ 🌟 🤗")
print(encoded.tokens)
print(encoded.ids)
text = tokenizer.decode(encoded.ids)

print("\n=== Readable Format ===")
readable_tokens = [token.replace('Ġ', ' ') for token in encoded.tokens]
print("Tokens with spaces: ", readable_tokens)
decoded = tokenizer.decode(encoded.ids)
print("Decoded back: ", repr(decoded))