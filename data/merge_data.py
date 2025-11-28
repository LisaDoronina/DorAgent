import numpy as np

wiki = np.load("../data/wiki_tokenized/wiki_data.npy")
books = np.load("../data/books_tokenized/books.npy")
conversations = np.load("../data/conversations_tokenized/conversations.npy")

all_data = np.concatenate((books, conversations, wiki), axis=0)
np.random.shuffle(all_data)

np.save("../data/final_training_data.npy", all_data)
