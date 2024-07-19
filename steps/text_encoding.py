import torch
import gensim.downloader as api

# Load pre-trained Word2Vec model
wv_model = api.load("glove-wiki-gigaword-50")

def encode_text(text):
    """Encode text using Word2Vec or GloVe"""
    words = text.split()
    embeddings = []
    for word in words:
        try:
            embedding = wv_model.get_vector(word)
            embeddings.append(embedding)
        except KeyError:
            # If the word is not in the vocabulary, skip it
            pass
    if len(embeddings) == 0:
        # If no words are found, return a random vector
        return torch.randn(wv_model.vector_size)  
    return torch.mean(torch.tensor(embeddings), dim=0)