import nltk
import torch
from models import InferSent
import numpy as np

#First, download the InferSent pre-trained models by running this in the InferSent directory:
#curl -Lo encoder/infersent1.pkl https://s3.amazonaws.com/senteval/infersent/infersent1.pkl
#curl -Lo encoder/infersent2.pkl https://s3.amazonaws.com/senteval/infersent/infersent2.pkl
# Next, set the W2V_PATH variable below and clickbait sentences accordingly.
# output is numpy array of 4096 dim. space.
name="InferSent"
V=2
MODEL_PATH = '/data/InferSent/encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = '../fasttext/fasttextwiki.en.vec'
infersent.set_w2v_path(W2V_PATH)

def get_vectors_batch(sentences):
  sentences = [s.lower() for s in sentences]
  #sentences = ["Hello, I am bakhtiyar", "wow here is a cake for you!"]
  #infersent.build_vocab(sentences, tokenize=True)
  infersent.build_vocab_k_words(K=10000)
  embeddings = infersent.encode(sentences, tokenize =  True)
  return embeddings

def get_vectors(sentences):
    #batch_size = 32768*2
    batch_size = 256 
    num_sentences = len(sentences)
    num_batches = int(1.0*num_sentences/batch_size)
    print("Number of batches=", num_batches, "batch_size=", batch_size)
    for i in range(num_batches+1):
      print("Calling gvb with i=", i)
      batch_embeddings = get_vectors_batch(sentences[i*batch_size:(i+1)*batch_size])
      if i == 0:
        embeddings = batch_embeddings
      else:
        embeddings = np.concatenate((embeddings, batch_embeddings), axis=0)
      print(embeddings.shape)
    return embeddings




if __name__ == "__main__":
  sentence = "I am a sentence for which I would like to get its embedding."
  print(get_vectors([sentence])[0])
  print( get_vectors([sentence]))
