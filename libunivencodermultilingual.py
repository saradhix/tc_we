import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
name='Universal Encoder Multilingual(Google)'
module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.KerasLayer(module_url)

# Compute a representation for each message, showing various lengths supported.

def get_vectors(sentences):
    embeddings = embed(sentences)
    return embeddings.numpy()

if __name__ == "__main__":
  sentence = "I am a sentence for which I would like to get its embedding."
  print( get_vectors([sentence]))
