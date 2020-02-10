
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_hub as hub
import numpy as np
name='Elmo v3'
module_url = "https://tfhub.dev/google/elmo/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# Import the Universal Sentence Encoder's TF Hub module
elmo = hub.Module(module_url, trainable=True)

# Compute a representation for each message, showing various lengths supported.
def get_vectors(sentences):
    embeddings = elmo(sentences, signature="default", as_dict=True)["default"]
    return np.array(embeddings)


if __name__ == "__main__":
  sentence = "I am a sentence for which I would like to get its embedding."
  print( get_vectors([sentence]))
