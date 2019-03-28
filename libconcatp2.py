
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
name='Concatenated p means'
np.set_printoptions(threshold=np.nan)
module_url = "https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/tf-hub/monolingual/1" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Compute a representation for each message, showing various lengths supported.

def get_vector(sentence):
    with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()]) 
      return session.run(embed([sentence.lower()]))

def get_vectors(sentences):
    batch_size = 32768
    #batch_size = 1024
    num_sentences = len(sentences)
    num_batches = int(num_sentences/batch_size)
    print("Number of batches=", num_batches)
    for i in range(num_batches+1):
      print("Calling gvb with i=", i)
      batch_embeddings = get_vectors_batch(sentences[i*batch_size:(i+1)*batch_size])
      if i == 0:
        embeddings = batch_embeddings
      else:
        embeddings = np.concatenate((embeddings, batch_embeddings), axis=0)
      print(embeddings.shape)
    return embeddings


def get_vectors_batch(sentences):
    message_embeddings = []
    with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()]) 
      for sentence in sentences:
        message_embeddings.append(sentence.lower())
      return session.run(embed(message_embeddings))



if __name__ == "__main__":
  sentence = "I am a sentence for which I would like to get its embedding."
  #print(get_vector(sentence))
  print(np.array(get_vectors([sentence])))
