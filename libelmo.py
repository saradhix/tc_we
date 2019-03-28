
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
name='Elmo v2'
module_url = "https://tfhub.dev/google/elmo/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# Import the Universal Sentence Encoder's TF Hub module
elmo = hub.Module(module_url, trainable=True)

# Compute a representation for each message, showing various lengths supported.

def get_vector(sentence):
    messages = [sentence]

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = elmo(messages)

    return message_embeddings[0]


def get_vectors(sentences):
    batch_size = 2*1024
    num_sentences = len(sentences)
    num_batches = int(num_sentences/batch_size)
    print("Sentences=", num_sentences, "Batches=", num_batches)
    for i in range(num_batches+1):
      print("Calling gvb with i=", i)
      batch_embeddings = get_vectors_batch(sentences[i*batch_size:(i+1)*batch_size])
      if i == 0:
        embeddings = batch_embeddings
      else:
        embeddings = np.concatenate((embeddings, batch_embeddings), axis=0)
      print(embeddings.shape)
    return embeddings

def get_vectors_batch(messages):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(elmo(messages, signature="default"))
        return message_embeddings



if __name__ == "__main__":
  sentence = "I am a sentence for which I would like to get its embedding."
  print(get_vector(sentence))
  print( get_vectors([sentence]))
