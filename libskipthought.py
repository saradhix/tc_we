from tqdm import tqdm
name='Skip Thought'
import numpy as np
from nltk.tokenize import word_tokenize
embeddings_index = {}
print("Loading Skip Thought Word Vectors")
emb_file = '/home/saradhix/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/embeddings.npy'
vocab_file ='/home/saradhix/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/vocab.txt'
vocab=[]
embeddings = np.load(emb_file)
embeddings_index={}
fp=open(vocab_file,'r')
for line in fp:
  vocab.append(line.strip())
fp.close()
for (word, coefs) in zip(vocab[:-1], embeddings):
  embeddings_index[word] = coefs
print("Done")
max_features=len(coefs)

def get_all_embeddings():
  return embeddings_index

def get_vector(sentence):
  final_vector = np.zeros(max_features)
  total = 0
  words = [word for word in word_tokenize(sentence) if word.isalpha()]
  for word in words:
    vector = embeddings_index.get(word.lower())
    if vector is not None:
      total = total+1
      final_vector = final_vector + vector
  if total ==0:
    total=1 #To prevent div by 0
  final_vector = final_vector / total
  return final_vector

def get_most_similar(word, howmany):
  word_vector = get_vector(word)

  similarities = {}
  for w, v in tqdm(embeddings_index.items()):
    sim = word_vector.dot(v)
    similarities[w]=sim

  return sorted(similarities.items(), key=lambda x:x[1], reverse=True)[:howmany]


def get_most_similar_by_vector(word_vector, howmany):
  similarities = {}
  norm_word_vector = np.linalg.norm(word_vector)
  for w, v in tqdm(embeddings_index.items()):
    norm_v = np.linalg.norm(v)
    sim = word_vector.dot(v)/(norm_v*norm_word_vector)
    similarities[w]=sim

  return sorted(similarities.items(), key=lambda x:x[1], reverse=True)[:howmany]
if __name__ == "__main__":
  sentence = "I am a sentence for which I would like to get its embedding."
  #print(get_vector(sentence))
  print(np.array(get_vectors([sentence])))
