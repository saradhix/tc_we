from tqdm import tqdm
import numpy as np
from nltk.tokenize import word_tokenize
name='LexVec'
embeddings_index = {}
print("Loading LexVec Word Vectors")
#f = open('../glove.840B/glove.840B.300d.txt')
f = open('/data/lexvec/lexvec.commoncrawl.300d.W.pos.vectors')
f.__next__() #Ignore the firs line
for line in tqdm(f,total=2000000, desc="#Vectors"):
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()
print("Done")
max_features=len(coefs)

def get_all_embeddings():
  return embeddings_index
def get_vectors(sentences):
  final_vectors=[]
  for sentence in tqdm(sentences):
    vector = get_vector(sentence)
    final_vectors.append(vector)
  return final_vectors
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

def get_vector2(sentence):
  final_vector = np.zeros(max_features)
  total = 0
  words = sentence.split(' ')
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
