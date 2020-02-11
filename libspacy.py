from spacy.parts_of_speech import ADV, ADJ, VERB, NOUN
import spacy
from tqdm import tqdm

nlp = spacy.load('en')
#probs = [lex.prob for lex in nlp.vocab]
#probs.sort()
def get_parsed(sentence):
  return nlp(sentence)

def get_nouns(sentence, parsed=None):
  nouns = set()
  if parsed is None:
    parsed = nlp(sentence)
  for token in parsed:
    if token.pos == spacy.parts_of_speech.NOUN:
      nouns.add(token.string)
  return list(nouns)

def get_adjs(sentence,parsed=None):
  adjs = set()
  if parsed is None:
    parsed = nlp(sentence)
  for token in parsed:
    if token.pos == spacy.parts_of_speech.ADJ:
      adjs.add(token.string)
  return list(adjs)

def get_advs(sentence, parsed=None):
  advs = set()
  if parsed is None:
    parsed = nlp(sentence)
  for token in parsed:
    if token.pos == spacy.parts_of_speech.ADV:
      advs.add(token.string)
  return list(advs)

def get_verbs(sentence, parsed=None):
  verbs = set()
  if parsed is None:
    parsed = nlp(sentence)
  for token in parsed:
    if token.pos == spacy.parts_of_speech.VERB:
      verbs.add(token.string)
  return list(verbs)

def get_nes(sentence, parsed=None):
  if parsed is None:
    parsed = nlp(sentence)
  nes = [ i.label_ for i in parsed.ents]
  return nes

#Gives a tuple of counts in this sequence Noun, Verb, Adj, Adv
def get_pos_counts(sentence, parsed=None):
  pos_counts=[0 for i in range(4)]
  if parsed is None:
    parsed = nlp(sentence)
  for token in parsed:
    if token.pos == spacy.parts_of_speech.NOUN:
      pos_counts[0] +=1
    if token.pos == spacy.parts_of_speech.VERB:
      pos_counts[1] +=1
    if token.pos == spacy.parts_of_speech.ADJ:
      pos_counts[2] +=1
    if token.pos == spacy.parts_of_speech.ADV:
      pos_counts[3] +=1
    #if token.pos == spacy.parts_of_speech.DET:
    #  pos_counts[4] +=1
    #if token.pos == spacy.parts_of_speech.PUNCT:
    #  pos_counts[5] +=1
    #if token.pos == spacy.parts_of_speech.CONJ:
    #  pos_counts[6] +=1
  return pos_counts

def get_noun_verb_pos(sentence, parsed=None):
  ret=[]
  if parsed is None:
    parsed = nlp(sentence)
  for token in parsed:
    if token.pos == spacy.parts_of_speech.NOUN:
      ret.append('N')
    if token.pos == spacy.parts_of_speech.VERB:
      ret.append('V')

  return ''.join(ret)

def get_nsubj(sentence, parsed=None):
  if parsed is None:
    parsed = nlp(sentence.decode('utf-8'))
  return [ i for i in parsed if i.dep_ == "nsubj"]

def get_noun_phrases(sentence, parsed=None):
  ret=[]
  if parsed is None:
    parsed = nlp(sentence.decode('utf-8'))
  for np in parsed.noun_chunks:
    ret.append(np.text)
  return ret

def get_vector(sentence, parsed=None):
  if parsed is None:
    parsed = nlp(sentence.decode('utf-8'))
  return parsed.vector
def get_vectors(sentences):
  final_vectors=[]
  for sentence in tqdm(sentences):
    vector = get_vector(sentence)
    final_vectors.append(vector)
  return final_vectors

def get_nouns_vector(sentence, parsed=None):
  if parsed is None:
    parsed = nlp(sentence.decode('utf-8'))
  nouns_vector = nlp(' '.join([ token.text for token in parsed if token.pos == spacy.parts_of_speech.NOUN]).decode('utf-8')).vector
  return nouns_vector

def get_verbs_vector(sentence,parsed=None):
  if parsed is None:
    parsed = nlp(sentence.decode('utf-8'))
  vector = nlp(' '.join([ token.text for token in parsed if token.pos == spacy.parts_of_speech.VERB]).decode('utf-8')).vector
  return vector

def get_adverbs_vector(sentence,parsed=None):
  if parsed is None:
    parsed = nlp(sentence.decode('utf-8'))
  vector = nlp(' '.join([ token.text for token in parsed if token.pos == spacy.parts_of_speech.ADV]).decode('utf-8')).vector
  return vector

def get_pos(sentence, parsed=None):
  if parsed is None:
    parsed = nlp(sentence.decode('utf-8'))
  for a in parsed:
    print( a, a.pos, a.pos_, a.tag, a.tag_)

def has_superlative(sentence, parsed=None):
  if parsed is None:
    parsed = nlp(sentence.decode('utf-8'))

  return sum([1 for i in parsed if i.tag_=='JJS'])
  
'''
s = "A healthy king lives happily"
print get_nsubj(s)
s = "I am very rich and beautiful girl"
print get_adjectives(s)
'''
'''
sentence = nlp(u'A healthy man lives happily')
print sentence
for token in sentence:
  print token, token.pos, is_adverb(token)
'''

'''
s='A happy dog barks happily'
print get_pos_counts(s)
'''
'''
s='Pizzas by drones : unmanned air delivery set to take off in New Zealand'
print has_superlative(s)
get_nouns_vector(s)
get_verbs_vector(s)
s='Cheetah is the fastest running land animal'
get_pos(s)
print has_superlative(s)
'''
