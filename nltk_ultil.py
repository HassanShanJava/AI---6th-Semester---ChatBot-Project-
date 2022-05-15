import nltk
nltk.download("punkt")
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer=PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(token_sentence, all_word):
    token_sentence=[stem(w) for w in token_sentence]
    bag=np.zeros(len(all_word), dtype=np.float32)
    for index,w in enumerate(all_word):
        if w in token_sentence:
            bag[index]=1.0
    return bag

    
"""
sent=['hello','how','are','you']
word=['hi','hello','!','you','bye','thank','cool']
bag=bag_of_words(sent,word)
print(bag)
"""

