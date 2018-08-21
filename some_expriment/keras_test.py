import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from collections import Counter
from python_speech_features import mfcc
import tensorflow as tf
import numpy as np
from keras.layers import Reshape
import random
from keras.preprocessing.text import Tokenizer


#测试python字典的生成方式
def dict_test():
	all_word = []
	label = []
	# python字典生成方法，没有语音，只生成汉字对应了序号
	label = ['hello','hello','hello','hi','hi','hi','you', 'you']
	all_word += [word for word in label]
	counter = Counter(all_word)
	words = sorted(counter)
	words_size = len(words)
	word_num_map = dict(zip(words, range(words_size)))
	to_num = lambda word: word_num_map.get(word, 4)
	label_vector = list(map(to_num,['hello','you','hi','you']))
	print(label_vector)
	print('\nall words i use :\n',all_word)
	print('\nafter counter:\n',counter)
	print('\nafter sort:\n',words)
	print('\nthe words size: ',words_size)
	print('\nthe words num map:\n',word_num_map)


def get_pad_seq(textlines, maxlen=48):

    # saving
    tok_path = 'tokenizer.pickle'
    if not os.path.exists(tok_path):
        tok=Tokenizer(char_level=True)
        tok.fit_on_texts(text_lines)
        with open(tok_path, 'wb') as handle:
            pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('create tok')
    # loading
    else:
        with open(tok_path, 'rb') as handle:
            tok = pickle.load(handle)
            print('load tok')
    
    seq_lines =  tok.texts_to_sequences(text_lines[:])
    print('num of words,', len(tok.word_index.keys()))

    len_lines = pd.Series(map(lambda x: len(x), seq_lines))
    print('max_len',len_lines.max())

    def new_pad_seq(line, maxlen):
        return pad_sequences(line, maxlen=maxlen, padding='post', truncating='pre')
    
    lines = seq_lines[:]
    pad_lines = new_pad_seq(lines, maxlen)
    return pad_lines, tok





if __name__ == '__main__':


	text_lines = [['hello','hello','hello','hi','hi','hi','you', 'you']]
	tok=Tokenizer(char_level=True)
	tok.fit_on_texts(text_lines)
	seq_lines =  tok.texts_to_sequences(text_lines)
	
	print(seq_lines)