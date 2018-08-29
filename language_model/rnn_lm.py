# -----------------------------------------------------------------------------------------------------
'''
&usage:		RNN的中文语语言模型
@author:	hongwen sun
#net_str:	
'''
# -----------------------------------------------------------------------------------------------------
import os
import random
import sys
import numpy as np
import scipy.io.wavfile as wav
from collections import Counter
from python_speech_features import mfcc
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape 
from keras.layers import Conv1D,LSTM,MaxPooling1D, Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D
from keras.layers.merge import add, concatenate
from keras import backend as K
from keras.optimizers import SGD, Adadelta
from keras.layers.recurrent import GRU
from keras.preprocessing.sequence import pad_sequences


# -----------------------------------------------------------------------------------------------------
'''
&usage:		[text]对文本文件进行处理，生成拼音训练数据、汉字训练数据
'''
# -----------------------------------------------------------------------------------------------------


# 文本转化为数字
def make_num_lable(textfile_path):
	lexcion,num2word = gendict(textfile_path)
	word2num = lambda word:lexcion.get(word, None)
	textfile = open(textfile_path, 'r+', encoding='utf-8')
	content_dict = {}
	for content in textfile.readlines():
		content = content.strip('\n')
		cont_id = content.split(' ',1)[0]
		content = content.split(' ',1)[1]
		content = content.split(' ')
		content = list(map(word2num,content))
		#add_num = list(np.zeros(50-len(content)))
		content = content + add_num
		content_dict[cont_id] = content

	return content_dict,lexcion


# -----------------------------------------------------------------------------------------------------
'''
&usage:		[text]对文本标注文件进行处理，包括生成拼音到数字的映射，以及将拼音标注转化为数字的标注转化
'''
# -----------------------------------------------------------------------------------------------------
# 利用训练数据生成词典
def gendict(textfile_path, encoding):
	dicts = []
	textfile = open(textfile_path, 'r+', encoding=encoding)
	for content in textfile:
		content = content.strip('\n')
		content = content.split(' ',1)[1]
		content = content.split(' ')
		dicts += (word for word in content)
	counter = Counter(dicts)
	words = sorted(counter)
	wordsize = len(words)
	word2num = dict(zip(words, range(wordsize)))
	num2word = dict(zip(range(wordsize), words))
	return word2num, num2word

# 文本转化为数字,生成文本到数字的字典，数字到文本的字典
def text2num(textfile_path, encoding=''):
	lexcion,num2word = gendict(textfile_path, encoding)
	word2num = lambda word:lexcion.get(word, None)
	textfile = open(textfile_path, 'r+', encoding=encoding)
	content_dict = {}
	for content in textfile.readlines():
		content = content.strip('\n')
		cont_id = content.split(' ',1)[0]
		content = content.split(' ',1)[1]
		content = content.split(' ')
		content = list(map(word2num,content))
		add_num = list(np.zeros(50-len(content)))
		content = content + add_num
		content_dict[cont_id] = content
	return content_dict, lexcion, num2word




# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&e-mail:	hit_master@163.com
'''
# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	input_dict, input2num, num2input = text2num('input.txt',encoding='gbk')
	lable_dict, lable2num, num2lable = text2num('lable.txt', encoding='utf-8')
	print(lable_dict['A11_103'])
	print(input_dict['A11_103'])