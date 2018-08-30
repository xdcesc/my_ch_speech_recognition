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
	words = ['_'] + words
	wordsize = len(words)
	word2num = dict(zip(words, range(wordsize)))
	num2word = dict(zip(range(wordsize), words))
	return word2num, num2word

# 文本转化为数字,生成文本到数字的字典，数字到文本的字典
def text2num(textfile_path, encoding=''):
	lexcion,num2word = gendict(textfile_path, encoding)
	word2num = lambda word:lexcion.get(word, 0)
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


# 构建网络结构，用于模型的训练和识别
def creatModel():
	# input num: 1344+1		lable num: 4463+1
	input_data = Input(name='the_input', shape=(50, 1345))
	layer_h1 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(input_data)
	layer_h1 = Dropout(0.2)(layer_h1)
	layer_h2 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h1)
	layer_h2 = Dropout(0.2)(layer_h2)
	layer_h3 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h2)
	layer_h4_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal', dropout=0.3)(layer_h3)
	layer_h4_2 = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', dropout=0.3)(layer_h3)
	layer_h4 = add([layer_h4_1, layer_h4_2])
	layer_h5 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h4)
	layer_h5 = Dropout(0.2)(layer_h5)
	layer_h6 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h5)
	layer_h6 = Dropout(0.2)(layer_h6)
	layer_h7 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h6)
	layer_h7 = Dropout(0.2)(layer_h7)
	layer_h8 = Dense(4464, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h7)
	output = Activation('softmax', name='Activation0')(layer_h8)
	model = Model(inputs=input_data, outputs=output)
	model.summary()
	ada_d = Adadelta(lr=0.01, rho=0.95, epsilon=1e-06)
	#model=multi_gpu_model(model,gpus=2)
	model.compile(loss='categorical_crossentropy', optimizer=ada_d)
	#test_func = K.function([input_data], [output])
	print("model compiled successful!")
	return model

# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&e-mail:	hit_master@163.com
'''
# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	input_dict, input2num, num2input = text2num('input.txt',encoding='gbk')
	lable_dict, lable2num, num2lable = text2num('lable.txt', encoding='utf-8')
	for x in input_dict:
		pass
	print(input2num)
	print(lable2num)
	