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
# 生成文本列表
def genlabellist(lablepath):
	lablefiles = {}
	fileids = []
	for (dirpath, dirnames, filenames) in os.walk(lablepath):
		for filename in filenames:
			if filename.endswith('.wav.trn'):
				filepath = os.sep.join([dirpath, filename])
				fileid = filename.strip('.wav.trn')
				lablefiles[fileid] = filepath
				fileids.append(fileid)
	return lablefiles,fileids

# 对文本文件提取提取标注和数据
def extractlabel(filename):
	fileid = filename.strip('.wav.trn')
	textfile = open(filename, 'r+', encoding='utf-8')
	text = []
	for content in textfile:
		content = content.strip('\n')
		text.append(content)
	return text

# 生成thchs30训练用的数据，input.txt和lable.txt两个文件
def make_text(lablepath = 'E:\\Data\\data_thchs30\\data'):
	lablefiles,fileids = genlabellist(lablepath)
	lable = {}
	pinyin = {}
	savefile = open('lm_lable.txt', 'w+', encoding='UTF-8')
	savefile2 = open('lm_input.txt', 'w+', encoding='UTF-8')
	for fileid in fileids:
		lablepath = lablefiles[fileid]
		text = extractlabel(lablepath)
		# 将汉字变为每一个字都为一个标注
		text_0 = text[0]
		text_0 = text_0.split(' ')
		text_0 = ''.join(text_0)
		text_00 = ''
		for x in text_0:
			text_00 = text_00 + x + ' '
		print(text_00)
		str1 = fileid + ' ' + text_00
		str2 = fileid + ' ' + text[1]
		savefile.write(str1 + "\n")
		savefile2.write(str2 + "\n")



# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&e-mail:	hit_master@163.com
'''
# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	make_text()