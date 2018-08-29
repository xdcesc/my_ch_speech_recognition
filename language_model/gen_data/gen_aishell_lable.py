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


# 生成aishell训练用的数据，仅有一个txt文件
def make_text():
	
	lable = {}
	pinyin = {}
	labelfile = open('aishell_transcript_v0.8.txt', 'r+', encoding='utf-8')
	savefile = open('aihell_lable.txt', 'w+', encoding='UTF-8')
	for content in labelfile:
		content = content.strip('\n')
		cont_id = content.split(' ',1)[0]
		text_0 = content.split(' ',1)[1]
		# 将汉字变为每一个字都为一个标注
		text_0 = text_0.split(' ')
		text_0 = ''.join(text_0)
		text_00 = ''
		for x in text_0:
			text_00 = text_00 + x + ' '
		print(text_00)
		str1 = cont_id + ' ' + text_00
		savefile.write(str1 + "\n")



# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&e-mail:	hit_master@163.com
'''
# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	make_text()
