import os
import random
import sys
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
from collections import Counter
from python_speech_features import mfcc
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization
from keras.layers import Conv1D,LSTM,MaxPooling1D, Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D
from keras.layers.merge import add, concatenate
from keras import backend as K
from keras.optimizers import SGD, Adadelta
from keras.layers.recurrent import GRU
from keras.preprocessing.sequence import pad_sequences
from keras.utils import multi_gpu_model











# -----------------------------------------------------------------------------------------------------
'''
&usage:		[audio]对音频文件进行处理，包括生成总的文件列表、特征提取等
'''
# -----------------------------------------------------------------------------------------------------
# 生成音频列表
def genwavlist(wavpath):
	wavfiles = {}
	fileids = []
	for (dirpath, dirnames, filenames) in os.walk(wavpath):
		for filename in filenames:
			if filename.endswith('.wav'):
				filepath = os.sep.join([dirpath, filename])
				fileid = filename.strip('.wav')
				wavfiles[fileid] = filepath
				fileids.append(fileid)
	return wavfiles,fileids

# 对音频文件提取mfcc特征
def compute_mfcc(file):
	fs, audio = wav.read(file)
	mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)
	mfcc_feat = mfcc_feat[::3]
	mfcc_feat = np.transpose(mfcc_feat)  
	mfcc_feat = pad_sequences(mfcc_feat, maxlen=500, dtype='float', padding='post', truncating='post').T
	return mfcc_feat

# 获取信号的时频图
def compute_fbank(file):
	fs, audio = wav.read(file)
	# wav波形 加时间窗以及时移10ms
	time_window = 25 # 单位ms
	window_length = fs / 1000 * time_window # 计算窗长度的公式，目前全部为400固定值
	wav_arr = np.array(wavsignal)
	#wav_length = len(wavsignal[0])
	wav_length = wav_arr.shape[1]
	range0_end = int(len(wavsignal[0])/fs*1000 - time_window) // 10 # 计算循环终止的位置，也就是最终生成的窗数
	data_input = np.zeros((range0_end, 200), dtype = np.float) # 用于存放最终的频率特征数据
	data_line = np.zeros((1, 400), dtype = np.float)
	for i in range(0, range0_end):
		p_start = i * 160
		p_end = p_start + 400
		data_line = wav_arr[0, p_start:p_end]	
		data_line = data_line * w # 加窗
		data_line = np.abs(fft(data_line)) / wav_length
		data_input[i]=data_line[0:200] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
	#print(data_input.shape)
	data_input = np.log(data_input + 1)
	data_input = data_input[::2]
	data_input = np.transpose(data_input)  
	data_input = pad_sequences(data_input, maxlen=800, dtype='float', padding='post', truncating='post').T	
	return data_input