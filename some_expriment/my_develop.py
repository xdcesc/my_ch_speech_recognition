import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from collections import Counter
from python_speech_features import mfcc
import tensorflow as tf
import numpy as np
from keras.layers import Reshape
import random


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
	label_vector = list(map(to_num,['hello','hello','hello','hi','hi','hi','you', 'you']))
	print(label_vector)
	print('\nall words i use :\n',all_word)
	print('\nafter counter:\n',counter)
	print('\nafter sort:\n',words)
	print('\nthe words size: ',words_size)
	print('\nthe words num map:\n',word_num_map)

#读取音频文件并提取特征
def readwav(audio_filename, n_input=26):
	# 读取音频文件
	fs, audio = wav.read(audio_filename)
    # 提取mfcc数值
	orig_inputs = mfcc(audio, samplerate=fs, numcep=26)
#	for i in range(orig_inputs.shape[0]):
#		plt.plot(orig_inputs[i])
#		plt.pause(0.01)
#		plt.close()

#测试format功能
def train(loop):
	section = '\n{0:=^40}\n'
	print(section.format('开始训练'))

#测试minimum功能
def minimum(a,b):
	y = tf.minimum(a,b)
	print(y)

def nparray():
	x = [100,10]
	y = x[:3]
	y = np.asarray(y, dtype=np.float32)
	print(y)

#查看矩阵形状变换
def dimshape():
	y = [[[1,2],[1,2],[1,2]]]
	y = np.array(y)
	Y = np.expand_dims(y, axis=3)
	print(Y)
	y = tf.transpose(y,[1,0,2])
	y = tf.reshape(y,[-1,2])
	print(y)

#生成list
def generate_list():
	i = 1
	x = list(np.zeros(4-i))
	print(len(x))


def random_num(batch_size=32):
	for i in range(batch_size):
		ran_num = random.randint(0,32 - 1)
		print(ran_num)

def inputlen(input_length):
	input_length = input_length // 8
	print(input_length)


def generator():
	fileObject = open('../wav.scp', 'r')
	
	for i in fileObject.readlines():
		i = i.strip('\n')
		fs, audio = wav.read(i)
		mfcc_feat = mfcc(audio, samplerate=fs, numcep=13)
		yield mfcc_feat

if __name__ == '__main__':
	mfccfile = generator()
	for i in mfccfile:
		print(i)