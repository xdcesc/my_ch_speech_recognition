import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from collections import Counter
from python_speech_features import mfcc
import tensorflow as tf
import numpy as np



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


def readwav(audio_filename, n_input=26):
	# 读取音频文件
	fs, audio = wav.read(audio_filename)
    # 提取mfcc数值
	orig_inputs = mfcc(audio, samplerate=fs, numcep=26)
#	for i in range(orig_inputs.shape[0]):
#		plt.plot(orig_inputs[i])
#		plt.pause(0.01)
#		plt.close()

def train(loop):
	section = '\n{0:=^40}\n'
	print(section.format('开始训练'))

def minimum(a,b):
	y = tf.minimum(a,b)
	print(y)



def main():
#	empty_mfcc = np.zero(5)
#	empty = list(empty_mfcc for em)
#	print(range(139))
	
	x = [100,10]
	y = x[:3]
	y = np.asarray(y, dtype=np.float32)
	print(y)
#	for i in range(1):
#		print(i)

if __name__ == '__main__':
	main()