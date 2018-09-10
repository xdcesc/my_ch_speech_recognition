# -----------------------------------------------------------------------------------------------------
'''
&usage:		CNN-CTC的中文语音识别模型
@author:	hongwen sun
#feat_in:	fbank[800,200]
#net_str:	cnn32*2 -> cnn64*2 -> cnn128*6 -> dense*2 -> softmax -> ctc_cost
'''
# -----------------------------------------------------------------------------------------------------
import os
import random
import sys
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
from scipy.fftpack import fft
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
from extra_utils.GetData import get_data



# -----------------------------------------------------------------------------------------------------
'''
&usage:		[net model]构件网络结构，用于最终的训练和识别
'''
# -----------------------------------------------------------------------------------------------------
# 被creatModel调用，用作ctc损失的计算
def ctc_lambda(args):
	labels, y_pred, input_length, label_length = args
	y_pred = y_pred[:, :, :]
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# 构建网络结构，用于模型的训练和识别
def creatModel():
	input_data = Input(name='the_input', shape=(800, 200, 1))
	# 800,200,32
	layer_h1 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(input_data)
	layer_h1 = BatchNormalization(mode=0,axis=-1)(layer_h1)
	layer_h2 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1)
	layer_h2 = BatchNormalization(axis=-1)(layer_h2)
	layer_h3 = MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(layer_h2)
	# 400,100,64
	layer_h4 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3)
	layer_h4 = BatchNormalization(axis=-1)(layer_h4)
	layer_h5 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4)
	layer_h5 = BatchNormalization(axis=-1)(layer_h5)
	layer_h5 = MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(layer_h5)
	# 200,50,128
	layer_h6 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h5)
	layer_h6 = BatchNormalization(axis=-1)(layer_h6)
	layer_h7 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6)
	layer_h7 = BatchNormalization(axis=-1)(layer_h7)
	layer_h7 = MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(layer_h7)
	# 100,25,128
	layer_h8 = Conv2D(128, (1,1), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7)
	layer_h8 = BatchNormalization(axis=-1)(layer_h8)
	layer_h9 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h8)
	layer_h9 = BatchNormalization(axis=-1)(layer_h9)
	# 100,25,128
	layer_h10 = Conv2D(128, (1,1), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h9)
	layer_h10 = BatchNormalization(axis=-1)(layer_h10)
	layer_h11 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h10)
	layer_h11 = BatchNormalization(axis=-1)(layer_h11)
	# Reshape层
	layer_h12 = Reshape((100, 3200))(layer_h11) 
	# 全连接层
	layer_h13 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h12)
	layer_h13 = BatchNormalization(axis=1)(layer_h13)
	layer_h14 = Dense(1449, use_bias=True, kernel_initializer='he_normal')(layer_h13)
	output = Activation('softmax', name='Activation0')(layer_h14)
	model_data = Model(inputs=input_data, outputs=output)
	# ctc层
	labels = Input(name='the_labels', shape=[50], dtype='float32')
	input_length = Input(name='input_length', shape=[1], dtype='int64')
	label_length = Input(name='label_length', shape=[1], dtype='int64')
	loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')([labels, output, input_length, label_length])

	model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
	model.summary()
	ada_d = Adadelta(lr=0.01, rho=0.95, epsilon=1e-06)
	#model=multi_gpu_model(model,gpus=2)
	model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=ada_d)
	#test_func = K.function([input_data], [output])
	print("model compiled successful!")
	return model, model_data


# -----------------------------------------------------------------------------------------------------
'''
&usage:		模型的解码，用于将数字信息映射为拼音
'''
# -----------------------------------------------------------------------------------------------------
# 对model预测出的softmax的矩阵，使用ctc的准则解码，然后通过字典num2word转为文字
def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = 100;
	r = K.ctc_decode(result, in_len, greedy = True, beam_width=1, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text


# -----------------------------------------------------------------------------------------------------
'''
&usage:		模型的训练
'''
# -----------------------------------------------------------------------------------------------------
# 训练模型
def train(datapath = 'data/',
		batch_size = 4, 
		steps_per_epoch = 1000, 
		epochs = 1):
	# 准备训练所需数据
	p = get_data(datapath = datapath, read_type = 'train', batch_size = batch_size)
	yielddatas = p.data_generator()
	# 导入模型结构，训练模型，保存模型参数
	model, model_data = creatModel()
	if os.path.exists('speech_model/model_cnn_fbank.mdl'):
		model.load_weights('speech_model/model_cnn_fbank.mdl')
	model.fit_generator(yielddatas, steps_per_epoch=steps_per_epoch, epochs=1)
	model.save_weights('speech_model/model_cnn_fbank.mdl')


# -----------------------------------------------------------------------------------------------------
'''
&usage:		模型的测试，看识别结果是否正确
'''
# -----------------------------------------------------------------------------------------------------
# 测试模型
def test(datapath = 'data/',
		batch_size = 1):
	# 准备测试数据，以及生成字典
	p = get_data(datapath = datapath, read_type = 'test', batch_size = batch_size)
	num2word = p.label_dict
	yielddatas = p.data_generator()
	# 载入训练好的模型，并进行识别
	model, model_data = creatModel()
	model.load_weights('speech_model/model_cnn_fbank.mdl')
	result = model_data.predict_generator(yielddatas, steps=1)
	print(result.shape)
	# 将数字结果转化为文本结果
	result, text = decode_ctc(result, num2word)
	print('数字结果： ', result)
	print('文本结果：', text)


# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&e-mail:	hit_master@163.com
'''
# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	# 通过python gru_ctc_am.py [run type]进行测试
	#run_type = sys.argv[1]
	run_type = 'train'
	if run_type == 'test':
		test()
	elif run_type == 'train':
		for x in range(10):
			train()
			print('there is ', x, 'epochs')
			test()
