# -----------------------------------------------------------------------------------------------------
'''
&usage:		GRU-CTC的中文语音识别模型
@author:	hongwen sun
#net_str:	dense -> dense -> bi_gru -> dense -> dense -> softmax -> ctc_cost
'''
# -----------------------------------------------------------------------------------------------------
import os
import random
import sys
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
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
from keras.utils import multi_gpu_model


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
#进行配置，使用95%的GPU
config = tf.ConfigProto(allow_soft_placement=True)
#config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
#config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
set_session(tf.Session(config=config))
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


# -----------------------------------------------------------------------------------------------------
'''
&usage:		[text]对文本标注文件进行处理，包括生成拼音到数字的映射，以及将拼音标注转化为数字的标注转化
'''
# -----------------------------------------------------------------------------------------------------
# 利用训练数据生成词典
def gendict(textfile_path):
	dicts = []
	textfile = open(textfile_path,'r+')
	for content in textfile.readlines():
		content = content.strip('\n')
		content = content.split(' ',1)[1]
		content = content.split(' ')
		dicts += (word for word in content)
	counter = Counter(dicts)
	words = sorted(counter)
	wordsize = len(words)
	word2num = dict(zip(words, range(wordsize)))
	num2word = dict(zip(range(wordsize), words))
	return word2num, num2word #1176个音素

# 文本转化为数字
def text2num(textfile_path):
	lexcion,num2word = gendict(textfile_path)
	word2num = lambda word:lexcion.get(word, 0)
	textfile = open(textfile_path, 'r+')
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
	return content_dict,lexcion


# -----------------------------------------------------------------------------------------------------
'''
&usage:		[data]数据生成器构造，用于训练的数据生成，包括输入特征及标注的生成，以及将数据转化为特定格式
'''
# -----------------------------------------------------------------------------------------------------
# 将数据格式整理为能够被网络所接受的格式，被data_generator调用
def get_batch(x, y, train=False, max_pred_len=50, input_length=500):
    X = np.expand_dims(x, axis=3)
    X = x # for model2
#     labels = np.ones((y.shape[0], max_pred_len)) *  -1 # 3 # , dtype=np.uint8
    labels = y
    input_length = np.ones([x.shape[0], 1]) * ( input_length - 2 )
#     label_length = np.ones([y.shape[0], 1])
    label_length = np.sum(labels > 0, axis=1)
    label_length = np.expand_dims(label_length,1)
    inputs = {'the_input': X,
              'the_labels': labels,
              'input_length': input_length,
              'label_length': label_length,
              }
    outputs = {'ctc': np.zeros([x.shape[0]])}  # dummy data for dummy loss function
    return (inputs, outputs)

# 数据生成器，默认音频为thchs30\train,默认标注为thchs30\train.syllable,被模型训练方法fit_generator调用
def data_generate(wavpath, textfile, bath_size):
	wavdict,fileids = genwavlist(wavpath)
	#print(wavdict)
	content_dict,lexcion = text2num(textfile)
	genloop = len(fileids)//bath_size
	print("all loop :", genloop)
	while True:
		feats = []
		labels = []
		# 随机选择某个音频文件作为训练数据
		i = random.randint(0,genloop-1)
		for x in range(bath_size):
			num = i * bath_size + x
			fileid = fileids[num]
			# 提取音频文件的特征
			mfcc_feat = compute_mfcc(wavdict[fileid])
			feats.append(mfcc_feat)
			# 提取标注对应的label值
			labels.append(content_dict[fileid])
		# 将数据格式修改为get_batch可以处理的格式
		feats = np.array(feats)
		labels = np.array(labels)
		# 调用get_batch将数据处理为训练所需的格式
		inputs, outputs = get_batch(feats, labels)
		yield inputs, outputs


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
	input_data = Input(name='the_input', shape=(500, 26))
	layer_h1 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(input_data)
	layer_h1 = Dropout(0.2)(layer_h1)
	layer_h2 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h1)
	layer_h2 = Dropout(0.2)(layer_h2)
	layer_h3 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h2)
	layer_h4_1 = GRU(512, return_sequences=True, kernel_initializer='he_normal', dropout=0.3)(layer_h3)
	layer_h4_2 = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', dropout=0.3)(layer_h3)
	layer_h4 = add([layer_h4_1, layer_h4_2])
	layer_h5 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h4)
	layer_h5 = Dropout(0.2)(layer_h5)
	layer_h6 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h5)
	layer_h6 = Dropout(0.2)(layer_h6)
	layer_h7 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h6)
	layer_h7 = Dropout(0.2)(layer_h7)
	layer_h8 = Dense(1177, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h7)
	output = Activation('softmax', name='Activation0')(layer_h8)
	model_data = Model(inputs=input_data, outputs=output)
	#ctc
	labels = Input(name='the_labels', shape=[50], dtype='float32')
	input_length = Input(name='input_length', shape=[1], dtype='int64')
	label_length = Input(name='label_length', shape=[1], dtype='int64')
	loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')([labels, output, input_length, label_length])
	model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
	model.summary()
	ada_d = Adadelta(lr=0.01, rho=0.95, epsilon=1e-06)
	model=multi_gpu_model(model,gpus=2)
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
	in_len[0] = 500;
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
def train(wavpath = 'E:\\Data\\data_thchs30\\train', 
		textfile = 'E:\\Data\\thchs30\\train.syllable.txt', 
		bath_size = 4, 
		steps_per_epoch = 100, 
		epochs = 1):
	# 准备训练所需数据
	yielddatas = data_generate(wavpath, textfile, bath_size)
	# 导入模型结构，训练模型，保存模型参数
	model, model_data = creatModel()
	if os.path.exists('speech_model\\model_gru.mdl'):
		model.load_weights('speech_model\\model_gru.mdl')
	model.fit_generator(yielddatas, steps_per_epoch=steps_per_epoch, epochs=1)
	model.save_weights('speech_model\\model_gru.mdl')


# -----------------------------------------------------------------------------------------------------
'''
&usage:		模型的测试，看识别结果是否正确
'''
# -----------------------------------------------------------------------------------------------------
# 测试模型
def test(wavpath = 'E:\\Data\\data_thchs30\\train', 
		textfile = 'E:\\Data\\thchs30\\train.syllable.txt', 
		bath_size = 1):
	# 准备测试数据，以及生成字典
	word2num, num2word = gendict(textfile)
	yielddatas = data_generate(wavpath, textfile, bath_size)
	# 载入训练好的模型，并进行识别
	model, model_data = creatModel()
	model.load_weights('speech_model\\model_gru.mdl')
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
	run_type = sys.argv[1]
	if run_type == 'test':
		test()
	elif run_type == 'train':
		train()