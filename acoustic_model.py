from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape, GRU # , Flatten,LSTM,Convolution1D,MaxPooling1D,Merge
from keras.layers import Conv1D,LSTM,MaxPooling1D, Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D #, Merge,Conv1D
from keras.layers.merge import add, concatenate
from keras import backend as K
from keras.optimizers import SGD, Adadelta

class speech_rnn():
	"""docstring for speech_rnn"""
	def __init__(self, input_dict, label_dict):
		super(speech_rnn, self).__init__()
		self.input_dict = input_dict
		self.label_dict = label_dict
		self.AUDIO_LENGTH = 
		self.AUDIO_FEATURE_LENGTH = 26

	def model(self):
		input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))
		layer_h1 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(input_data) # 卷积层
