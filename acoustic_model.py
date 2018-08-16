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
		self.AUDIO_LENGTH = 200
		self.AUDIO_FEATURE_LENGTH = 26
		self.OUTPUT_SIZE = 1500

	def rnn_model(self):
		input_data = Input(name='the_input', shape=(self.AUDIO_FEATURE_LENGTH))
		layer_h1 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(input_data)
		layer_h2_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal')(layer_h1) # GRU
		layer_h2_2 = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(layer_h1) # GRU
		layer_h2 = add([layer_h2_1, layer_h2_2])
		layer_h3 = Dense(self.OUTPUT_SIZE，use_biae=True, kernel_initializer='he_normal')(layer_h2)
		output = Activation('softmax', name='Activation0')(layer_h3)

		labels = Input(name='the_labels', shape[self.label_max_string_length], dtype='float32')
		input_length = Input(name='input_length', shape=[1], dtype='int64')
		label_length = Input(name='label_length', shape=[1], dtype='int64')
		loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([output, labels, input_length, label_length])
		model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
		model.summary()
		ada_d = Adadelta(lr=0.01, rho=0.95, epsilon=1e-06)
		model.compile(loss={'ctc': lambda y_true, output: output}, optimizers=ada_d)
		test_func = K.function([input_data], [output])
		return model


	def ctc_lambda(self, args):
		y_pred, labels, input_length, label_length = args
		y_pred = y_pred[:, :, :]
		return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


	def train(self):
		model.fit()