from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape # , Flatten,LSTM,Convolution1D,MaxPooling1D,Merge
from keras.layers import Conv1D,LSTM,MaxPooling1D, Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D #, Merge,Conv1D
from keras.layers.merge import add, concatenate
from keras import backend as K
from keras.optimizers import SGD, Adadelta
from keras.layers.recurrent import GRU
import numpy as np
#from keras.utils import multi_gpu_model


class speech_rnn():
	"""docstring for speech_rnn"""
	def __init__(self):
		super(speech_rnn, self).__init__()
		#self.input_dict = input_dict
		#self.label_dict = label_dict
		self.AUDIO_LENGTH = 500
		self.AUDIO_FEATURE_LENGTH = 26
		self.OUTPUT_SIZE = 1200
		self.model = self.rnn_model()

	def rnn_model(self):
		input_data = Input(name='the_input', shape=(500, self.AUDIO_FEATURE_LENGTH,))
		layer_h1 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(input_data)
		layer_h2_1 = GRU(512, return_sequences=True, kernel_initializer='he_normal')(layer_h1) # GRU
		layer_h2_2 = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(layer_h1) # GRU
		layer_h2 = add([layer_h2_1, layer_h2_2])
		layer_h3 = Dense(1200, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h2)
		output = Activation('softmax', name='Activation0')(layer_h3)
		#ctc
		labels = Input(name='the_labels', shape=[50], dtype='float32')
		input_length = Input(name='input_length', shape=[1], dtype='int64')
		label_length = Input(name='label_length', shape=[1], dtype='int64')
		loss_out = Lambda(self.ctc_lambda, output_shape=(1,), name='ctc')([output, labels, input_length, label_length])
		model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
		model.summary()
		ada_d = Adadelta(lr=0.01, rho=0.95, epsilon=1e-06)
		#model=multi_gpu_model(model,gpus=2)
		model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=ada_d)
		#test_func = K.function([input_data], [output])
		return model


	def ctc_lambda(self, args):
		y_pred, labels, input_length, label_length = args
		y_pred = y_pred[:, :, :]
		return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

	def get_batch(self, feats, labels, train=False, max_pred_len=50, input_length=500):
	    X = np.expand_dims(feats, axis=3)
	    X = feats # for model2
	#     labels = np.ones((y.shape[0], max_pred_len)) *  -1 # 3 # , dtype=np.uint8
	    labels = labels
	    
	    input_length = np.ones([feats.shape[0], 1]) * ( input_length - 2 )
	#     label_length = np.ones([y.shape[0], 1])
	    label_length = np.sum(labels > 0, axis=1)
	    label_length = np.expand_dims(label_length,1)

	    inputs = {'the_input': X,
	              'the_labels': labels,
	              'input_length': input_length,
	              'label_length': label_length,
	              }
	    outputs = {'ctc': np.zeros([feats.shape[0]])}  # dummy data for dummy loss function
	    return (inputs, outputs)


	def train(self, inputs, outputs):
		self.model.fit(inputs, outputs, epochs=50, batch_size=32)
