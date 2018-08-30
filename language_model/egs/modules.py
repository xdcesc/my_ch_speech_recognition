
from __future__ import print_function
import tensorflow as tf
from hyperparams import Hyperparams as hp

# 包含了CBHG中含有的网络，相关：https://arxiv.org/pdf/1703.10135.pdf

''' ------------------------------------------------------------------------------
embed layer
光有对应的id，没法很好的表征文本信息，这里就涉及到构造词向量，关于词向量不在说明，
网上有很多资料，模型中使用词嵌入层，通过训练不断的学习到语料库中的每个字的词向量，代码如下
'''
def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimesionality
        should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)


    return tf.nn.embedding_lookup(lookup_table, inputs)



def normalize(inputs,
              type="bn",
              decay=.99,
              epsilon=1e-8,
              is_training=True,
              activation_fn=None,
              reuse=None,
              scope="normalize"):
    '''Applies {batch|layer} normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but
        the last dimension. Or if type is `ln`, the normalization is over
        the last dimension. Note that this is different from the native
        `tf.contrib.layers.batch_norm`. For this I recommend you change
        a line in ``tensorflow/contrib/layers/python/layers/layer.py`
        as follows.
        Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
        After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)
      type: A string. Either "bn" or "ln".
      decay: Decay for the moving average. Reasonable values for `decay` are close
        to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
      is_training: Whether or not the layer is in training mode. W
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    if type == "bn":
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims

        # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
        # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
        if inputs_rank in [2, 3, 4]:
            if inputs_rank == 2:
                inputs = tf.expand_dims(inputs, axis=1)
                inputs = tf.expand_dims(inputs, axis=2)
            elif inputs_rank == 3:
                inputs = tf.expand_dims(inputs, axis=1)

            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                   decay=decay,
                                                   center=True,
                                                   scale=True,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   scope=scope,
                                                   zero_debias_moving_mean=True,
                                                   fused=True,
                                                   reuse=reuse)
            # restore original shape
            if inputs_rank == 2:
                outputs = tf.squeeze(outputs, axis=[1, 2])
            elif inputs_rank == 3:
                outputs = tf.squeeze(outputs, axis=1)
        else:  # fallback to naive batch norm
            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                   decay=decay,
                                                   center=True,
                                                   scale=True,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   scope=scope,
                                                   reuse=reuse,
                                                   fused=False)
    elif type in ("ln", "ins"):
        reduction_axis = -1 if type == "ln" else 1
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [reduction_axis], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta
    else:
        outputs = inputs

    if activation_fn:
        outputs = activation_fn(outputs)


    return outputs

'''
CBHG模块由1-D convolution bank ，highway network ，bidirectional GRU 组成。
它的功能是从输入中提取有价值的特征，有利于提高模型的泛化能力。
输入序列首先会经过一个卷积层，注意这个卷积层，它有K个大小不同的1维的filter，其中filter的大小为1,2,3…K。
这些大小不同的卷积核提取了长度不同的上下文信息。其实就是n-gram语言模型的思想，K的不同对应了不同的gram, 
例如unigrams, bigrams, up to K-grams，然后，将经过不同大小的k个卷积核的输出堆积在一起
（注意：在做卷积时，运用了padding，因此这k个卷积核输出的大小均是相同的），
也就是把不同的gram提取到的上下文信息组合在一起，下一层为最大池化层，stride为1，width为2。
'''
def conv1d(inputs,
       filters=None,
       size=1,
       rate=1,
       padding="SAME",
       use_bias=False,
       activation_fn=None,
       scope="conv1d",
       reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list[-1]

        params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding, "activation": activation_fn,
                  "use_bias": use_bias, "reuse": reuse}

        outputs = tf.layers.conv1d(**params)
    return outputs


def conv1d_banks(inputs, num_units=None, K=16, is_training=True, scope="conv1d_banks", reuse=None):
    '''Applies a series of conv1d separately.

    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      K: An int. The size of conv1d banks. That is,
        The `inputs` are convolved with K filters: 1, 2, ..., K.
      is_training: A boolean. This is passed to an argument of `batch_normalize`.

    Returns:
      A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = hp.embed_size // 2
        outputs = conv1d(inputs, hp.embed_size // 2, 1)  # k=1
        for k in range(2, K + 1):  # k = 2...K
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, num_units, k)
                outputs = tf.concat((outputs, output), -1)
        outputs = normalize(outputs, type=hp.norm_type, is_training=is_training,
                            activation_fn=tf.nn.relu)
    return outputs  # (N, T, Hp.embed_size//2*K)


def gru(inputs, num_units=None, bidirection=False, seqlen=None, scope="gru", reuse=None):
    '''Applies a GRU.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: An int. The number of hidden units.
      bidirection: A boolean. If True, bidirectional results
        are concatenated.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units],
        otherwise [N, T, num_units].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]

        cell = tf.contrib.rnn.GRUCell(num_units)
        if bidirection:
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs,
                                                         sequence_length=seqlen,
                                                         dtype=tf.float32)
            return tf.concat(outputs, 2)
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs,
                                           sequence_length=seqlen,
                                           dtype=tf.float32)


    return outputs
'''
Encoder pre-net module
embeding layer之后是一个encoder pre-net模块，它有两个隐藏层，层与层之间的连接均是全连接；
第一层的隐藏单元数目与输入单元数目一致，
第二层的隐藏单元数目为第一层的一半；两个隐藏层采用的激活函数均为ReLu，并保持0.5的dropout来提高泛化能力 
'''
def prenet(inputs, num_units=None, is_training=True, scope="prenet", reuse=None):
    '''Prenet for Encoder and Decoder1.
    Args:
      inputs: A 2D or 3D tensor.
      num_units: A list of two integers. or None.
      is_training: A python boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3D tensor of shape [N, T, num_units/2].
    '''
    if num_units is None:
        num_units = [hp.embed_size, hp.embed_size // 2]

    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name="dense1")
        outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=is_training, name="dropout1")
        outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name="dense2")
        outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=is_training, name="dropout2")
    return outputs  # (N, ..., num_units[1])

'''
下一层输入到highway layers，highway nets的每一层结构为：把输入同时放入到两个一层的全连接网络中，
这两个网络的激活函数分别采用了ReLu和sigmoid函数，假定输入为input，ReLu的输出为output1，sigmoid的输出为output2，
那么highway layer的输出为output=output1∗output2+input∗（1−output2)。论文中使用了4层highway layer。
代码如下：
'''
def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387
    Args:
      inputs: A 3D tensor of shape [N, T, W].
      num_units: An int or `None`. Specifies the number of units in the highway layer
             or uses the input size if `None`.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A 3D tensor of shape [N, T, W].
    '''
    if not num_units:
        num_units = inputs.get_shape()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0), name="dense2")
        C = 1. - T
        outputs = H * T + inputs * C


    return outputs