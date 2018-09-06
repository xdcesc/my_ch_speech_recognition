# -----------------------------------------------------------------------------------------------------
'''
&usage:     CBHG的中文语音识别语言模型
@author:    hongwen sun
'''
# -----------------------------------------------------------------------------------------------------
from __future__ import print_function
from hyperparams import Hyperparams as hp
from pypinyin import pinyin, lazy_pinyin, Style# pip install pypinyin
import codecs
import os
import regex# pip install regex
import pickle
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import re
from model_layers import *


# -----------------------------------------------------------------------------------------------------
'''
&usage:     生成训练所需的数据集，保存到指定文件data/zh.tsv中，主要函数为build_corpus()
'''
# -----------------------------------------------------------------------------------------------------
def align(sent):
    # 求出汉字拼音，并且放到list中去
    # 在漢字中添加
    pnyn = pinyin(sent,style=Style.TONE3)
    hanzis = sent
    pnyns = ''
    for i in pnyn:
        i = "".join(i)
        pnyns = pnyns + i
    # 汉字
    hanzis = "".join(hanzis)
    return pnyns, hanzis

def clean(text):
    if regex.search("[A-Za-z0-9]", text) is not None: # For simplicity, roman alphanumeric characters are removed.
        return ""
    text = regex.sub(u"[^ \p{Han}。，！？]", "", text)
    return text
    
#读取fin中的中文数据，转化为拼音，按照一定格式将结果保存到fout中去
def build_corpus():
    # 保存训练数据的文件
    with codecs.open("data/zh.tsv", 'w', 'utf-8') as fout:

        with codecs.open("data/lable.txt", 'r', 'utf-8') as fin:
            i = 1
            while 1:
                line = fin.readline()
                if not line: break
                
                try:
                    # 对原文本处理，idx为句子的id，sent为句子的文本，保留汉字并求出句子拼音
                    line = line.strip('\n')
                    idx = line.split(' ', 1)[0]
                    sent = line.split(' ', 1)[1]
                    # 去除掉句子中的其他符号，保留汉字
                    sent = clean(sent)
                    if len(sent) > 0:
                        # 求出拼音,返回两个字符串
                        pnyns, hanzis = align(sent)
                        # 将结果保存到fout中
                        fout.write(u"{}\t{}\t{}\n".format(idx, pnyns, hanzis))
                except:
                    continue # it's okay as we have a pretty big corpus!
                # 每隔10000行就打印一下
                if i % 10000 == 0: print(i, )
                i += 1


# -----------------------------------------------------------------------------------------------------
'''
&usage:     生成训练所需的字典，包括拼音和index映射，汉字和index映射
'''
# -----------------------------------------------------------------------------------------------------
#   构造字典
#   因为纯文本数据是没法作为深度学习输入的，所以我们首先得把文本转化为一个个对应的向量
#   这里我使用字典下标作为字典中每一个字对应的id, 然后每一条文本就可以通过遍历字典转化成其对应的向量了。
#   所以字典主要是应用在将文本转化成其在字典中对应的id, 根据语料库构造，这里我使用的方法是根据语料库中的字频构造字典
#   (我使用的是基于语料库中的字构造字典，有的人可能会先分词，基于词构造。
#   不使用基于词是现在就算是最好的分词都会有一些误分词问题，而且基于字还可以在一定程度上缓解OOV的问题)。

def build_vocab():
    from collections import Counter
    from itertools import chain

    # pinyin，制作拼音到index的字典，用于训练过程中将拼音输入转化为index输入          0: empty, 1: unknown, 2: blank
    # hanzis，制作汉字映射为index的字典，将所有结果保存到data/vocab.qwerty.pkl中     0: empty, 1: unknown, 2: blank
    pnyn_sents = [(line.split('\t')[1]).split(' ') for line in codecs.open('data/zh.tsv', 'r', 'utf-8').read().splitlines()]
    # 统计词频，通过词频建立字典
    # pinyin，制作拼音映射为index的字典，将所有结果保存到data/vocab.pkl中
    pnyn2cnt = Counter(chain.from_iterable(pnyn_sents))
    pnyns = [pnyn for pnyn, cnt in pnyn2cnt.items() if cnt > 5] # remove long-tail characters
    pnyns = pnyns[0:-1]
    pnyns = ["E", "U", "_" ] + pnyns # 0: empty, 1: unknown, 2: blank
    pnyn2idx = {pnyn:idx for idx, pnyn in enumerate(pnyns)}
    idx2pnyn = {idx:pnyn for idx, pnyn in enumerate(pnyns)}
    
    # hanzis，制作汉字映射为index的字典，将所有结果保存到data/vocab.pkl中
    hanzi_sents = [(line.split('\t')[2]).split(' ') for line in codecs.open('data/zh.tsv', 'r', 'utf-8').read().splitlines()]
    
    hanzi2cnt = Counter(chain.from_iterable(hanzi_sents))
    hanzis = [hanzi for hanzi, cnt in hanzi2cnt.items() if cnt > 5] # remove long-tail characters
    hanzis = hanzis[0:-1]
    hanzis = ["E", "U", "_" ] + hanzis # 0: empty, 1: unknown, 2: blank
    hanzi2idx = {hanzi:idx for idx, hanzi in enumerate(hanzis)}
    idx2hanzi = {idx:hanzi for idx, hanzi in enumerate(hanzis)}

    pickle.dump((pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi), open('data/vocab.pkl', 'wb'), 0)



# -----------------------------------------------------------------------------------------------------
'''
&usage:     数据处理，将拼音汉字和index互相转化，主要用于转化为训练数据
'''
# -----------------------------------------------------------------------------------------------------
def load_vocab():
    import pickle
    return pickle.load(open('data/vocab.pkl', 'rb'))


def load_train_data():
    '''Loads vectorized input training data'''
    pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab()

    print("pnyn vocabulary size is", len(pnyn2idx))
    print("hanzi vocabulary size is", len(hanzi2idx))

    xs, ys = [], []
    with codecs.open('pinyin_sentence', 'w', 'utf-8') as fout:
        for line in codecs.open('data/zh.tsv', 'r', 'utf-8'):
            try:
                #这里生成的数据pnyn_sent, hanzi_sent是string格式的数据
                _, pnyn_sent, hanzi_sent = line.strip().split("\t")
            except ValueError:
                continue
            # 将标点符号断句，分开作为训练数据
            pnyn_sents = re.sub(u"(?<=([。，！？]))", r"|", pnyn_sent).split("|")
            hanzi_sents = re.sub(u"(?<=([。，！？]))", r"|", hanzi_sent).split("|")
            fout.write(pnyn_sent + "===" + "|".join(pnyn_sents) + "\n")

            for pnyn_sent, hanzi_sent in zip(pnyn_sents, hanzi_sents):
                #assert len(pnyn_sent)==len(hanzi_sent)
                pnyn_sent = pnyn_sent.split(' ')
                hanzi_sent = hanzi_sent.split(' ')
                if hp.minlen < len(pnyn_sent) <= hp.maxlen:
                    # 通过字典将拼音映射为数字，如果字典中没有拼音的值，则返回1,1就是定义的Unkown，out of vocabulary
                    x = [pnyn2idx.get(pnyn, 1) for pnyn in pnyn_sent] # 1: OOV
                    # 对于汉字字典，0: empty, 1: unknown, 2: blank
                    y = [hanzi2idx.get(hanzi, 1) for hanzi in hanzi_sent] # 1: OOV
                    # xs,ys返回值是一个list格式的输入和输出数据
                    xs.append(np.array(x, np.int32).tostring())
                    ys.append(np.array(y, np.int32).tostring())
    return xs, ys


# 训练数据生成，格式处理
def get_batch():
    '''Makes batch queues from the training data.
    Returns:
      A Tuple of x (Tensor), y (Tensor).
      x and y have the shape [batch_size, maxlen].
    '''
    import tensorflow as tf

    # Load data
    X, Y = load_train_data()

    # Create Queues，相当于一个数据生成器，使训练可以边读数据边训练
    x, y = tf.train.slice_input_producer([tf.convert_to_tensor(X),
                                          tf.convert_to_tensor(Y)])

    x = tf.decode_raw(x, tf.int32)
    y = tf.decode_raw(y, tf.int32)

    x, y = tf.train.batch([x, y],
                          shapes=[(None,), (None,)],
                          num_threads=8,
                          batch_size=hp.batch_size,
                          capacity=hp.batch_size * 64,
                          allow_smaller_final_batch=False,
                          dynamic_pad=True)
    num_batch = len(X) // hp.batch_size

    return x, y, num_batch  # (N, None) int32, (N, None) int32, ()


def load_test_string(pnyn2idx, test_string):
    '''Embeds and vectorize words in user input string'''
    pnyn_sent = test_string
    pnyn_sent = pnyn_sent.split(' ')
    xs = []
    x = [pnyn2idx.get(pnyn, 1) for pnyn in pnyn_sent]
    x += [0] * (hp.maxlen - len(x))
    xs.append(x)
    X = np.array(xs, np.int32)
    return X


# -----------------------------------------------------------------------------------------------------
'''
&usage:     构建语言网络结构模型
'''
# -----------------------------------------------------------------------------------------------------
class Graph():
    '''Builds a model graph'''

    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                # 使用TensorFlow的生成器来传递输入数据
                self.x, self.y, self.num_batch = get_batch()
            else:  # Evaluation
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen,))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen,))

            # Load vocabulary
            #字典，上一步生成的拼音和汉字映射为数字的字典
            pnyn2idx, _, hanzi2idx, _ = load_vocab()

            # Character Embedding for x
            enc = embed(self.x, len(pnyn2idx), hp.embed_size, scope="emb_x")

            # Encoder pre-net
            prenet_out = prenet(enc,
                                num_units=[hp.embed_size, hp.embed_size // 2],
                                is_training=is_training)  # (N, T, E/2)

            # Encoder CBHG
            ## Conv1D bank
            enc = conv1d_banks(prenet_out,
                               K=hp.encoder_num_banks,
                               num_units=hp.embed_size // 2,
                               is_training=is_training)  # (N, T, K * E / 2)

            ## Max pooling
            enc = tf.layers.max_pooling1d(enc, 2, 1, padding="same")  # (N, T, K * E / 2)

            ## Conv1D projections
            enc = conv1d(enc, hp.embed_size // 2, 5, scope="conv1d_1")  # (N, T, E/2)
            enc = normalize(enc, type=hp.norm_type, is_training=is_training,
                            activation_fn=tf.nn.relu, scope="norm1")
            enc = conv1d(enc, hp.embed_size // 2, 5, scope="conv1d_2")  # (N, T, E/2)
            enc = normalize(enc, type=hp.norm_type, is_training=is_training,
                            activation_fn=None, scope="norm2")
            enc += prenet_out  # (N, T, E/2) # residual connections

            ## Highway Nets
            for i in range(hp.num_highwaynet_blocks):
                enc = highwaynet(enc, num_units=hp.embed_size // 2,
                                 scope='highwaynet_{}'.format(i))  # (N, T, E/2)

            ## Bidirectional GRU
            enc = gru(enc, hp.embed_size // 2, True, scope="gru1")  # (N, T, E)

            ## Readout
            self.outputs = tf.layers.dense(enc, len(hanzi2idx), use_bias=False)
            self.preds = tf.to_int32(tf.arg_max(self.outputs, dimension=-1))

            if is_training:
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.outputs)
                self.istarget = tf.to_float(tf.not_equal(self.y, tf.zeros_like(self.y)))  # masking
                self.hits = tf.to_float(tf.equal(self.preds, self.y)) * self.istarget
                self.acc = tf.reduce_sum(self.hits) / tf.reduce_sum(self.istarget)
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / tf.reduce_sum(self.istarget)

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('mean_loss', self.mean_loss)
                tf.summary.scalar('acc', self.acc)
                self.merged = tf.summary.merge_all()

# -----------------------------------------------------------------------------------------------------
'''
&usage:     训练语言模型
'''
# -----------------------------------------------------------------------------------------------------
def train():
    g = Graph(); print("Training Graph loaded")

    with g.graph.as_default():
        # Training
        sv = tf.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=0)
        with sv.managed_session() as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            for epoch in range(1, hp.num_epochs + 1):
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)

                # Write checkpoint files at every epoch
                gs = sess.run(g.global_step)
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))


# -----------------------------------------------------------------------------------------------------
'''
&usage:     测试模型效果
'''
# -----------------------------------------------------------------------------------------------------

def test():  
    g = Graph(is_training=False)
    
    # Load vocab
    pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab()
    
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")

            # Get model
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
            while True:
                line = input("请输入测试拼音：")

                if len(line) > hp.maxlen:
                    print('最长拼音不能超过50')
                    continue
                x = load_test_string(pnyn2idx, line)
                #print(x)
                preds = sess.run(g.preds, {g.x: x})
                #got = "".join(idx2hanzi[str(idx)] for idx in preds[0])[:np.count_nonzero(x[0])].replace("_", "")
                got = "".join(idx2hanzi[idx] for idx in preds[0])[:np.count_nonzero(x[0])].replace("_", "")
                print(got)


# -----------------------------------------------------------------------------------------------------
'''
&usage:     生成训练所需的数据集，保存到指定文件中，主要函数为build_corpus()
'''
# -----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # run type: pre 准备训练所需要数据；train 开始训练；test 测试模型是否好用
    run_type = input("input run type, you can choose from [pre train test]:\n")
    if run_type == 'pre':
        build_corpus(); print("Done")
        build_vocab(); print("Done" )
    if run_type == 'train':
        train(); print("Done")
    elif run_type == 'test':
        test(); print('Done')
    else:
        pass