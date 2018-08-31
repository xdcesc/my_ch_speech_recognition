from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
from prepro import *
from data_load import load_vocab, load_test_data, load_test_string
from train import Graph
import codecs
import distance
import os


#For user input test                
def main():  
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

                                                                                                   
if __name__ == '__main__':
    main(); print("Done")