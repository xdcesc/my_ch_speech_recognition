
'''
Preprocessing.
Note:
Nine key pinyin keyboard layout sample:

`      ABC   DEF
GHI    JKL   MNO
POQRS  TUV   WXYZ

'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import codecs
import pickle
import json

#   构造字典
#   因为纯文本数据是没法作为深度学习输入的，所以我们首先得把文本转化为一个个对应的向量
#   这里我使用字典下标作为字典中每一个字对应的id, 然后每一条文本就可以通过遍历字典转化成其对应的向量了。
#   所以字典主要是应用在将文本转化成其在字典中对应的id, 根据语料库构造，这里我使用的方法是根据语料库中的字频构造字典
#   (我使用的是基于语料库中的字构造字典，有的人可能会先分词，基于词构造。
#   不使用基于词是现在就算是最好的分词都会有一些误分词问题，而且基于字还可以在一定程度上缓解OOV的问题)。


def build_vocab():
    """Builds vocabulary from the corpus.
    Creates a pickle file and saves vocabulary files (dict) to it.
    """
    from collections import Counter
    from itertools import chain

    # pinyin，选择全键盘输入方式，输入的值为：abcdefghijklmnopqrstuvwxyz0123456789。，！？   #E:Empty U:UnKnown
    # 制作拼音到index的字典，用于训练过程中将拼音输入转化为index输入
    if hp.isqwerty:
        pnyns = "EUabcdefghijklmnopqrstuvwxyz0123456789。，！？" #E: Empty, U: Unknown
        pnyn2idx = {pnyn:idx for idx, pnyn in enumerate(pnyns)}
        idx2pnyn = {idx:pnyn for idx, pnyn in enumerate(pnyns)}
    else:
        pnyn2idx, idx2pnyn = dict(), dict()
        pnyns_list = ["E", "U", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz", 
                      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", u"。", u"，", u"！", u"？"] #E: Empty, U: Unknown
        for i, pnyns in enumerate(pnyns_list):
            for pnyn in pnyns:
                pnyn2idx[pnyn] = i
    
    # hanzis，制作汉字映射为index的字典，将所有结果保存到data/vocab.qwerty.pkl中
    hanzi_sents = [line.split('\t')[2] for line in codecs.open('data/zh.tsv', 'r', 'utf-8').read().splitlines()]
    hanzi2cnt = Counter(chain.from_iterable(hanzi_sents))
    hanzis = [hanzi for hanzi, cnt in hanzi2cnt.items() if cnt > 5] # remove long-tail characters
    
    hanzis.remove("_")
    hanzis = ["E", "U", "_" ] + hanzis # 0: empty, 1: unknown, 2: blank
    hanzi2idx = {hanzi:idx for idx, hanzi in enumerate(hanzis)}
    idx2hanzi = {idx:hanzi for idx, hanzi in enumerate(hanzis)}
    
    if hp.isqwerty:
        pickle.dump((pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi), open('data/vocab.qwerty.pkl', 'wb'), 0)
        #json.dump((pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi), open('data/vocab.qwerty.json', 'w'))
    else:
        pickle.dump((pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi), open('data/vocab.nine.pkl', 'wb'), 0)
        #json.dump((pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi), open('data/vocab.nine.json', 'w'))
if __name__ == '__main__':
    build_vocab(); print("Done" )