from xpinyin import Pinyin # pip install xpinyin 
import regex
import codecs
from collections import Counter
from itertools import chain
import re
import tensorflow as tf

def align(sent):
    '''
    Args:
      sent: A string. A sentence.
    
    Returns:
      A tuple of pinyin and chinese sentence.
    '''
    pinyin = Pinyin()
    # 求出汉字拼音，并且放到list中去
    pnyns = pinyin.get_pinyin(sent, " ").split()
    # 在漢字中添加_
    hanzis = []
    for char, p in zip(sent.replace(" ", ""), pnyns):
        hanzis.extend([char] + ["_"] * (len(p) - 1))
        
    pnyns = "".join(pnyns)
    hanzis = "".join(hanzis)
    
    assert len(pnyns) == len(hanzis), "The hanzis and the pinyins must be the same in length."
    return pnyns, hanzis



def clean(text):
    if regex.search("[A-Za-z0-9]", text) is not None: # For simplicity, roman alphanumeric characters are removed.
        return ""
    text = regex.sub(u"[^ \p{Han}。，！？]", "", text)
    return text

def load_data():
    for line in codecs.open('data/zh.tsv', 'r', 'utf-8'):
        try:
            _, pnyn_sent, hanzi_sent = line.strip().split("\t")
        except ValueError:
            continue
        pnyn_sents = re.sub(u"(?<=([。，！？]))", r"|", pnyn_sent).split("|")
        hanzi_sents = re.sub(u"(?<=([。，！？]))", r"|", hanzi_sent).split("|")
        #print(pnyn_sent)

        for pnyn_sent, hanzi_sent in zip(pnyn_sents, hanzi_sents):
            #assert len(pnyn_sent)==len(hanzi_sent)
            # string
            print(pnyn_sent)
            print(hanzi_sent)


if __name__ == '__main__':
    pnyn2idx = {'hello':1, 'chifan':2}
    pnyn_sent = ['hello','chifan','chifan','hello']
    x = [pnyn2idx.get(pnyn, 1) for pnyn in pnyn_sent]
    print(x)