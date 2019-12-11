# -*- coding:utf8 -*-

import jieba.posseg as pseg
import numpy as np


# 词向量到句向量（简单平均）
def add_word_to_sentence(word_model, yl):
    line_vecs = np.zeros((len(yl), word_model.vector_size))
    for l in range(len(yl)):
        line = yl[l]
        # words = ''.join(line.strip().split(' '))
        word_vec = []
        for w in line:
            try:
                word_vec.append(word_model[w])
            except KeyError:
                # print('%s not in model vocabulary, so give it a zero vector.'%w)
                word_vec.append(np.zeros(word_model.vector_size,))
        word_vec = np.array(word_vec)
        line_vecs[l, :] = np.mean(word_vec, axis=0)
    return line_vecs


# 将语料整理成list，每句为一个子list，其中包括分词后的词语
def word_cut(data_array, out='l'):
    """按照word2vec训练要求的格式进行分词"""
    # cutout = codecs.open(out_path, 'w', 'utf-8')
    lines = []
    count = 0
    for s in data_array:
        count += 1
        print(count)
        seg = pseg.cut(s)  # 分词和词性标注
        if out == 'l':  # list输出
            linei = []
            for x in seg:
                w = x.word
                if w != '' and w != ' ' and len(w) < 25:
                    linei.append(w)
        elif out == 's':    # str+空格输出
            linei = ''
            for x in seg:
                if x.flag not in ['w', 'x']:
                    linei += x.word + ' '
        lines.append(linei)
        # cutout.write('\n')
    # cutout.close()
    return lines