# -*- coding:utf8 -*-
import collections
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from gensim.models.word2vec import Word2Vec
from jianzhi.language_classifier.word_cut import add_word_to_sentence


def one_hot_x(cuted, ignore_num=1, word_set=None):
    """
    用one-hot的形式生成自变量，不考虑词频
    cuted: 需要转变成独热形式的字符的list的array
    ignore_num: 忽略低于该频数的词
    """
    cuted = cuted.values
    xs = len(cuted)
    out = pd.DataFrame()
    for i in range(xs):
        counti = pd.Series(dict(collections.Counter(cuted[i])))
        out = pd.concat([out, counti], axis=1)
        # print(out.shape)
    if word_set is None:
        outc = out.T.columns[np.sum(out, axis=1) > ignore_num]
        out = out.T[outc]
        word_set = list(out.columns)
        return out, word_set
    else:
        outc = pd.concat([out, pd.Series(np.zeros(len(word_set)), index=word_set, name='word_set__')], axis=1)
        outc = outc[~np.isnan(outc['word_set__'])].iloc[:, :-1].T
        print(outc.columns[:5])
        return outc


def pca(x, test_x, keep_var=0.8):
    """
    用主成分分析法对数据降维
    :param x: 原始的x
    :param keep_var: 保留的方差比例
    :return: 降维后的x
    """
    pca = PCA(n_components=keep_var)
    pca.fit(x)
    print(pca.explained_variance_ratio_)
    x_new = pca.transform(x)
    test_x_new = pca.transform(test_x)
    return x_new, test_x_new


def word2vec(train, test, min_count=1, sg=1, vec_size=100):
    """
    需要用train数据来训练模型，然后用训练好的模型生成test数据，所以需要分开给出train和test
    """
    # 训练word2vec基础词汇表
    model = Word2Vec(min_count=min_count, size=vec_size, sg=sg)
    print(1)
    model.build_vocab(train)
    print(2)
    model.train(train, total_examples=model.corpus_count, epochs=model.epochs)
    print(3)
    # 计算句子向量：最简单方式——词向量的简单平均
    line_vecs_train = add_word_to_sentence(model, train)
    line_vecs_test = add_word_to_sentence(model, test)
    print(4)
    return line_vecs_train, line_vecs_test
