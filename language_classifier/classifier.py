# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree


def naive_bayes(train_x, train_y, test_x):
    model = BernoulliNB(fit_prior=False)
    model.fit(train_x, train_y)
    pred_train = model.predict(train_x)
    pred_test = model.predict(test_x)
    return pred_train, pred_test


def decision_tree(train_x, test_x, train_y, max_d=100, max_f='sqrt'):
    model = tree.DecisionTreeClassifier(max_depth=max_d, max_features=max_f)
    model.fit(train_x, train_y)
    pred_train = model.predict(train_x)
    pred_test = model.predict(test_x)
    return pred_train, pred_test


def performance(pre_lable, true_lable):
    """
    三大指标评价模型表现
    :param pred: 预测标签值
    :param true: 实际标签值
    :return: 总准确率， 分类别准确率percision，召回率recall
    """
    correct_rate_all = np.sum(pre_lable == true_lable) / len(pre_lable)
    percision = {}
    recall = {}
    for lable in np.unique(true_lable):
        prei = pre_lable[true_lable == lable]
        truei = true_lable[true_lable == lable]
        percision[lable] = np.sum(prei == truei) / len(pre_lable[pre_lable == lable])
        recall[lable] = np.sum(prei == truei) / len(truei)
    return {'Overall_accuracy': correct_rate_all, 'Percision': percision,
            'Recall': recall}