# -*- coding:utf8 -*-

import os
import shutil
import pandas as pd


def get_file(file_type, dictionary, destination):
    """遍历原始文件夹，把所需的文件类型抽出来，汇总在同一文件夹下以便后续处理"""
    # dictionary:'C:/Users/yangyujie8/PycharmProjects/deep_learning/jianzhi/data/language_classifier_data/...'
    # destination: 'C:/Users/yangyujie8/PycharmProjects/deep_learning/jianzhi/data/language_classifier_data/all_files'
    for root, __, files in os.walk(dictionary):
        # print(str(root).split(dictionary.split('/')[-1])[-1])
        count = 0
        for f in files:
            if f.split('.')[-1] == file_type:
                # print(f)
                des_path = destination + '/file' + str(count) + '.' + file_type
                count += 1
                shutil.move(root + '/' + f, des_path)
    return


def file_to_df(file_path):
    """直接把text内容解析出来存入df"""
    text_all = []
    err_count = 0
    for root, __, files in os.walk(file_path):
        for f in files:
            try:
                fo = open(root + '/' + f, 'r', encoding='gbk')
                text = fo.read()
                text = text.strip().replace('\n', '').replace('\t', '')
                text_all.append([text, f.split('.')[-1]])
            except Exception as e:
                err_count += 1
                if err_count%10 == 0:
                    print(err_count)
                continue
    text_df = pd.DataFrame(text_all, columns=['text', 'type'])
    return text_df



