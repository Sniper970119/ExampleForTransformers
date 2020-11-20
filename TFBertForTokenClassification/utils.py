# -*- coding:utf-8 -*-

"""
      ┏┛ ┻━━━━━┛ ┻┓
      ┃　　　　　　 ┃
      ┃　　　━　　　┃
      ┃　┳┛　  ┗┳　┃
      ┃　　　　　　 ┃
      ┃　　　┻　　　┃
      ┃　　　　　　 ┃
      ┗━┓　　　┏━━━┛
        ┃　　　┃   神兽保佑
        ┃　　　┃   代码无BUG！
        ┃　　　┗━━━━━━━━━┓
        ┃CREATE BY SNIPER┣┓
        ┃　　　　         ┏┛
        ┗━┓ ┓ ┏━━━┳ ┓ ┏━┛
          ┃ ┫ ┫   ┃ ┫ ┫
          ┗━┻━┛   ┗━┻━┛

"""

import os
import numpy as np
import pickle
from tqdm import tqdm
import tensorflow as tf
from transformers import BertTokenizer

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

labels_map = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'B-PER': 3, 'I-PER': 4, 'B-LOC': 5, 'I-LOC': 6}


def load_file(file_name):
    with open('../data/NER/' + file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    res = []
    labels = []
    temp_texts = []
    temp_labels = []
    for each in tqdm(lines):
        if each == '\n':
            res.append(''.join(temp_texts))
            # 处理开始结束标记的label
            temp_labels.append(0)
            temp_labels.insert(0, 0)
            labels.append(temp_labels)
            temp_texts = []
            temp_labels = []
            continue
        text, label = each.split(' ')
        label = labels_map[label.strip()]
        temp_texts.append(text)
        temp_labels.append(label)
    return res, labels


def get_token(data_list, tokenizer, labels):
    temp_ids = []
    temp_mask = []
    # temp_token = []
    for each in tqdm(data_list):
        temp = tokenizer.encode_plus(each, max_length=100, truncation='only_first', padding='max_length')
        temp_ids.append(temp['input_ids'])
        temp_mask.append(temp['attention_mask'])
        # temp_token.append(temp['token_type_ids'])

    temp_ids = np.asarray(temp_ids, dtype=np.int64)
    temp_mask = np.asarray(temp_mask, dtype=np.int32)
    # temp_token = np.asarray(temp_token, dtype=np.int32)
    text_list = [temp_ids, temp_mask, labels]
    return text_list


def load_data():
    if os.path.exists('handled.data'):
        with open('handled.data', 'rb') as f:
            data = pickle.load(f)
        return data['train_text'], data['train_label'], data['dev_text'], data['dev_label'], data['test_text'], data[
            'test_label']
    else:
        train_text, train_labels = load_file('train.txt')
        dev_text, dev_labels = load_file('dev.txt')
        test_text, test_labels = load_file('test.txt')
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        # train_text_list = tokenizer(train_text, max_length=100, padding='max_length')
        # dev_text_list = tokenizer(dev_text, max_length=100, padding='max_length')
        # test_text_list = tokenizer(test_text, max_length=100, padding='max_length')

        train_labels_padded = tf.keras.preprocessing.sequence.pad_sequences(train_labels, maxlen=100, padding='post')
        dev_labels_padded = tf.keras.preprocessing.sequence.pad_sequences(dev_labels, maxlen=100, padding='post')
        test_labels_padded = tf.keras.preprocessing.sequence.pad_sequences(test_labels, maxlen=100, padding='post')

        train_text_res = get_token(train_text, tokenizer, train_labels_padded)
        dev_text_res = get_token(dev_text, tokenizer, dev_labels_padded)
        test_text_res = get_token(test_text, tokenizer, test_labels_padded)

        train_labels_ont_hot = tf.keras.utils.to_categorical(train_labels_padded, num_classes=7)
        dev_labels_ont_hot = tf.keras.utils.to_categorical(dev_labels_padded, num_classes=7)
        test_labels_ont_hot = tf.keras.utils.to_categorical(test_labels_padded, num_classes=7)

        data = {}
        data['train_text'] = train_text_res
        data['train_label'] = train_labels_ont_hot
        data['dev_text'] = dev_text_res
        data['dev_label'] = dev_labels_ont_hot
        data['test_text'] = test_text_res
        data['test_label'] = test_labels_ont_hot

        with open('handled.data', 'wb') as f:
            pickle.dump(data, f)

        return train_text_res, train_labels_padded, dev_text_res, dev_labels_padded, test_text_res, test_labels_padded


if __name__ == '__main__':
    load_data()
