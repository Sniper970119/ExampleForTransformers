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
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
import pickle


def handle_data(data_list, tokenizer):
    temp_ids = []
    temp_mask = []
    temp_token = []
    for each in tqdm(data_list):
        temp = tokenizer.encode_plus(each, max_length=50, padding='max_length')
        temp_ids.append(temp['input_ids'])
        temp_mask.append(temp['attention_mask'])
        temp_token.append(temp['token_type_ids'])

    temp_ids = np.asarray(temp_ids, dtype=np.int32)
    temp_mask = np.asarray(temp_mask, dtype=np.int32)
    temp_token = np.asarray(temp_token, dtype=np.int32)
    text_list = [temp_ids, temp_mask, temp_token]
    return text_list


def load_data(tokenizer):
    if os.path.exists('handled.data'):
        with open('handled.data', 'rb') as f:
            data = pickle.load(f)
        return data['train_x'], data['train_y'], data['dev_x'], data['dev_y'], data['test_x'], data['test_y']
    else:
        train_text = []
        train_label = []
        dev_text = []
        dev_label = []
        test_text = []
        test_label = []
        with open('../data/classisfy/train.txt', 'r', encoding='utf-8') as f:
            train_list = f.read().split('\n')
            for line in train_list:
                if line == '':
                    continue
                temp = line.split('\t')
                train_text.append(temp[0])
                train_label.append(temp[1])
            del train_list
        with open('../data/classisfy/dev.txt', 'r', encoding='utf-8') as f:
            dev_list = f.read().split('\n')
            for line in dev_list:
                if line == '':
                    continue
                temp = line.split('\t')
                dev_text.append(temp[0])
                dev_label.append(temp[1])
            del dev_list
        with open('../data/classisfy/test.txt', 'r', encoding='utf-8') as f:
            test_list = f.read().split('\n')
            for line in test_list:
                if line == '':
                    continue
                temp = line.split('\t')
                test_text.append(temp[0])
                test_label.append(temp[1])
            del test_list
        del temp
        del line
        del f

        train_text_list = handle_data(train_text, tokenizer)
        dev_text_list = handle_data(dev_text, tokenizer)
        test_text_list = handle_data(test_text, tokenizer)

        train_label = tf.keras.utils.to_categorical(train_label, num_classes=10)
        dev_label = tf.keras.utils.to_categorical(dev_label, num_classes=10)
        test_label = tf.keras.utils.to_categorical(test_label, num_classes=10)

        data = {
            'train_x': train_text_list,
            'train_y': train_label,
            'dev_x': dev_text_list,
            'dev_y': dev_label,
            'test_x': test_text_list,
            'test_y': test_label
        }

        with open('handled.data', 'wb') as f:
            pickle.dump(data, f)

        return train_text_list, train_label, dev_text_list, dev_label, test_text_list, test_label


if __name__ == '__main__':
    from transformers import BertTokenizer,TFBertForSequenceClassification
    s = TFBertForSequenceClassification.from_pretrained()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    load_data(tokenizer)
