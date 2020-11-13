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

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from transformers import TFBertModel, TFBertMainLayer, TFPreTrainedModel, TFBertPreTrainedModel


class MyTFBert(tf.keras.Model):
    def __init__(self):
        super(MyTFBert, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-chinese', return_dict=True)
        # self.bert = TFBertMainLayer()
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
        self.dense2 = tf.keras.layers.Dense(768, activation='tanh')

    def call(self, inputs):
        idx, attn, ids = inputs
        hidden = self.bert(idx, attention_mask=attn, token_type_ids=ids, training=False)
        temp = hidden[0][:, 0]
        temp = self.dense2(temp)
        temp = hidden[1]
        temp = self.dropout(temp, training=True)
        out = self.dense(temp)
        return out


if __name__ == '__main__':
    from transformers import BertTokenizer
    import numpy as np

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    inputs = tokenizer.encode_plus("你好", max_length=50, padding='max_length')
    ipt = []
    ipt.append(inputs['input_ids'])
    ipt.append(inputs['input_ids'])
    ipt = np.asarray(ipt, dtype=np.int32)

    attn = []
    attn.append(inputs['attention_mask'])
    attn.append(inputs['attention_mask'])
    attn = np.asarray(attn, dtype=np.int32)

    ids = []
    ids.append(inputs['token_type_ids'])
    ids.append(inputs['token_type_ids'])
    ids = np.asarray(ids, dtype=np.int32)

    m = MyTFBert()
    res = m.call([ipt, attn, ids])
    res = m.call([ipt, attn, ids])
    print(res.shape)
