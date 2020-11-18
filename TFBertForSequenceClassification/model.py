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
from transformers import TFBertForSequenceClassification


class MyModel(tf.keras.Model):
    def __init__(self, num_labels):
        super(MyModel, self).__init__()
        self.bert = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', return_dict=True,
                                                                    num_labels=num_labels)

    def call(self, inputs):
        idx, attn, ids = inputs
        out = self.bert(idx, attention_mask=attn, token_type_ids=ids, training=True)
        return out.logits


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

    m = MyModel(10)
    res = m.call([ipt, attn, ids])
    print(res)
