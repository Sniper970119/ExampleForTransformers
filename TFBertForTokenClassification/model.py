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
from transformers import TFBertForTokenClassification, TFBertModel

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bert = TFBertForTokenClassification.from_pretrained('bert-base-chinese', return_dict=True, num_labels=7)

        # TFBertForTokenClassification by yourself
        # self.bert = TFBertModel.from_pretrained('bert-base-chinese')
        # self.dropout = tf.keras.layers.Dropout(0.1)
        # self.classifier = tf.keras.layers.Dense(7, name="classifier")
        pass

    def call(self, inputs):
        inputs_text, attention_mask, labels = inputs
        out = self.bert(inputs_text, attention_mask=attention_mask, training=True)

        # TFBertForTokenClassification by yourself
        # seq_output = out[0]
        # seq_output = self.dropout(seq_output, training=True)
        # logits = self.classifier(seq_output)
        # # print(out.loss)
        # return logits
        return out.logits


if __name__ == '__main__':
    from transformers import BertTokenizer
    import numpy as np

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    inputs = tokenizer.encode_plus("你好", max_length=100, padding='max_length')
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
    data = [ipt, attn, ids]
    m = MyModel()
    res = m.call(data)
    res = m.call(data)
    print(res.shape)

    pass
