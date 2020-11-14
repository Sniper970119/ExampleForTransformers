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
import numpy as np

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from transformers import BertTokenizer, TFBertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

model = TFBertForMaskedLM.from_pretrained('bert-base-cased', return_dict=True)

inputs = tokenizer("The capital of France is [MASK].", return_tensors="tf")

outputs = model(inputs)
logits = outputs.logits

output = np.argmax(logits[0][6])
o1 = tokenizer.decode(int(output))

inputs = tokenizer("The capital of [MASK] is BeiJing.", return_tensors="tf")

outputs = model(inputs)
logits = outputs.logits

output = np.argmax(logits[0][4])
o2 = tokenizer.decode(int(output))

print()
