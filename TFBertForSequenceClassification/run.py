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

from transformers import BertTokenizer

from TFBertForSequenceClassification.model import MyModel
from TFBertForSequenceClassification.utils import load_data

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_text_list, train_label, dev_text_list, dev_label, test_text_list, test_label = load_data(tokenizer)

model = MyModel(10)
optimizer = tf.keras.optimizers.Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
model.fit(
    x=test_text_list,
    y=test_label,
    batch_size=32,
    epochs=5,
    validation_data=(dev_text_list, dev_label)
)
