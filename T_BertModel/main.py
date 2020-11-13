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
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from T_BertModel.model import MyTFBert
from T_BertModel.utils import load_data

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_text_list, train_label, dev_text_list, dev_label, test_text_list, test_label = load_data(tokenizer)

model = MyTFBert()

optimizer = tf.keras.optimizers.Adam(lr=1e-6)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
model.fit(
    x=test_text_list,
    y=test_label,
    batch_size=32,
    epochs=5,
    validation_data=(dev_text_list, dev_label)
)
# model.fit(
#     x=train_text_list,
#     y=train_label,
#     batch_size=32,
#     epochs=1,
#     validation_data=(dev_text_list, dev_label)
# )

