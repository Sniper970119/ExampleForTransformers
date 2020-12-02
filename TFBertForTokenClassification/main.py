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
#
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from TFBertForTokenClassification.model import MyModel
from TFBertForTokenClassification.utils import load_data

if __name__ == '__main__':
    train_text_res, train_labels_padded, dev_text_res, dev_labels_padded, test_text_res, test_labels_padded = load_data()
    model = MyModel()


    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=1000):
            super(CustomSchedule, self).__init__()

            self.d_model = tf.cast(d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


    learning_rate = CustomSchedule(768)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)

    # optimizer = tf.keras.optimizers.Adam(lr=1e-7)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    model.fit(
        x=train_text_res,
        y=train_labels_padded,
        batch_size=16,
        epochs=3,
        validation_data=(test_text_res, test_labels_padded)
    )

    model.evaluate(x=test_text_res,y=test_labels_padded,batch_size=16)
