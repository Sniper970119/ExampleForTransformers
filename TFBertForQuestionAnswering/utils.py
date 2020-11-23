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
import json
import pickle
import tensorflow as tf
from tqdm import tqdm

def load_file(file_name):
    res_questions = []
    res_answers = []
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = data['data']
        for each in data:
            questions = each['paragraphs']
            for question in questions:

                pass
    print()

if __name__ == '__main__':
    load_file('../data/SQuAD/dev.json')