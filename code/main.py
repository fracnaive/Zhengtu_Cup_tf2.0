# 生成json文件

import tensorflow as tf
import cv2
import json
import numpy as np
import os
import glob
from numba import jit
import config
from train.model import AES

content = {}
paths = glob.glob(os.path.join(config.test_path2, '*.bmp'))
for step, path in enumerate(paths):
    print(path)
    rec_model = AES
    # z_path = os.path.join(config.test_path, '00b5CacodX4rVA9jmo6aad56xvwMT3.bmp')
    src = cv2.imread(path)
    gray = src.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = tf.expand_dims(gray, axis=2)
    # print(gray.shape)
    # gray = tf.io.read_file(z_path)
    # gray = tf.image.decode_bmp(gray, channels=1)
    z = tf.cast(gray, dtype=tf.float32) / 255.  # (128, 128, 1)
    z = tf.expand_dims(z, axis=0)  # (1, 128, 128, 1)

    rec_model.load_weights('../model/rec_model_part2.h5')
    logits = rec_model(z)
    x_hat = tf.nn.sigmoid(logits)
    x_hat = tf.cast(x_hat, dtype=tf.float32)
    x_hat = tf.image.resize(x_hat, [128, 128])
    x_hat = tf.squeeze(x_hat, axis=3)
    x_hat = x_hat.numpy() * 255.
    x_hat = x_hat.astype(np.uint8)  # shape(1, 128, 128)
    x_hat = tf.squeeze(x_hat, axis=0)
    # print('x_hat.shape:', x_hat.shape)

    noise_image_src = cv2.imread(path)
    noise_image = noise_image_src.copy()
    noise_image = cv2.cvtColor(noise_image, cv2.COLOR_BGR2GRAY)
    # print(noise_image.shape)

    # gpu加速，处理两个for循环时间过长的问题
    @jit
    def substract(a1, b1):
        l = np.full(128*128, -1)
        r = np.full(128*128, -1)
        n = 0
        for i in range(128):
            for j in range(128):
                noise_lbp_point = b1[i][j]
                rec_lbp_point = a1[i][j]
                # 由于这些数组都是uint8类型的，小值减去大值不会得到负数，得到的是错误的值，所以需要提前判断大小
                if noise_lbp_point > rec_lbp_point:
                    error_point = noise_lbp_point - rec_lbp_point
                else:
                    error_point = rec_lbp_point - noise_lbp_point
                if error_point >= 105:
                    l[n] = i
                    r[n] = j
                    n += 1
        return l, r
    b = []
    i, j = substract(x_hat, noise_image)
    for lg in range(len(i)):
        b.append('{}, {}'.format(i[lg], j[lg]))

    name = path.split(os.sep)[-1]
    content["Height"] = 128
    content["Width"] = 128
    content["name"] = name

    p = {}
    b = []
    i, j = substract(x_hat, noise_image)
    for lg in range(len(i)):
        if i[lg] == -1:
            break
        b.append('{}, {}'.format(i[lg], j[lg]))
    p["points"] = b
    ps = [p]

    content["regions"] = ps
    # print("content:", content)
    json_str = json.dumps(content)
    new_dict = json.loads(json_str)
    image_name = name.split('.')[-2]
    with open('./data/focusight1_round1_train_part2/TC_Images/'+str(image_name)+'.json', 'w') as f:
        json.dump(new_dict, f)
        print('加载文件中...\t{}/{}'.format(step+1, len(paths)))

