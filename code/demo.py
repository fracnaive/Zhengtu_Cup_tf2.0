# 空格键显示下一张测试图像，ESC键退出

import tensorflow as tf
import numpy as np
import config
import cv2
import os
import glob
from numba import jit
from train.model import AES
import time


def main():
    key = 0
    paths = glob.glob(os.path.join(config.test_path2, '*.bmp'))
    for z_path in paths:
        if key == 27:
            break

        rec_model = AES
        # z_path = os.path.join(config.test_path, '00b5CacodX4rVA9jmo6aad56xvwMT3.bmp')
        src = cv2.imread(z_path)
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
        print('x_hat.shape:', x_hat.shape)

        noise_image_src = cv2.imread(z_path)
        noise_image = noise_image_src.copy()
        noise_image = cv2.cvtColor(noise_image, cv2.COLOR_BGR2GRAY)
        print(noise_image.shape)

        # gpu加速，处理两个for循环时间过长的问题
        @jit
        def substract(a, b):
            temp = np.zeros(b.shape, np.uint8)
            for i in range(128):
                for j in range(128):
                    noise_lbp_point = b[i][j]
                    rec_lbp_point = a[i][j]
                    # 由于这些数组都是uint8类型的，小值减去大值不会得到负数，得到的是错误的值，所以需要提前判断大小
                    if noise_lbp_point > rec_lbp_point:
                        error_point = noise_lbp_point - rec_lbp_point
                    else:
                        error_point = rec_lbp_point - noise_lbp_point
                    temp[i][j] = error_point
            return temp

        t0 = time.time()
        temp = substract(x_hat, noise_image)
        t1 = time.time()
        print("加速时间：", (t1 - t0))

        # _, error = cv2.threshold(temp, 25, 255, cv2.THRESH_BINARY)  # part1部分阈值选用为25
        _, error = cv2.threshold(temp, 105, 255, cv2.THRESH_BINARY)  # part2部分阈值选用为105
        # cv2.imwrite('demo1.bmp', error)
        x_hat = x_hat.numpy()
        # print(x_hat)
        # print(temp.shape)
        print(np.min(temp))
        print(np.max(temp))

        name = z_path.split(os.sep)[-1]

        while 1:
            cv2.imshow(name, noise_image)
            cv2.imshow('rec_image', x_hat)
            cv2.imshow('defect', error)

            value = cv2.waitKey(10)
            if value == 32:
                break
            elif value == 27:
                key = 27
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
