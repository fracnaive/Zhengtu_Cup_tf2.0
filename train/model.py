# 模型训练时，需要先将AES里的trainable参数改为True，模型调用改为False

import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential, layers
import code.config as config

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


AES = Sequential([
    layers.Conv2D(64, kernel_size=5, strides=3, padding='valid', input_shape=(128, 128, 1)),
    layers.Activation(tf.nn.leaky_relu),

    layers.Conv2D(128, kernel_size=5, strides=3, padding='valid'),
    layers.BatchNormalization(trainable=False),
    layers.Activation(tf.nn.leaky_relu),

    layers.Conv2D(256, 5, 3, 'valid'),
    layers.BatchNormalization(trainable=False),
    layers.Activation(tf.nn.leaky_relu),

    layers.Flatten(),

    layers.Dense(256),
    layers.Activation(tf.nn.leaky_relu),
    layers.Dense(128),
    layers.Activation(tf.nn.leaky_relu),
    layers.Dense(config.h_dim),

    layers.Dense(7 * 7 * 512),
    layers.Reshape((7, 7, 512)),
    layers.Activation(tf.nn.leaky_relu),

    layers.Conv2DTranspose(256, 2, 2, 'valid'),
    layers.BatchNormalization(trainable=False),
    layers.Activation(tf.nn.leaky_relu),

    layers.Conv2DTranspose(128, 3, 3, 'valid'),
    layers.BatchNormalization(trainable=False),
    layers.Activation(tf.nn.leaky_relu),

    layers.Conv2DTranspose(1, 5, 3, 'valid')
])


class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        self.conv1 = layers.Conv2D(64, kernel_size=5, strides=3, padding='valid')
        self.conv2 = layers.Conv2D(128, kernel_size=5, strides=3, padding='valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(256)
        self.fc2 = layers.Dense(128)
        self.fc3 = layers.Dense(config.h_dim)

        # Decoders
        # z: [b, h_dim] => [b, 7*7*512] => [b, 7, 7, 512] => [b, 128, 128, 1]
        self.fc11 = layers.Dense(7 * 7 * 512)
        # channel, kernel_size, strides, padding
        self.conv11 = layers.Conv2DTranspose(256, 2, 2, 'valid')
        self.bn11 = layers.BatchNormalization()

        self.conv12 = layers.Conv2DTranspose(128, 3, 3, 'valid')
        self.bn12 = layers.BatchNormalization()
        # 必须使得最后输出的size为128*128*1
        self.conv13 = layers.Conv2DTranspose(1, 5, 3, 'valid')

    def encoder(self, x, training=None):
        # x.shape: [b, 128, 128, 1] => [b, h_dim]
        x = tf.nn.leaky_relu(self.conv1(x))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x = self.flatten(x)
        x = tf.nn.leaky_relu(self.fc1(x))
        x = tf.nn.leaky_relu(self.fc2(x))
        out = self.fc3(x)
        return out

    def decoder(self, x, training=None):
        # [b, h_dim] => [b, 128, 128, 1]
        x = self.fc11(x)
        x = tf.reshape(x, [-1, 7, 7, 512])
        x = tf.nn.leaky_relu(x)
        x = tf.nn.leaky_relu(self.bn11(self.conv11(x), training=training))
        x = tf.nn.leaky_relu(self.bn12(self.conv12(x), training=training))
        out = self.conv13(x)
        return out

    def call(self, inputs, training=None):
        # [b, 128, 128, 1]=>[b, h_dim]
        x = self.encoder(inputs, training=training)
        # print('encoder.shape:', h.shape)
        # [b, h_dim]=>[b, 128, 128, 1]
        x_hat = self.decoder(x, training=training)
        # print('decoder.shape:', x_hat.shape)

        return x_hat


DS = Sequential([
    layers.Conv2D(64, 5, 3, 'valid', input_shape=(128, 128, 1)),
    layers.Activation(tf.nn.leaky_relu),

    layers.Conv2D(128, 5, 3, 'valid'),
    layers.BatchNormalization(trainable=True),
    layers.Activation(tf.nn.leaky_relu),

    layers.Conv2D(256, 5, 3, 'valid'),
    layers.BatchNormalization(trainable=True),
    layers.Activation(tf.nn.leaky_relu),

    layers.Flatten(),

    layers.Dense(1)
])


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        # [b, 128, 128, 1] => [b, 1]
        self.conv1 = layers.Conv2D(64, 5, 3, 'valid')
        self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        # [b, h, w, 1] => [b, -1]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):

        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

        # [b, h, w, c] => [b, -1]
        x = self.flatten(x)
        # print(x.shape)
        # [b, -1] => [b, 1]
        logits = self.fc(x)

        return logits


# def main():
#     pass
#
#
# if __name__ == '__main__':
#     main()
