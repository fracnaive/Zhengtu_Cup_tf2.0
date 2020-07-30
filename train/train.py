# 模型训练时，需要先将AES里的trainable参数改为True，模型调用改为False

# import sys
# sys.path.append('..')
import os
import tensorflow as tf
import numpy as np
from model import AES, DS
from PIL import Image
import cv2
import code.config as config
from dataset import load_data_path, prep

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


# 把多张image保存到一张image函数里面去
def save_image(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 80):
        for j in range(0, 280, 80):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)


def celoss_ones(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1, 1,],label全为1
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # [b, 1]
    # [b] = [0, 0, 0, 0, 0,],label全为0
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


# 梯度惩罚WGAN-GP
def gradient_penalty(discriminator, label, fake_image):
    batchsz = label.shape[0]
    # [b, h, w, c]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # [b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, label.shape)
    interplate = t * label + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        # interplate 是一个tensor类型，对它求梯度需要人为地将它加到watch list里面去，如果是variable就不需要加了
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate)
    grads = tape.gradient(d_interplote_logits, interplate)

    # grads:[b, h, w, c]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    # 对梯度求范数
    gp = tf.norm(grads, axis=1)  # [b]
    # 计算均方差
    gp = tf.reduce_mean((gp-1)**2)
    return gp


# 计算准确率,...,非常耗时
# def cc_num(bs, logits, labels, value):
#     correct_img_num = 0
#     for b in range(bs):
#         correct_point_num = 0
#         for i in range(128):
#             for j in range(128):
#                 src_point = labels[b][i][j] * 255.0
#                 rec_point = logits[b][i][j] * 255.0
#                 # 由于这些数组都是uint8类型的，小值减去大值不会得到负数，得到的是错误的值，所以需要提前判断大小
#                 if src_point > rec_point:
#                     error_point = src_point - rec_point
#                 else:
#                     error_point = rec_point - src_point
#                 if error_point < value:
#                     correct_point_num += 1
#         acc_per_img = 100 * correct_point_num / (128*128)
#         if acc_per_img > 90:
#             correct_img_num += 1
#     return correct_img_num


h_dim = 100
batchsz = 16
lr = 1e-3

images_train, labels_train = load_data_path(config.train_noise_path2, config.train_label_path2)
train_db = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
train_db = train_db.shuffle(10000).map(prep).batch(batchsz)


def main():

    # 自编码器网络，输入图像大小为128*128的单通道图像，输出也为大小128*128的单通道图像
    # rec_model = AE()
    # rec_model.build(input_shape=(None, 128, 128, 1))
    rec_model = AES
    rec_model.summary()

    # 鉴别器网络，输入图像大小为128*128的单通道图像，输出大小为1
    # disc_model = Discriminator()
    # disc_model.build(input_shape=(None, 128, 128, 1))
    disc_model = DS
    disc_model.summary()

    # 设置两个网络的优化器均为Adam
    rec_optimizer = tf.optimizers.Adam(lr=lr)
    disc_optimizer = tf.optimizers.Adam(lr=lr)

    min_loss = 100000

    for epoch in range(1001):
        sum_loss = 0
        for step, (x, y) in enumerate(train_db):

            # train D
            with tf.GradientTape() as tape:
                rec_logits = rec_model(x, training=True)
                # 将网络得到的输出图像像素值均转换为0~1之间
                x_rec = tf.nn.sigmoid(rec_logits)
                x_rec = tf.cast(x_rec, dtype=tf.float32)
                x_rec = tf.image.resize(x_rec, [128, 128])

                # y为真实好品图像，将真实好品图送入鉴别器得到一个输出
                d_y_logits = disc_model(y, training=True)
                # 通过将经过自编码器网络重建的图像送入鉴别器得到一个输出
                d_rec_logits = disc_model(x_rec, training=True)
                # 鉴别器网络的loss要求，真实好品图鉴别为真（即输出为1），而重建的图鉴别为假（即输出为0）
                d_loss_real = celoss_ones(d_y_logits)
                d_loss_fake = celoss_zeros(d_rec_logits)
                # 梯度惩罚
                gp = gradient_penalty(disc_model, y, x_rec)
                # 最终的鉴别器loss为以上三者之和
                d_loss = d_loss_real + d_loss_fake + 1.0 * gp
            grads = tape.gradient(d_loss, disc_model.trainable_variables)
            disc_optimizer.apply_gradients(zip(grads, disc_model.trainable_variables))

            # # 每disc_model训练5次，rec_model训练1次
            # if epoch % 5 == 0:
            # train AE
            with tf.GradientTape() as tape:
                rec_logits = rec_model(x, training=True)
                x_rec = tf.nn.sigmoid(rec_logits)
                x_rec = tf.cast(x_rec, dtype=tf.float32)
                x_rec = tf.image.resize(x_rec, [128, 128])

                d_rec_logits = disc_model(x_rec, training=True)
                # 自编码器网络loss的要求，鉴别器将重建的图像鉴别为真（即输出为1）
                d_error_loss = celoss_ones(d_rec_logits)
                # 同时重建的图像需要与原图像差值很小
                ae_loss = tf.losses.binary_crossentropy(y, rec_logits, from_logits=True)
                ae_loss = tf.reduce_mean(ae_loss) * 500.0
                # 最终自编码器网络loss为以上两者之和
                rec_loss = 1.0 * ae_loss + d_error_loss

            grads = tape.gradient(rec_loss, rec_model.trainable_variables)
            rec_optimizer.apply_gradients(zip(grads, rec_model.trainable_variables))
            sum_loss += rec_loss

            if step % 10 == 0:
                print('epoch:{}\t'.format(epoch),
                      'step:{}\t'.format(step),
                      'd_loss:{}\t'.format(float(d_loss)),
                      'ae_loss:{}\t'.format(float(ae_loss)),
                      'd_error_loss:{}\t'.format(float(d_error_loss)),
                      'rec_loss:{}\t'.format(float(rec_loss)))
        sum_loss = sum_loss / 1000.0
        if sum_loss < min_loss:
            min_loss = sum_loss
            rec_model.save("../model/rec_model_part2.h5")
            disc_model.save("../model/disc_model_part2.h5")
            print('<INFO> min loss is {}'.format(sum_loss))
            print('<INFO> model updated...')

            # evaluation
            # (y, x) = next(iter(train_db))
            # logits = model(tf.reshape(x, [-1, 4096]))
            # x_hat = tf.nn.sigmoid(logits)
            # # x_hat.shape [b, 784]
            # x_hat = tf.reshape(x_hat, [-1, 64, 64])
            #
            # x = tf.squeeze(x, axis=3)
            # x_concat = tf.concat([x, x_hat], axis=0)
            # x_concat = x_hat
            # x_concat = x_concat.numpy()*255.
            # x_concat = x_concat.astype(np.uint8)
            # save_image(x_concat, 'ae_images/rec_epoch_%d.png' % epoch)
        # if epoch % 5 == 0:
        z_path = os.path.join(config.test_path2, '00wlIJpgj81RJYwMgGZsMd6ZLkoLbL.bmp')
        src = cv2.imread(z_path)
        gray = src.copy()
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = tf.expand_dims(gray, axis=2)
        # print(gray.shape)
        # gray = tf.io.read_file(z_path)
        # gray = tf.image.decode_bmp(gray, channels=1)
        z = tf.cast(gray, dtype=tf.float32) / 255.  # (128, 128, 1)
        z = tf.expand_dims(z, axis=0)  # (1, 128, 128, 1)

        logits = rec_model(z, training=False)
        x_hat = tf.nn.sigmoid(logits)
        x_hat = tf.cast(x_hat, dtype=tf.float32)
        x_hat = tf.image.resize(x_hat, [128, 128])
        x_hat = tf.squeeze(x_hat, axis=3)
        x_hat = x_hat.numpy() * 255.
        x_hat = x_hat.astype(np.uint8)
        cv2.imwrite('part2_images/aaa{}.bmp'.format(int(epoch)), x_hat[0])


if __name__ == '__main__':
    main()
