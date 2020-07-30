import tensorflow as tf

train_noise_path = '../temp_data/part1_noise_images/'
train_label_path = 'D:/zhengtu_cup/data/focusight1_round1_train_part1/OK_Images/'
test_path = 'D:/zhengtu_cup/data/focusight1_round1_train_part1/TC_Images'

train_noise_path2 = '../temp_data/part2_noise_images/'
train_label_path2 = 'D:/zhengtu_cup/data/focusight1_round1_train_part2/OK_Images/'
test_path2 = 'D:/zhengtu_cup/data/focusight1_round1_train_part2/TC_Images'

image_len = 1000

img_mean = tf.constant([0.485])
img_std = tf.constant([0.229])

h_dim = 100
