import tensorflow as tf
import os
import code.config as config


def load_data_path(noise_path, label_path):
    imgs, labels = [], []
    for i in range(1, config.image_len):
        img = os.path.join(config.train_noise_path, 'image_'+str(i)+'.bmp')
        label = os.path.join(config.train_label_path, 'Image_'+str(i)+'.bmp')
        imgs.append(img)
        labels.append(label)
    return imgs, labels


def normalize(x, mean=config.img_mean, std=config.img_std):
    x = (x - mean) / std
    return x


def prep(noise_path, label_path):
    noise = tf.io.read_file(noise_path)
    noise = tf.image.decode_bmp(noise, channels=1)
    noise_tensor = tf.cast(noise, dtype=tf.float32) / 255.
    x = tf.image.resize(noise_tensor, [128, 128])

    label = tf.io.read_file(label_path)
    label = tf.image.decode_bmp(label, channels=1)
    label_tensor = tf.cast(label, dtype=tf.float32) / 255.
    y = tf.image.resize(label_tensor, [128, 128])

    return x, y


def main():
    images_train, labels_train = load_data_path(config.train_noise_path, config.train_label_path)
    db_train = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    db_train = db_train.shuffle(10000).map(prep).batch(1)

    sample_train = next(iter(db_train))
    print('sample_train:', sample_train[0].shape, sample_train[1].shape)


if __name__ == '__main__':
    main()
