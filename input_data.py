from __future__ import print_function

from sklearn.datasets import fetch_lfw_pairs
import tensorflow as tf
import numpy as np
import os

lfw_pairs_train = fetch_lfw_pairs(subset='train', resize = 1)
print(lfw_pairs_train.target.shape)
print(lfw_pairs_train.data.shape)
train_img = lfw_pairs_train.data.reshape((-1, 125, 94, 2))
train_label = lfw_pairs_train.target
train_label = train_label.tolist()
print(train_img.shape)
print(type(train_label))

lfw_pairs_test = fetch_lfw_pairs(subset='test', resize = 1)
test_img = lfw_pairs_test.data.reshape(-1, 125, 94, 2)
test_label = lfw_pairs_test.target
test_label = test_label.tolist()
print(test_img.shape)
print(type(test_label))
print(train_img[0,:,:,:].shape)

def int64_feature(value):
    '''Wrapper for inserting int64 features into Example proto'''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def convert_to_tfrecord(images, labels, save_dir, name):
    '''
    Args:
        images: list of images
        labels: list of labels
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1'
        name: the name of tfrecord file, string type, e.g.: 'train'
    '''
    n_samples = len(labels)

    filename = os.path.join(save_dir, name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    print('\nTransform start......')

    i = 0
    j = 0
    for i in np.arange(0, n_samples):
        try:
            image = images[i,:,:,:]
            image_raw = image.tostring()
            label = int(labels[i])
            height, width, channel = image.shape
            example = tf.train.Example(features = tf.train.Features(feature={
                'height': int64_feature(height),
                'width': int64_feature(width),
                'channel': int64_feature(channel),
                'image_raw': bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i,:,:,:])
            print('error: %s' %e)
            print('Skip it\n')
    writer.close()
    print('Transform done!\n')

def read_and_decode(tfrecords_file, batch_size):
    '''
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channel': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    height = tf.cast(img_features['height'], tf.int32)
    width = tf.cast(img_features['width'], tf.int32)
    channel = tf.cast(img_features['channel'], tf.int32)
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)

    image = tf.reshape(image, [height, width, channel])
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size = batch_size,
                                                      num_threads = 64,
                                                      capacity = 2000,
                                                      min_after_dequeue = 1000)

    return image_batch, tf.reshape(label_batch, [batch_size])

save_dir = '/Users/chenyucong/Desktop/master_course/cs698/code/'
name_train = 'lfw_train'
name_test = 'lfw_test'
convert_to_tfrecord(train_img, train_label, save_dir, name_train)
convert_to_tfrecord(test_img, test_label, save_dir, name_test)
