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
#train_label = train_label.tolist()
print(train_img.shape)
print(type(train_label))

lfw_pairs_test = fetch_lfw_pairs(subset='test', resize = 1)
test_img = lfw_pairs_test.data.reshape(-1, 125, 94, 2)
test_label = lfw_pairs_test.target
#test_label = test_label.tolist()
print(test_img.shape)
print(type(test_label))
print(train_img[0,:,:,:].shape)

np.save("train_image.npy", train_img)
np.save("train_label.npy", train_label)
np.save("test_image.npy", test_img)
np.save("test_label.npy", test_label)

#train_image = tf.convert_to_tensor(train_img, dtype = tf.int32)
#train_label = tf.convert_to_tensor(train_label, dtype = tf.int32)
#test_image = tf.convert_to_tensor(test_img, dtype = tf.int32)
#test_label = tf.convert_to_tensor(test_label, dtype = tf.int32)
#print(train_image)
#print(train_label)
#print(test_image)
#print(test_label)



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
                'label':int64_feature(label),
                'image_raw': bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i,:,:,:])
            print('error: %s' %e)
            print('Skip it\n')
    writer.close()
    print('Transform done!\n')

#save_dir = '/Users/chenyucong/Desktop/master_course/cs698/code/'
#name_train = 'lfw_train'
#name_test = 'lfw_test'
#convert_to_tfrecord(train_img, train_label, save_dir, name_train)
#convert_to_tfrecord(test_img, test_label, save_dir, name_test)
