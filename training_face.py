from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os.path
import time
import tool
import vgg
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf

num_step = 4500

# Import data
#tfrecords_file_train = 'lfw_train.tfrecords'
#tfrecords_file_test = 'lfw_test.tfrecords'
#train_dir = '/data/shenyc/face/'
train_log_dir = '/data/shenyc/face/log0424/'

#filename = os.path.join(train_dir, tfrecords_file_train)
#filename_test = os.path.join(train_dir, tfrecords_file_test)i
with tf.name_scope('input'):
    #filename_queue = tf.train.string_input_producer([filename])
    #filename_queue_test = tf.train.string_input_producer([filename_test], num_epochs = 3)
    #images, label = tool.read_and_decode(filename_queue)
    #images_test, label_test = read_and_decode(filename_queue_test)
    train_image = np.load("train_image.npy")
    train_label = np.load("train_label.npy")
    images = tf.convert_to_tensor(train_image, dtype = tf.int32)
    images = tf.image.resize_images(images, (64, 64))
    label = tf.convert_to_tensor(train_label, dtype = tf.int32)
    label = tf.one_hot(tf.cast(label, tf.int32), depth = 2)
    images_batch, label_batch = tf.train.shuffle_batch([images, label],
                                                       enqueue_many=True, batch_size=50, num_threads=1,capacity=1000 + 3 * 25, min_after_dequeue = 1000)
    #images_test_batch, label_test_batch = tf.train.batch([images_test, label_test], batch_size = 125, num_threads = 64, capacity = 1000+3*15)

#filename_test = os.path.join(train_dir, tfrecords_file_train)
#images_test, label_test = read_and_decode(filename_test)

x = tf.placeholder(tf.float32, [None, 64, 64, 2], name = "x")
#tf.summary.image('input', x, 3)

  # Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2], name = "labels")

  # Build the graph for the deep net
y_conv = vgg.VGG16(x, 2, True)

with tf.name_scope('loss'):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy, name = "loss")
  tf.summary.scalar("loss", cross_entropy)

with tf.name_scope('adam_optimizer'):
  train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  tf.summary.scalar("accuracy", accuracy)

summ = tf.summary.merge_all()

#graph_location = tempfile.mkdtemp()
#print('Saving graph to: %s' % graph_location)
#train_writer = tf.summary.FileWriter(graph_location)
#train_writer.add_graph(tf.get_default_graph())

saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
  #tool.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  tra_summary_writer = tf.summary.FileWriter(train_log_dir)
  tra_summary_writer.add_graph(sess.graph)
  try:
      for i in range(num_step):
          images, label = sess.run([images_batch, label_batch])
          #images_test, label_test = sess.run([images_test_batch, label_test_batch])
          #print(images, label)
          #images_test, label_test = sess.run([images_test, label_test])
          if i % 100 == 0:
              train_accuracy = accuracy.eval(feed_dict={x: images, y_: label})
              #test_accuracy = accuracy.eval(feed_dict={x: images_test, y_: label_test, keep_prob: 1.0})
          _, tra_loss, summary = sess.run([train_step, cross_entropy, summ], feed_dict={x: images, y_: label})
          tra_summary_writer.add_summary(summary, i)
          if i % 100 == 0:
              print('step %d, loss %.4f, trainig accuracy %.4f' %(i, tra_loss, train_accuracy))
          if i % 500 == 0 or (i + 1) == num_step:
              checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=i)
  except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
  finally:
      coord.request_stop()

    #print('test accuracy %.4f' % accuracy.eval(feed_dict={
    #    x: images_test, y_: label_test, keep_prob: 1.0}))

  coord.request_stop()
  coord.join(threads)
