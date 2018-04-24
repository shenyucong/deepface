import os
import os.path

import numpy as np
import tensorflow as tf
import training_bee
import math
import tool
import vgg
import numpy as np

#tfrecords_file_test = 'bees_test.tfrecords'
#train_dir = '/Users/chenyucong/Desktop/research/ecology'
log_dir = '/data/shenyc/face/log0424'

filename = os.path.join(train_dir, tfrecords_file_test)
with tf.name_scope('input'):
    #filename_queue = tf.train.string_input_producer([filename])
    #images_test, label_test = training_bee.read_and_decode(filename_queue)
    test_image = np.load("test_image.npy")
    test_label = np.load("test_label.npy")
    images_test = tf.convert_to_tensor(test_image, dtype = tf.int32)
    images_test = tf.image.resize_images(image, (64, 64))
    label_test = tf.convert_to_tensor(test_label, dtype = tf.int32)
    label_test = tf.one_hot(tf.cast(label_test, tf.int32), depth = 2)

    images_test_batch, label_test_batch = tf.train.batch([images_test, label_test],
                                                         enqueue_many = True, batch_size = 50, num_threads=1, capacity=1000+3*25)

y_conv = vgg.VGG16(images_test)
correct = tool.num_correct_prediction(y_conv, label_test)
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
        return

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    try:
        print('\nEvaluating......')
        num_step = int(math.floor(1000 / 50))
        num_sample = num_step*50
        step = 0
        total_correct = 0
        while step < num_step and not coord.should_stop():
            batch_correct = sess.run(correct)
            total_correct += np.sum(batch_correct)
            step += 1
        print('Total testing samples: 1000')
        print('Total correct predictions: %d' %total_correct)
        print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)
