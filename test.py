import sys
import time
import cv2
import numpy as np
import tensorflow as tf
from net import inference
from loader import loadDataLabel, IMAGE_SIZE

IMAGE_HEIGH = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE
DATADIR = 'dataset_color_test'
NETPATH = 'data/net.ckpt'
EVAL_FREQUENCY = 1000

def main(argv=None):
    print 'Loading......'
    start_time = time.time()
    begin_time = start_time

    data, label = loadDataLabel(DATADIR, various=True, shuffle=True)
    train_size = len(label)
    print 'Loaded %d images.' % train_size

    elapsed_time = time.time() - start_time
    print('Loading images with label elapsed %.1f s' % elapsed_time)
    print 'Building net......'
    start_time = time.time()

    x = tf.placeholder(tf.float32, shape=[1, IMAGE_HEIGH, IMAGE_WIDTH, 1], name='data')
    keep_prob = tf.placeholder(tf.float32, name='prob')

    train_prediction = inference(x, keep_prob)
    prediction = tf.nn.softmax(train_prediction)


    def eval_in_batches(data, sess):
        feed_dict = {x: np.reshape(data, [1, IMAGE_HEIGH, IMAGE_WIDTH, 1]),
                     keep_prob: 1.0}
        tp, p = sess.run([train_prediction, prediction], feed_dict=feed_dict)
        return tp, p

    elapsed_time = time.time() - start_time
    print('Building net elapsed %.1f s' % elapsed_time)
    print 'Begin testing..., train dataset size:{0}'.format(train_size)
    start_time = time.time()

    saver = tf.train.Saver()

    elapsed_time = time.time() - start_time
    print('loading net elapsed %.1f s' % elapsed_time)
    start_time = time.time()

    ls = []
    with tf.Session() as sess:
        saver.restore(sess, NETPATH)
        # saver.save(sess, 'pb_saver/net.ckpt')
        tf.train.write_graph(sess.graph_def, '.', 'data/train.pb', False)
        for i in range(train_size):
            batch_data = np.reshape(data[i, ...], [1, IMAGE_HEIGH, IMAGE_WIDTH, 1])
            tp, p = eval_in_batches(batch_data, sess)
            label_prediction = np.argmax(p)
            ls.append(label_prediction)
            if i % EVAL_FREQUENCY == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d, %.1f ms.' %
                      (i, 1000 * elapsed_time / EVAL_FREQUENCY))
                print('True label: %d' % label[i])
                print('Prediction: %d' % label_prediction)
                # cv2.imshow('data', np.reshape(batch_data, [IMAGE_HEIGH, IMAGE_WIDTH]))
                # cv2.waitKey(0)
            sys.stdout.flush()


    ls = np.asarray(ls, np.int)
    error_count = train_size - np.sum(ls == label)
    error_rate = 100.0 * error_count / train_size
    print('Total size: %d, Test error count: %d, error rate: %f%%' % (train_size, error_count, error_rate))

    elapsed_time = time.time() - begin_time
    print('Total time: %.1f s' % elapsed_time)


if __name__ == '__main__':
    tf.app.run()
