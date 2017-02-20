import sys
import time
import cv2
import numpy as np
import tensorflow as tf
from net import inference
from loader import getDataFromPic, IMAGE_SIZE
from ctypes import *
from collections import Counter

FRAME_SIZE = 640 * 480 * 3

IMAGE_HEIGH = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE
NUM_EPOCHS = 20
NETPATH = 'data/net.ckpt'
MIN_PREDICTION = 0.8

def main(argv=None):
    print 'Loading......'
    begin_time = time.time()

    print 'Building net......'

    x = tf.placeholder(tf.float32, shape=[1, IMAGE_HEIGH, IMAGE_WIDTH, 1])
    keep_prob = tf.placeholder(tf.float32)

    train_prediction = inference(x, keep_prob)
    prediction = tf.nn.softmax(train_prediction)


    def eval_in_batches(data, sess):
        feed_dict = {x: np.reshape(data, [1, IMAGE_HEIGH, IMAGE_WIDTH, 1]),
                     keep_prob: 1.0}
        tp, p = sess.run([train_prediction, prediction], feed_dict=feed_dict)
        return tp, p


    saver = tf.train.Saver()


    hand_array = c_ubyte * FRAME_SIZE
    hand_data = hand_array(0)

    rows_array = c_int * 1
    cols_array = c_int * 1
    rows = rows_array(0)
    cols = cols_array(0)

    clib = cdll.LoadLibrary("./libhandpose.so")
    clib.init()

    state = 's'
    machine = {}
    machine['s'] = ['s', 'a1', 's', 'c1', 'b1', 's']

    machine['a1'] = ['a4', 'a1', 's', 's', 's', 'a2']
    machine['a2'] = ['a4', 's', 's', 's', 's', 'a3']
    machine['a3'] = ['a4', 's', 's', 's', 's', 's']
    machine['a4'] = ['a4', 'a7', 's', 's', 's', 'a5']
    machine['a5'] = ['s', 'a7', 's', 's', 's', 'a6']
    machine['a6'] = ['s', 'a7', 's', 's', 's', 's']
    machine['a7'] = ['s', 'a1', 's', 's', 's', 's']

    machine['b1'] = ['b4', 's', 's', 's', 'b1', 'b3']
    machine['b2'] = ['b4', 's', 's', 's', 's', 'b3']
    machine['b3'] = ['b4', 's', 's', 's', 's', 's']
    machine['b4'] = ['b4', 's', 's', 's', 'b7', 'b5']
    machine['b5'] = ['s', 's', 's', 's', 'b7', 'b6']
    machine['b6'] = ['s', 's', 's', 's', 'b7', 's']
    machine['b7'] = ['s', 's', 's', 's', 'b1', 's']

    machine['c1'] = ['s', 's', 'c4', 'c1', 's', 'c2']
    machine['c2'] = ['s', 's', 'c4', 's', 's', 'c3']
    machine['c3'] = ['s', 's', 'c4', 's', 's', 's']
    machine['c4'] = ['c3', 'c3', 'c4', 'c3', 'c3', 'c3']

    label_array = np.zeros((1,3), np.int) + 5
    i = 0

    with tf.Session() as sess:
        saver.restore(sess, NETPATH)


        while (cv2.waitKey(10) & 0xFF) != 27:
            time_temp = time.time()
            if clib.get_hand_color(hand_data, cols, rows) < 0:
                label_present = 5
                hand = None

            else:
                hand = np.reshape(hand_data, FRAME_SIZE)
                hand = np.asarray(hand[:cols[0] * rows[0] * 3])
                hand = np.reshape(hand, (rows[0], cols[0], 3))
                hand = np.asarray(hand)
                data = getDataFromPic(hand)
                batch_data = np.reshape(data, [1, IMAGE_HEIGH, IMAGE_WIDTH, 1])
                tp, p = eval_in_batches(batch_data, sess)
                label_prediction = np.argmax(p)



                if p[0][label_prediction] >= MIN_PREDICTION:
                    label_present = label_prediction
                    cv2.putText(hand, "detect: {}".format(label_prediction), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                else:
                    label_present = 5
                    cv2.putText(hand, "detect: error", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                sys.stdout.flush()

            label_array[0][i] = label_present
            if i == 2:
                label_counter = Counter(label_array.flatten().tolist()).most_common(1)
                if label_counter[0][1] > 1:
                    window_label = label_counter[0][0]
                else:
                    window_label = 5

                # print 'window_label: ' , window_label

                state = machine[state][window_label]
                if cmp(state, 'a1') == 0:
                    print 'Choose.'
                elif cmp(state, 'a7') == 0:
                    print 'Click!!!!!!!!!!!!!!'
                elif cmp(state, 'b7') == 0:
                    print 'Return.'
                elif cmp(state, 'c4') == 0:
                    print 'Drag.'

            if hand is not None:
                if cmp(state, 'a1') == 0:
                    cv2.putText(hand, "Move", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                elif cmp(state, 'a7') == 0:
                    cv2.putText(hand, "Click", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                elif cmp(state, 'b7') == 0:
                    cv2.putText(hand, "Return", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                elif cmp(state, 'c4') == 0:
                    cv2.putText(hand, "Drag", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                else:
                    cv2.putText(hand, "None", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                frame_time = time.time() - time_temp
                fps = 1 / frame_time
                cv2.putText(hand, "fps: {}".format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.imshow("hand", hand)

            i = (i + 1) % 3

    clib.release()

    elapsed_time = time.time() - begin_time
    print('Total time: %.1f s' % elapsed_time)


if __name__ == '__main__':
    tf.app.run()
