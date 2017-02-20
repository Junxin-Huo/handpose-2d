import sys
import time
import numpy as np
import tensorflow as tf
from net import inference, total_loss, train, BATCH_SIZE
from loader import loadDataLabel, IMAGE_SIZE

IMAGE_HEIGH = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE
DATADIR = 'dataset_color_train'
NUM_EPOCHS = 30
NETPATH = 'data/net.ckpt'
PBPATH = 'data/train.pb'
EVAL_FREQUENCY = 50
KEEP_PROB = 0.5

def main(argv=None):
    with tf.Graph().as_default():
        print 'Start.'
        start_time = time.time()
        begin_time = start_time

        print 'Loading images.'
        data, label = loadDataLabel(DATADIR, various=True, shuffle=True)
        train_size = len(label)
        print 'Loaded %d images.' % train_size

        elapsed_time = time.time() - start_time
        print('Loading images with label elapsed %.1f s' % elapsed_time)
        print 'Building net......'
        start_time = time.time()

        def get_input_x(x, offset=0, length=BATCH_SIZE):
            a = x[offset:(offset + length), ...]
            return np.reshape(a, [length, IMAGE_HEIGH, IMAGE_WIDTH, 1])

        def get_input_y(y, offset=0, length=BATCH_SIZE):
            b = y[offset:(offset + length)]
            return np.reshape(b, [length,])

        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_HEIGH, IMAGE_WIDTH, 1], name='data')
        y = tf.placeholder(tf.int32, shape=[BATCH_SIZE,])
        keep_prob = tf.placeholder(tf.float32, name='prob')

        # Train model.
        train_prediction = inference(x, keep_prob)

        batch = tf.Variable(0, dtype=tf.float32)

        learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            train_size,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        tf.summary.scalar('learn', learning_rate)

        loss = total_loss(train_prediction, y)
        tf.summary.scalar('loss', loss)

        trainer = train(loss, learning_rate, batch)

        elapsed_time = time.time() - start_time
        print('Building net elapsed %.1f s' % elapsed_time)
        print 'Begin training..., train dataset size:{0}'.format(train_size)
        start_time = time.time()
        best_validation_loss = 100000.0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('graph/train', sess.graph)

            # Inital the whole net.
            tf.global_variables_initializer().run()
            print('Initialized!')
            for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
                offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)

                batch_data = get_input_x(offset=offset, x=data)
                batch_labels = get_input_y(offset=offset, y=label)

                # Train CNN net.
                feed_dict = {x: batch_data,
                             y: batch_labels,
                             keep_prob: KEEP_PROB}
                summary, _, l, lr, predictions = sess.run(
                    [merged, trainer, loss, learning_rate, train_prediction],
                    feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                if l < best_validation_loss:
                    print 'Saving net.'
                    print('Net loss:%.3f, learning rate: %.6f' % (l, lr))
                    best_validation_loss = l
                    saver.save(sess, NETPATH)
                if step % EVAL_FREQUENCY == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print('Step %d (epoch %.2f), %.1f ms' %
                          (step, np.float32(step) * BATCH_SIZE / train_size,
                           1000 * elapsed_time / EVAL_FREQUENCY))
                    print('Net loss:%.3f, learning rate: %.6f' % (l, lr))
                sys.stdout.flush()
            train_writer.close()

        elapsed_time = time.time() - begin_time
        print('Total time: %.1f s' % elapsed_time)

if __name__ == '__main__':
    tf.app.run()