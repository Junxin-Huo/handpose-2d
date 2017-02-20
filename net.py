import tensorflow as tf
import numpy as np

BATCH_SIZE = 32


def _variable_with_weight_decay(shape, stddev, wd, name):
    # Genetate value weight and added to weight decay L2.
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev),
                      dtype=tf.float32,
                      name=name)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(data, prob):
    # Convolution Layer 1
    shape = [5, 5, 1, 6]
    stddev = np.sqrt(2. / float(np.prod(shape[:-1])))
    W_conv1 = _variable_with_weight_decay(shape=shape,
                                          stddev=np.asarray(stddev, np.float32),
                                          wd=0.0,
                                          name='W_conv1')
    b_conv1 = tf.Variable(tf.constant(0.0, shape = [6]),
                          dtype=tf.float32,
                          name='b_conv1')

    h_conv1 = tf.nn.relu(
    tf.nn.conv2d(data, W_conv1,
                     strides=[1, 1, 1, 1],
                     padding='VALID')
    + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',)

    # Convolution Layer 2
    shape = [3, 3, 6, 16]
    stddev = np.sqrt(2. / float(np.prod(shape[:-1])))
    W_conv2 = _variable_with_weight_decay(shape=shape,
                                          stddev=np.asarray(stddev, np.float32),
                                          wd=0.0,
                                          name='W_conv2')
    b_conv2 = tf.Variable(tf.constant(0.0, shape=[16]),
                          dtype=tf.float32,
                          name='b_conv2')

    h_conv2 = tf.nn.relu(
    tf.nn.conv2d(h_pool1, W_conv2,
                 strides=[1, 1, 1, 1],
                 padding='VALID')
    + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',)


    # Hidden Layer 1
    # Move everything into depth so we can perform a single matrix multiply.
    pool_shape = h_pool2.get_shape().as_list()
    reshape = tf.reshape(
        h_pool2,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    shape = [pool_shape[1] * pool_shape[2] * pool_shape[3], 120]
    W_hidden1 = _variable_with_weight_decay(shape=shape,
                                        stddev=0.01,
                                        wd=5e-4,
                                        name='W_hidden1')
    b_hidden1 = tf.Variable(tf.constant(0.0, shape=[120]),
                        dtype=tf.float32,
                        name='b_hidden1')
    h_hidden1 = tf.nn.relu(tf.matmul(reshape, W_hidden1) + b_hidden1)
    # Dropout Layer 1
    h_hidden1_drop = tf.nn.dropout(h_hidden1, prob, name='drop1')

    # Hidden Layer 2
    shape = [120, 84]
    W_hidden2 = _variable_with_weight_decay(shape=shape,
                                        stddev=0.01,
                                        wd=5e-4,
                                        name='W_hidden2')
    b_hidden2 = tf.Variable(tf.constant(0.0, shape=[84]),
                        dtype=tf.float32,
                        name='b_hidden2')
    h_hidden2 = tf.nn.relu(tf.matmul(h_hidden1_drop, W_hidden2) + b_hidden2)
    # Dropout Layer 2
    h_hidden2_drop = tf.nn.dropout(h_hidden2, prob, name='drop2')

    # Hidden Layer 3
    shape = [84, 5]
    W_hidden3 = _variable_with_weight_decay(shape=shape,
                                            stddev=0.01,
                                            wd=5e-4,
                                            name='W_hidden3')
    b_hidden3 = tf.Variable(tf.constant(0.0, shape=[5]),
                            dtype=tf.float32,
                            name='b_hidden3')
    h_hidden3 = tf.add(tf.matmul(h_hidden2_drop, W_hidden3), b_hidden3, name='h_hidden3')

    argmax = tf.nn.softmax(h_hidden3, name='argmax')


    return h_hidden3

def total_loss(logits, labels):
    entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits))
    tf.add_to_collection('losses', entropy)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(loss, learning_rate, batch):
    # Train the net with gradient descent.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    return optimizer.minimize(loss, global_step=batch)

# def label_prediction(predictions):
#     return np.argmax(tf.nn.softmax(predictions), 1)


