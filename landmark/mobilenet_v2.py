import tensorflow as tf
from ops import *
from config import *


def mobilenetv2_auto(iterator=None, is_train=True, reuse=False):
    if iterator:
        img, landmarks = iterator.get_next()
    else:
        img = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        landmarks = tf.placeholder(dtype=tf.int8, shape=[None, 10])
    exp = 6  # expansion ratio
    #img, landmarks = batch.get_next()
    with tf.variable_scope('mobilenetv2'):
        net = conv2d_block(img, 16, 3, 2, is_train, name='conv1_1')  # size/2,%

        net = res_block(net, exp, 24, 2, is_train, name='res2_1')  # %
        net = res_block(net, exp, 24, 1, is_train, name='res3_1')  # size/4
        net = res_block(net, exp, 32, 2, is_train, name='res3_2')
        net = res_block(net, exp, 32, 1, is_train, name='res4_1')  # size/8

        net = res_block(net, exp, 64, 2, is_train, name='res4_2')
        net = res_block(net, exp, 64, 1, is_train, name='res4_3')

        net = res_block(net, exp, 128, 2, is_train, name='res5_1')
        net = res_block(net, exp, 128, 1, is_train, name='res5_2')

        net = fc1(net, 256, activation=tf.nn.relu, name='fc1')
        net = fc2(net, 10, activation=tf.nn.relu, name='fc2')

        return net, landmarks, img

def fc(input, out_dim, activation, name = None):
    input_shape = tf.get_shape(input)
    with tf.variable_scope(name):
        w = tf.get_variable(name='w1', shape=input_shape, initializer=tf.random_normal_initializer(mean=0, stddev=1))
        b = tf.get_variable(name='b1', shape=[out_dim], initializer=tf.constant_initializer(0.1))
        net = tf.matmul(w, input) + b
    return net


def mobilenetv2(iterator = None, is_train=True, reuse=False):
    if iterator:
        img, landmarks = iterator.get_next()
    else:
        img = tf.placeholder(dtype=tf.float32, shape=[None, args.width, args.height, 3])
        landmarks = tf.placeholder(dtype=tf.int8, shape=[None, 10])
    exp = 6  # expansion ratio
    #img, landmarks = batch.get_next()
    with tf.variable_scope('mobilenetv2'):
        net = conv2d_block(img, 16, 3, 2, is_train, name='conv1_1')  # size/2,%

        net = res_block(net, exp, 24, 2, is_train, name='res2_1')  # %
        net = res_block(net, exp, 24, 1, is_train, name='res3_1')  # size/4
        net = res_block(net, exp, 32, 2, is_train, name='res3_2')
        net = res_block(net, exp, 32, 1, is_train, name='res4_1')  # size/8

        net = res_block(net, exp, 64, 2, is_train, name='res4_2')
        net = res_block(net, exp, 64, 1, is_train, name='res4_3')

        net = res_block(net, exp, 128, 2, is_train, name='res5_1')
        net = res_block(net, exp, 128, 1, is_train, name='res5_2')

        net = res_block(net, exp, 128, 2, is_train, name='res6_1')
        net = res_block(net, exp, 128, 1, is_train, name='res6_2')

        net = res_block(net, exp, 128, 2, is_train, name='res7_1')
        net = res_block(net, exp, 128, 1, is_train, name='res7_2')

        net = res_block(net, exp, 128, 2, is_train, name='res8_1')
        net = res_block(net, exp, 128, 1, is_train, name='res8_2')

        net = fc1(net, 256, activation=tf.nn.relu, name='fc1')
        net = fc2(net, 10, activation=tf.nn.relu, name='fc2')

        '''
        net = pwise_block(net, 256, is_train, name='conv5_1')
        net = pwise_block(net, 10, is_train, name= 'conv6_2')
        '''
        #net = global_avg(net)
        #logits = flatten(conv_1x1(net, num_classes, name='logits'))
        #pred = tf.nn.softmax(logits, name='prob')
        #euclidean_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(net, labels)), axis=1)/2, axis=0)

        return net, landmarks, img