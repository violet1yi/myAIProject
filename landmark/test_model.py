import tensorflow as tf
import cv2
import sys
import numpy as np
from config import *
from mobilenet_v2 import mobilenetv2

def draw_points(lands, img):
    for i in range(0, len(lands), 2):
        point = (lands[i], lands[i+1])
        cv2.circle(img, point, 1, (255, 0, 0))
        #cv2.imshow('img', img)
        cv2.imwrite('./results/result_3.jpg', img)
    print('draw points done')

def test(img_input):
    img_exp = img_input[np.newaxis, :]
    print(img_exp.shape)
    '''
    dataset = tf.data.Dataset().batch(1)
    handle = tf.placeholder(dtype=tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
    '''
    landmarks, _ , img = mobilenetv2(None, is_train=args.is_train)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(check_point_dir))
        landmarks_out = sess.run([landmarks], feed_dict={img: img_exp})
        print('landmarks:', landmarks_out)
    return landmarks


if __name__ == "__main__":
    image_path = './test_images/face1.png'
    check_point_dir = './CheckPoints/checkpoints_no_align_v3'

    img = cv2.imread(image_path)
    img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_AREA)
    landmarks_out = test(img)
    lands_int = [int(i) for i in landmarks_out]
    draw_points(lands_int, img)
    print("model output change to int, landmarks: %s", lands_int)








