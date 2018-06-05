#encodinf-utf-8
import tensorflow as tf
from PIL import Image
import os
from config import *

#classes = {'train', 'test', 'val'}
cwd = os.getcwd()
data_path = '/media/leve/bbc9553b-1436-4d09-ac0f-db8b13de42131/shareT/dataset/CelebA/'
landmark_img = 'Anno/part_train_val/9w_train_list_landmarks_celeba.txt'
img_path = 'Img/img_celeba.7z/img_celeba'


def create_record():
    writer = tf.python_io.TFRecordWriter("./tfrecords/9w_train.tfrecords")
    landmark_file = os.path.join(data_path, landmark_img)
    print('data_path', data_path)
    print('land mark path', landmark_file)
    count = 0
    with open(landmark_file) as input:
        for line in input:
            if count % 200 == 0:
                print('%s has beed processed'%(count))
            line = line.strip().split(' ')
            img_name = line[0]
            if not is_imageName(img_name):
                continue
            img_landmark = line[1:]
            img_landmarks = []
            for elem in img_landmark:
                if elem != '':
                    img_landmarks.append(elem)


            image_path = os.path.join(data_path, img_path, str(img_name))
            img = Image.open(image_path)
            size = img.size
            width = size[0]
            height = size[1]
            img_landmarks = [int(i) for i in img_landmarks]
            landmarks_resize = img_landmarks_resize(img_landmarks, width, height, args.height)
            img = img.resize((args.height, args.width))
            img_raw = img.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'landmarks': tf.train.Feature(int64_list=tf.train.Int64List(value=landmarks_resize)),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
            count += 1
    writer.close()

def is_imageName(str):
    split = str.split('.')
    if split[-1] != 'jpg':
        return False
    else:
        return True

def img_landmarks_resize(img_landmarks, width, height, s):
    w = width
    h = height
    l = []
    for i in range(len(img_landmarks)):
        if i % 2 == 0:
            new_x = s * img_landmarks[i] // w
            l.append(int(new_x))
        else:
            new_y = s * img_landmarks[i] // h
            l.append(int(new_y))
    return l


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'landmarks': tf.FixedLenFeature([10], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    img_landmarks = features['landmarks']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [args.height, args.width, 3])
    img = tf.cast(img, tf.float32) * (1./255) - 0.5
    img_landmarks = tf.cast(img_landmarks, tf.int32)
    return img, img_landmarks

def createBatch(filename, batchsize):
    images, labels = read_and_decode(filename)
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batchsize

    image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                          batch_size=batchsize,
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue
                                                          )
    '''
    image_batch, label_batch = tf.train.batch([images, labels],
                                                          batch_size=batchsize,
                                                          capacity=capacity)
    '''
    return image_batch, label_batch

def get_batch_dataset(record_file, parser):
    num_threads = tf.constant(2, dtype=tf.int32)

    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(200).repeat()
    dataset=dataset.batch(args.batch_size)
    return dataset

def get_dataset(record_file, parser):
    num_threads = tf.constant(2, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(args.eval_batch_size)
    return dataset

def get_record_parser():
    def parse(example):
        features = tf.parse_single_example(example,
                                           features={
                                               'landmarks': tf.FixedLenFeature([10], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string)
                                           })
        img_landmarks = features['landmarks']
        img = features['img_raw']
        img = tf.decode_raw(img, tf.uint8)
        img = tf.reshape(img, [args.height, args.width, 3])
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        img_landmarks = tf.cast(img_landmarks, tf.int32)
        return img, img_landmarks
    return parse

if __name__ == '__main__':
    create_record()
    #img, labels = createBatch('./tfrecords/train.tfrecords', 30)
    '''
    sess = tf.Session()
    print('image', sess.run(img))
    print('labels', sess.run(labels))
    sess.close()
    '''
    #print('image', img)
    #print('labels', labels)
    #print('done!')




