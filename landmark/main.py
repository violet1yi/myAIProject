import tensorflow as tf
from mobilenet_v2 import mobilenetv2
#from config import *
#from utils import *
from get_tf_record import *


import time
#import glob
import os


def load(sess, saver, checkpoint_dir):
    import re
    print("[*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print("[*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print("[*] Failed to find a checkpoint")
        return False, 0


def main():

    # read queue
    glob_pattern = os.path.join(args.dataset_dir, '2w_train.tfrecords')
    #dev_record_file = os.path.join(args.dataset_dir, '5k_val.tfrecords')
    print('path', glob_pattern)
    #-------------------------------------------------------
    parser = get_record_parser()
    train_dataset = get_batch_dataset(glob_pattern, parser)
    #eval_dataset = get_dataset(dev_record_file, parser)
    handle = tf.placeholder(dtype=tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    #eval_iterator = eval_dataset.make_one_shot_iterator()

    #--------------------------------------------------------------
    net_out, landmarks = mobilenetv2(iterator, is_train=args.is_train)
    # loss
    #euclidean_loss = tf.log(tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(tf.cast(landmarks, tf.float32), net_out)), axis=3)/2))
    euclidean_loss = tf.log(tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(tf.cast(landmarks, tf.float32), net_out)), axis=1) / 2))
    #l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # learning rate decay
    base_lr = tf.constant(args.learning_rate)
    lr_decay_step = args.num_samples // args.batch_size * 2  # every epoch
    global_step = tf.placeholder(dtype=tf.float32, shape=())
    lr = tf.train.exponential_decay(base_lr, global_step=global_step, decay_steps=lr_decay_step, decay_rate=args.lr_decay)

    # optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=args.beta1).minimize(euclidean_loss)

    # summary
    tf.summary.scalar('euclidean_loss', euclidean_loss)
    tf.summary.scalar('learning_rate', lr)
    summary_op = tf.summary.merge_all()

    step=0
    '''
    if not args.renew:
        print('[*] Try to load trained model...')
        could_load, step = load(sess, saver, args.checkpoint_dir)
    '''
    max_steps = int(args.num_samples / args.batch_size * args.epoch)

    print('START TRAINING...')
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logs_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        #eval_handle = sess.run(eval_iterator.string_handle())
        coord = tf.train.Coordinator()

        for _step in range(step+1, max_steps+1):
            start_time = time.time()

            _, _lr = sess.run([train_op, lr], feed_dict={global_step: _step, handle: train_handle})
            #  print logs and write summary
            if _step % 50 == 0:
                '''
                #_summ, _loss = sess.run([summary_op, euclidean_loss],  feed_dict={global_step: _step, handle: train_handle})
                #writer.add_summary(_summ, _step)
                '''
                _loss = sess.run([euclidean_loss], feed_dict={global_step: _step, handle: train_handle})
                print('global_step:{0}, time:{1:.3f}, lr:{2:.8f}, loss:'.format(_step, time.time() - start_time, _lr, _loss))
                _net_out, _landmarks = sess.run([net_out, landmarks], feed_dict={global_step: _step, handle: train_handle})
                print('net_out', _net_out)
                print('landmarks', _landmarks)
                #print('loss:'.format(_loss))
            # save model
            if _step % 50 == 0:
                save_path = saver.save(sess, os.path.join(args.checkpoint_dir, args.model_name), global_step=_step)
                print('Current model saved in ' + save_path)

            #----------------------------------------------------------------------------------
            """
            if _step % 200 == 0:
                losses = []
                for _ in tqdm(range(1, args.eval_batch_size)):
                    _loss = sess.run([euclidean_loss], feed_dict={global_step: _step, handle: eval_handle})
                    losses.append(_loss)
                t_loss = tf.reduce_mean(losses)
                print('global_step:{0}, time:{1:.3f}, lr:{2:.8f}, loss:'.format(_step, time.time() - start_time, _lr, t_loss))
                '''
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="{}/loss".format('eval'), simple_value=metrics["loss"]), ])
                for s in loss_sum:
                    writer.add_summary(s, global_step)
                '''
            """
            #-----------------------------------------------------------------------------

        tf.train.write_graph(sess.graph_def, args.checkpoint_dir, args.model_name + '.pb')
        save_path = saver.save(sess, os.path.join(args.checkpoint_dir, args.model_name), global_step=max_steps)
        print('Final model saved in ' + save_path)
    print('FINISHED TRAINING.')

if __name__=='__main__':
    main()