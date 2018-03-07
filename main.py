from __future__ import division
import tensorflow as tf
import numpy as np
import os
import time
from model import Model
from load_image import *
import utils
import cv2
import logging

tf.app.flags.DEFINE_integer("batch_size", 20, "batch size for training")
tf.app.flags.DEFINE_integer("test_size", 30, "number of picture to output")
tf.app.flags.DEFINE_integer("num_epochs", 30, "number of epochs")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "drop out rate")
tf.app.flags.DEFINE_boolean("is_train", True, "False to inference")
tf.app.flags.DEFINE_string("data_dir", "./MNIST_data", "data dir")
tf.app.flags.DEFINE_string("train_dir", "./train", "training dir")
tf.app.flags.DEFINE_integer("inference_version", 0, "the param version for inference")
tf.app.flags.DEFINE_integer("picture_size", 256, "picture will be resize to size * size")
FLAGS = tf.app.flags.FLAGS
np.set_printoptions(threshold=np.inf) 


def shuffle(X, shuffle_parts = 1): #shuffle the sequence of data
    chunk_size = len(X) / shuffle_parts
    chunk_size = int(chunk_size)
    shuffled_range = range(chunk_size)

    X_buffer = np.copy(X[0:chunk_size])

    for k in range(shuffle_parts):
        np.random.shuffle(list(shuffled_range))
        for i in range(chunk_size):
            X_buffer[i] = X[k * chunk_size + shuffled_range[i]]

        X[k * chunk_size:(k + 1) * chunk_size] = X_buffer

    return X

def train_epoch(model, sess, train_list, merged):
    loss = 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    #print(len(train_list))
    while st < len(train_list) and ed <= len(train_list):
      start_time = time.time()
      train_batch = train_list[st : ed]
      rg_list = []
      black_list = []
      encode_list = []
      for image_dir in train_batch:
        img, img_black = load_image(image_dir, FLAGS.picture_size)
        img_encode = naive_encode(img[:, :, 1:], FLAGS.picture_size)
        rg_list.append(img[:, :, 1:])
        black_list.append(img_black)
        encode_list.append(img_encode)
      feed = {model.x_: black_list, model.y_: rg_list, model.keep_prob: FLAGS.keep_prob, model.encode: encode_list, model.train: True}
      load_time = time.time()
      logging.debug("load:" + str(load_time - start_time))
      summary, loss_, _ = sess.run([merged, model.loss, model.train_op], feed)
      logging.debug("run:" + str(time.time() - load_time))
      loss += loss_
      st, ed = ed, ed+FLAGS.batch_size
      times += 1
    loss /= times
    return loss, summary

def valid_epoch(model, sess, valid_list, merged):
    loss = 0.0
    st, ed, times = 0, 1, 0
    index = 0
    while st < len(valid_list) and ed <= len(valid_list):
       valid_batch = valid_list[st : ed]
       rg_list = [] ## this is the list of picture's a and b in the lab color space
       black_list = [] ## this is the list of picture's l in the lab color space
       encode_list = []
       for image_dir in valid_batch:
       	img, img_black = load_image(image_dir, FLAGS.picture_size)
       	img_encode = naive_encode(img[:, :, 1:], FLAGS.picture_size)
        rg_list.append(img[:, :, 1:])
        black_list.append(img_black)
        encode_list.append(img_encode)
       feed = {model.x_: black_list, model.y_: rg_list, model.keep_prob: FLAGS.keep_prob, model.encode: encode_list, model.train: False}
       loss_, output = sess.run([model.loss, model.output], feed)
       start_time = time.time()
       decode_output = naive_decode(output[0], FLAGS.picture_size)
       decode_time = time.time()
       logging.debug("decode:" + str(decode_time - start_time))
       for_display(black_list[0], rg_list[0], decode_output, "./display/" + str(index))
       logging.debug("displaytime:" + str(time.time() - decode_time))
       index += 1
       loss += loss_
       st, ed = ed, ed+1
       times += 1
    loss /= times
    return loss






os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='./history/' + time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time())) + '_color.log',
                filemode='w')
logging.info('Start:')


with tf.Session() as sess:
  #with tf.device('/gpu:0'):
  if not os.path.exists(FLAGS.train_dir):
    os.mkdir(FLAGS.train_dir)
  if FLAGS.is_train:
    train_list, test_list = get_image_list()
    cnn_model = Model()
    tf.global_variables_initializer().run()
    pre_losses = 1e18 * 3
    best_val_loss = 1e18 * 3
    summary_writer = tf.summary.FileWriter('%s/log' % FLAGS.train_dir, sess.graph)
    merged = tf.summary.merge_all()
    for epoch in range(FLAGS.num_epochs):
      start_time = time.time()
      train_loss, summary = train_epoch(cnn_model, sess, train_list, merged)
      #summary_writer.add_summary(summary, epoch)
      train_time = time.time()
      train_list = shuffle(train_list)
      shuffle_time = time.time()
      val_loss = valid_epoch(cnn_model, sess, test_list, merged)
      epoch_time = time.time() - start_time
      logging.info("Epoch " + str(epoch + 1) + " of " + str(FLAGS.num_epochs) + " took " + str(epoch_time) + "s")
      logging.info("  learning rate:                 " + str(cnn_model.learning_rate.eval()))
      logging.info("  training loss:                 " + str(train_loss))
      logging.info("  validation loss:               " + str(val_loss))
      print("Epoch " + str(epoch + 1) + " of " + str(FLAGS.num_epochs) + " took " + str(epoch_time) + "s")
  else:
    pass

logging.info('End')

