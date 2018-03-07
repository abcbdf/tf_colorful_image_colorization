# -*- coding: utf-8 -*-

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
nilboy_weight_decay = 0.001
class Model:
    def __init__(self,
                 learning_rate=0.0001, #important
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, FLAGS.picture_size, FLAGS.picture_size])
        self.y_ = tf.placeholder(tf.float32, [None, FLAGS.picture_size, FLAGS.picture_size, 2])
        self.encode = tf.placeholder(tf.float32, [None, FLAGS.picture_size / 4, FLAGS.picture_size / 4, 256])
        self.keep_prob = tf.placeholder(tf.float32)
        self.weight_decay = 0.001
        self.train = tf.placeholder(tf.bool, [])

        self.x_reshape = tf.reshape(self.x_, [-1, FLAGS.picture_size, FLAGS.picture_size, 1])

        self.graph_upsample_enhanced()

        #logits = self.graph_upsample()

        #self.Euclidean_loss(logits)

        #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)) ### This is wrong. In picture there shouldn't be softmax

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         dtype=tf.float32)
        #self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate, beta2 = 0.99).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)
        


        #self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)



    def graph_basic(self):
        # input -- Conv -- sigmoid -- Pool -- Conv -- sigmoid -- Pool -- Linear -- sigmoid -- loss
        conv1_weight = weight_variable([3, 3, 1, 32])
        conv1_bias = bias_variable([32])
        conv2_weight = weight_variable([3, 3, 32, 64])
        conv2_bias = bias_variable([64])
        l1_weight = weight_variable([16 * 16 * 64, 64 * 64 * 2])
        l1_bias = bias_variable([64 * 64 * 2])
        Conv1 = tf.nn.conv2d(self.x_reshape, conv1_weight, strides = [1, 1, 1, 1], padding = 'SAME') + conv1_bias
        sigmoid1 = tf.nn.sigmoid(Conv1)
        MaxPool1 = tf.nn.max_pool(sigmoid1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        Conv2 = tf.nn.conv2d(MaxPool1, conv2_weight, strides = [1, 1, 1, 1], padding = 'SAME') + conv2_bias
        sigmoid2 = tf.nn.sigmoid(Conv2)
        MaxPool2 = tf.nn.max_pool(sigmoid2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        MaxPool2_flat = tf.reshape(MaxPool2, [-1, 16 * 16 * 64])
        logits = tf.nn.sigmoid(tf.matmul(MaxPool2_flat, l1_weight) + l1_bias)
        #logits = tf.matmul(MaxPool2_flat, l1_weight) + l1_bias
        return logits

    def graph_MultiCNN(self):
        # input -- Conv -- sigmoid -- Pool -- Conv -- sigmoid -- Pool -- Conv -- Conv -- Conv -- Conv -- Conv -- loss
        conv1_weight = weight_variable([3, 3, 1, 32])
        conv1_bias = bias_variable([32])
        conv2_weight = weight_variable([3, 3, 32, 64])
        conv2_bias = bias_variable([64])
        conv3_weight = weight_variable([3, 3, 64, 64])
        conv3_bias = bias_variable([64])
        conv4_weight = weight_variable([3, 3, 64, 64])
        conv4_bias = bias_variable([64])
        conv5_weight = weight_variable([3, 3, 64, 64])
        conv5_bias = bias_variable([64])
        conv6_weight = weight_variable([3, 3, 64, 64])
        conv6_bias = bias_variable([64])
        conv7_weight = weight_variable([3, 3, 64, 32])
        conv7_bias = bias_variable([32])
        Conv1 = tf.nn.conv2d(self.x_reshape, conv1_weight, strides = [1, 1, 1, 1], padding = 'SAME') + conv1_bias
        sigmoid1 = tf.nn.sigmoid(Conv1)
        MaxPool1 = tf.nn.max_pool(sigmoid1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        Conv2 = tf.nn.conv2d(MaxPool1, conv2_weight, strides = [1, 1, 1, 1], padding = 'SAME') + conv2_bias
        sigmoid2 = tf.nn.sigmoid(Conv2)
        MaxPool2 = tf.nn.max_pool(sigmoid2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        Conv3 = tf.nn.conv2d(MaxPool2, conv3_weight, strides = [1, 1, 1, 1], padding = 'SAME') + conv3_bias
        Conv4 = tf.nn.conv2d(Conv3, conv4_weight, strides = [1, 1, 1, 1], padding = 'SAME') + conv4_bias
        Conv5 = tf.nn.conv2d(Conv4, conv5_weight, strides = [1, 1, 1, 1], padding = 'SAME') + conv5_bias
        Conv6 = tf.nn.conv2d(Conv5, conv6_weight, strides = [1, 1, 1, 1], padding = 'SAME') + conv6_bias
        Conv7 = tf.nn.conv2d(Conv6, conv7_weight, strides = [1, 1, 1, 1], padding = 'SAME') + conv7_bias
        logits = tf.reshape(Conv7, [-1, 64 * 64 * 2])
        return logits        

    def graph_upsample(self):
        Conv1_1 = tf.layers.conv2d(self.x_reshape, 32, 3, padding = "SAME", activation = tf.sigmoid)
        Conv1_2 = tf.layers.conv2d(Conv1_1, 32, 3, padding = "SAME", activation = tf.sigmoid)
        Conv1_3 = tf.layers.conv2d(Conv1_2, 32, 3, padding = "SAME", activation = tf.sigmoid)
        MaxPool1 = tf.layers.max_pooling2d(Conv1_3, 2, 2, padding = "SAME") 
        Conv2_1 = tf.layers.conv2d(MaxPool1, 64, 3, padding = "SAME", activation = tf.sigmoid)
        Conv2_2 = tf.layers.conv2d(Conv2_1, 64, 3, padding = "SAME", activation = tf.sigmoid)
        Conv2_3 = tf.layers.conv2d(Conv2_2, 64, 3, padding = "SAME", activation = tf.sigmoid)
        MaxPool2 = tf.layers.max_pooling2d(Conv2_3, 2, 2, padding = "SAME")
        Conv3 = tf.layers.conv2d(MaxPool2, 128, 3, padding = "SAME", activation = tf.sigmoid)
        MaxPool3 = tf.layers.max_pooling2d(Conv3, 2, 2, padding = "SAME")
        TransConv1 = tf.layers.conv2d_transpose(MaxPool3, 32, 3, strides = (2, 2), padding = "SAME", activation = tf.sigmoid)
        TransConv2 = tf.layers.conv2d_transpose(TransConv1, 8, 3, strides = (2, 2), padding = "SAME", activation = tf.sigmoid)
        TransConv3 = tf.layers.conv2d_transpose(TransConv2, 2, 3, strides = (2, 2), padding = "SAME", activation = tf.sigmoid)
        logits = tf.reshape(TransConv2, [-1, FLAGS.picture_size * FLAGS.picture_size * 2])
        return logits

    def graph_upsample_enhanced(self):
        #conv1
        conv_num = 1

        temp_conv = conv2d('conv' + str(conv_num), self.x_reshape, [3, 3, 1, 64], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 64, 64], stride=2, wd=self.weight_decay)
        conv_num += 1

        #self.nilboy = temp_conv

        temp_conv = batch_norm('bn_1', temp_conv,train=self.train)
        #conv2
        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 64, 128], stride=1, wd=self.weight_decay)
        conv_num += 1
        
        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 128], stride=2, wd=self.weight_decay)
        conv_num += 1

        temp_conv = batch_norm('bn_2', temp_conv,train=self.train)
        #conv3
        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 256], stride=1, wd=self.weight_decay)
        conv_num += 1
        
        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
        conv_num += 1    

        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 256], stride=2, wd=self.weight_decay)
        conv_num += 1

        temp_conv = batch_norm('bn_3', temp_conv, train=self.train)
        #conv4
        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1, wd=self.weight_decay)
        conv_num += 1
        
        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
        conv_num += 1

        
        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = batch_norm('bn_4', temp_conv,train=self.train)

        #conv5
        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
        conv_num += 1    



        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
        conv_num += 1

        temp_conv = batch_norm('bn_5', temp_conv,train=self.train)
        #conv6
        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
        conv_num += 1    

        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
        conv_num += 1    

        temp_conv = batch_norm('bn_6', temp_conv,train=self.train)    
        #conv7
        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = batch_norm('bn_7', temp_conv,train=self.train)
        #conv8
        temp_conv = deconv2d('conv' + str(conv_num), temp_conv, [4, 4, 512, 256], stride=2, wd=self.weight_decay)
        conv_num += 1    

        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
        conv_num += 1

        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
        conv_num += 1

        #Unary prediction
        temp_conv = conv2d('conv' + str(conv_num), temp_conv, [1, 1, 256, 256], stride=1, relu=False, wd=self.weight_decay)
        conv_num += 1        
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = temp_conv, labels = self.encode)) / (FLAGS.batch_size)
        self.output = tf.nn.softmax(temp_conv)

    def Euclidean_loss(self, logits):
        y = tf.reshape(self.y_, [-1, FLAGS.picture_size * FLAGS.picture_size * 2])
        self.output = tf.reshape(logits, [-1, FLAGS.picture_size, FLAGS.picture_size, 2])
        self.loss = tf.reduce_mean(tf.square(logits - y))





def _variable(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the Variable
    shape: list of ints
    initializer: initializer of Variable

  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  tf.summary.histogram(name, var)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with truncated normal distribution
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable 
    shape: list of ints
    stddev: standard devision of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight 
    decay is not added for this Variable.

 Returns:
    Variable Tensor 
  """
  var = _variable(name, shape,
    tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def conv2d(scope, input, kernel_size, stride=1, dilation=1, relu=True, wd=nilboy_weight_decay):
  name = scope
  with tf.variable_scope(scope) as scope:
    kernel = _variable_with_weight_decay('weights', 
                                    shape=kernel_size,
                                    stddev=5e-2,
                                    wd=wd)
    if dilation == 1:
      conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME')
    else:
      conv = tf.nn.atrous_conv2d(input, kernel, dilation, padding='SAME')
    biases = _variable('biases', kernel_size[3:], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    if relu:
      conv1 = tf.nn.relu(bias)
    else:
      conv1 = bias
  return conv1

def deconv2d(scope, input, kernel_size, stride=1, wd=nilboy_weight_decay):
  """convolutional layer

  Args:
    input: 4-D tensor [batch_size, height, width, depth]
    scope: variable_scope name 
    kernel_size: [k_height, k_width, in_channel, out_channel]
    stride: int32
  Return:
    output: 4-D tensor [batch_size, height * stride, width * stride, out_channel]
  """
  pad_size = int((kernel_size[0] - 1)/2)
  #input = tf.pad(input, [[0,0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "CONSTANT")
  batch_size, height, width, in_channel = input.shape
  out_channel = kernel_size[3] 
  kernel_size = [kernel_size[0], kernel_size[1], kernel_size[3], kernel_size[2]]
  output_shape = [tf.shape(input)[0], height * stride, width * stride, out_channel]
  with tf.variable_scope(scope) as scope:
    kernel = _variable_with_weight_decay('weights', 
                                    shape=kernel_size,
                                    stddev=5e-2,
                                    wd=wd)
    deconv = tf.nn.conv2d_transpose(input, kernel, output_shape, [1, stride, stride, 1], padding='SAME')

    biases = _variable('biases', (out_channel), tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(deconv, biases)
    deconv1 = tf.nn.relu(bias)

  return deconv1

def batch_norm(scope, x, train=True, reuse=False):
  return tf.contrib.layers.batch_norm(x, center=True, scale=True, updates_collections=None, is_training=train, trainable=True, scope=scope)