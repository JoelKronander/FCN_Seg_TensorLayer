import tensorflow as tf
import tensorlayer as tl
import numpy as np

def conv_layers(net_in):
    with tf.name_scope('preprocess') as scope:
        """
        Notice that we include a preprocessing layer that takes the RGB image
        with pixels values in the range of 0-255 and subtracts the mean image
        values (calculated over the entire ImageNet training set).
        """
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - mean
    """ conv1 """
    network = tl.layers.Conv2dLayer(net_in,
                    act = tf.nn.relu,
                    shape = [3, 3, 3, 64],  # 64 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv1_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 64, 64],  # 64 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv1_2')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool1')
    """ conv2 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv2_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv2_2')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool2')
    """ conv3 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool3')
    """ conv4 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool4')
    """ conv5 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool5')
    return network


def fc_layers(net):
    network = tl.layers.FlattenLayer(net, name='flatten')
    network = tl.layers.DenseLayer(network, n_units=4096,
                        act = tf.nn.relu,
                        name = 'fc1_relu')
    network = tl.layers.DenseLayer(network, n_units=4096,
                        act = tf.nn.relu,
                        name = 'fc2_relu')
    network = tl.layers.DenseLayer(network, n_units=1000,
                        act = tf.identity,
                        name = 'fc3_relu')
    return network

def load_pretrained_conv_weights(net_cnn, weight_file, session):

    # you need to download the model from http://www.cs.toronto.edu/~frossard/post/vgg16/
    npz = np.load(weight_file)

    params = []
    for key, val in sorted( npz.items() ):
        if(key[:4] == "conv"):
            print("  Loading %s" % (key))
            print("  with size %s " % str(val.shape))
            params.append(val)

            tl.files.assign_params(session, params, net_cnn)

    return net_cnn
