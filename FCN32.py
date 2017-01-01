import time
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import scipy.misc
import vgg16_model

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

keep_prob = 0.85
NUM_CLASSES = 32
VGG16_WEIGHTS_PATH = 'vgg16_weights.npz'

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
y_ = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, NUM_CLASSES], name='y_')

net_in = tl.layers.InputLayer(x, name='input_layer')
net_vgg16_conv_layers = vgg16_model.conv_layers(net_in)

#Fully convolutional layers on top of VGG16 conv layers
network = tl.layers.Conv2dLayer(net_vgg16_conv_layers,
                act = tf.nn.relu,
                shape = [7, 7, 512, 4096],
                strides = [1, 1, 1, 1],
                padding='SAME',
                name ='conv6')
network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop1')

network = tl.layers.Conv2dLayer(network,
                act = tf.nn.relu,
                shape = [1, 1, 4096, 4096],
                strides = [1, 1, 1, 1],
                padding='SAME',
                name ='conv7')
network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop2')

network = tl.layers.Conv2dLayer(network,
                act = tf.identity,
                shape = [1, 1, 4096, NUM_CLASSES],
                strides = [1, 1, 1, 1],
                padding='SAME',
                name ='conv8')

#Upsampling to actual image size
network = tl.layers.UpSampling2dLayer(network, (32, 32))
# network = tl.layers.DeConv2dLayer(network,
#                 shape = [5, 5, NUM_CLASSES, NUM_CLASSES],
#                 output_shape = [batch_size, 224, 224, NUM_CLASSES],
#                 strides=[1, 32, 32, 1],
#                 act=tf.identity, name='deconv_1')

y = network.outputs
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_), name = "cost")
#probs = tf.nn.softmax(y)
#y_op = tf.argmax(tf.nn.softmax(y), 1)
#cost = tl.cost.cross_entropy(y, y_)

#correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(y_, tf.float32))
#acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
vgg16_model.load_pretrained_conv_weights(net_vgg16_conv_layers, VGG16_WEIGHTS_PATH, sess)

network.print_params()
network.print_layers()

img1 = scipy.misc.imread('data/laska.png', mode='RGB') # test data in github
img1 = scipy.misc.imresize(img1, (224, 224))

start_time = time.time()
#for testing time dp_dict = tl.utils.dict_to_one( network.all_drop ) # disable noise layers
feed_dict={x: [img1]}
feed_dict.update( network.all_drop )
y = sess.run(y, feed_dict=feed_dict)
print(y.shape)
y_part = y[0];
scipy.misc.toimage(np.squeeze(y_part[:,:,1]), cmin=0, cmax=1).save('test.png')
print("  End time : %.5ss" % (time.time() - start_time))
#preds = (np.argsort(prob)[::-1])[0:5]
#for p in preds:
#    print(class_names[p], prob[p])
