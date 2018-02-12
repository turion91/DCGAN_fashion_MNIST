'''
GAN Architecture and basic implementation based on https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py
'''
import os
import argparse
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # load mnist
from tensorflow.contrib.learn.python.learn.datasets import base



# ------------------------- Parser  -----------------------

parser = argparse.ArgumentParser()

# I/O related args
parser.add_argument('--output_path', default='out/', type=str,
                    help='Output path for the generated images.')

parser.add_argument('--input_path', default='mnist/', type=str,
                    help='Input path for the fashion mnist.'
                         'If not available data will be downloaded.')

parser.add_argument('--log_path', default='tensorboard_log/', type=str,
                    help='Log path for tensorboard.')

parser.add_argument('--mnist', default='fashion', type=str,
                    help='Choose to use "fashion" (fashion-mnist)'
                         ' or "mnist" (classic mnist) dataset.')

# hyper-parameters
parser.add_argument('--z_dim', default=100, type=int,
                    help='Output path for the generated images.')

parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size used for training.')

parser.add_argument('--train_steps', default=100000, type=int,
                    help='Number of steps used for training.')

FLAGS = parser.parse_args()

# ---------- Helper functions and variables  -----------------------

# MNIST related constants
# images have shape of 28x28 in gray scale
MNIST_HEIGHT = 28
MNIST_WIDTH = 28
MNIST_DIM = 1  # gray scale

# to keep things simple we'll deal with the images as a
# flat tensor of MNIST_FLAT shape
MNIST_FLAT_DIM = MNIST_HEIGHT * MNIST_WIDTH * MNIST_DIM

os.chdir('/home/turion91/Desktop/Temp_work/Deep Learning/Fashion/GAN out')

def generate_4x4_figure(samples):
  '''Generate a 4x4 figure.'''
  fig = plt.figure(figsize=(4, 4))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

  return fig

def maybe_create_out_path(out_path):
  '''If output path does not exist it will be created.'''
  if not os.path.exists(out_path):
    os.makedirs(out_path)

def save_plot(samples, out_path, train_step):
  '''Generates a plot and saves it.'''
  fig = generate_4x4_figure(samples)
  
  file_name = 'step-{}.png'.format(str(train_step).zfill(3))
  full_path = os.path.join(out_path, file_name)
  
  print ('Saving image:', full_path)
  
  maybe_create_out_path(out_path)
  
  plt.savefig(full_path, bbox_inches='tight')
  plt.close(fig)

def maybe_download(input_path, mnist):
  '''If dataset not available in the input path download it.'''

  if mnist == 'fashion':
  
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    file_names = ['train-images-idx3-ubyte.gz',
                  'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz',
                  't10k-labels-idx1-ubyte.gz']
  
    print ('Maybe will download the dataset, this can take a while')
    for name in file_names:
      base.maybe_download(name, input_path, base_url + name)
  
  elif mnist == 'mnist':
    pass
  else:
    raise ValueError('Invalid dataset use only mnist = ["fashion", "mnist"])')

def weight_variable(shape):
      '''
      Set a function to create the initial weights from a truncated normal distribution
      '''  
        
      initial = tf.truncated_normal(shape, stddev=0.1, seed=42)
      return tf.Variable(initial)
    
def bias_variable(shape):
      '''
      Create the initial biases 
      '''
        
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
    
def conv2d(x, W):
      '''
      set the convolutionary layer with a stride of 1
      the output image will have the same size of the input one
      '''
        
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def max_pool_2x2(x):
      '''
      set the maxpool layer with a filter size of 2*2 and a stride of 2
      the output image will have the same size of the input one
      '''   
        
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

def sample_Z():
  '''Generate random noise for the generator.'''
  return np.random.uniform(-1., 1., size=[FLAGS.batch_size, FLAGS.z_dim])

###############################
# Define the discriminator network
def discriminator(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        # First convolutional and pool layers
        # This finds 32 different 5 x 5 pixel features
        d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # d4 contains unscaled values
        return d4

# Define the generator network
def generator(batch_size, z_dim):
    z = tf.random_normal([batch_size, z_dim], mean=0, stddev=1, name='z')
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5)
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5)
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5)
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g4

z_dimensions = 100
batch_size = 16
z = tf.random_normal([batch_size, z_dimensions], mean=0, stddev=1, name='z')

x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder')
# x_placeholder is for feeding input images to the discriminator

Gz = generator(batch_size, z_dimensions)
# Gz holds the generated images

Dx = discriminator(x_placeholder)
# Dx will hold discriminator prediction probabilities
# for the real MNIST images

Dg = discriminator(Gz, reuse_variables=True)
# Dg will hold discriminator prediction probabilities for generated images

# Define losses
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

D_loss = D_loss_real + D_loss_fake

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]


D_solver = tf.train.AdamOptimizer(0.000001).minimize(D_loss, var_list=d_vars)  # only updates the discriminator vars
G_solver = tf.train.AdamOptimizer(0.00005).minimize(G_loss, var_list=g_vars)  # only updates the generator vars

# -------------- TensorBoard summaries -----------------

summ_D_loss_real = tf.summary.scalar("D_loss_real", D_loss_real)
summ_D_loss_fake = tf.summary.scalar("D_loss_fake", D_loss_fake)
summ_D_loss = tf.summary.scalar("D_loss", D_loss)

summ_D_losses = tf.summary.merge([summ_D_loss_real, summ_D_loss_fake,
                                  summ_D_loss])

summ_G_loss = tf.summary.scalar("G_loss", G_loss)

# -------------- Load the dataset ------------------------

# download mnist if needed
#utils.maybe_download(FLAGS.input_path, FLAGS.mnist)

# import mnist dataset
data = input_data.read_data_sets('/home/adriano/fashion/NN_backup/NetworksTutorials/', one_hot=True)


# -------------- Train models ------------------------

# create session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

## Send summary statistics to TensorBoard
#tf.summary.scalar('Generator_loss', summ_G_loss)
#tf.summary.scalar('Discriminator_loss_real', summ_D_loss_real)
#tf.summary.scalar('Discriminator_loss_fake', summ_D_loss_fake)


# create summary writer
summary_writer = tf.summary.FileWriter(FLAGS.log_path, graph=tf.get_default_graph())
start_time = time.time()
for i in range(FLAGS.train_steps):

  # eventually plot images that are being generated
    if i % 10000 == 0:
        samples = sess.run(Gz, feed_dict={z: sample_Z()})
        save_plot(samples, FLAGS.output_path, i)
    if i == FLAGS.train_steps-1:
        samples = sess.run(Gz, feed_dict={z: sample_Z()})
        save_plot(samples, FLAGS.output_path, i)
#
  # get real data
    X_batch = data.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])

  # train discriminator
    _, D_loss_curr, summ = sess.run([D_solver, D_loss, summ_D_losses],
                                  feed_dict={x_placeholder: X_batch, z: sample_Z()})
    summary_writer.add_summary(summ, i)
  # train generator
    _, G_loss_curr, summ = sess.run([G_solver, G_loss, summ_G_loss], feed_dict={z: sample_Z()})
    summary_writer.add_summary(summ, i)

  # eventually print train losses
    if i % 10000 == 0:
        print('Iter: {}'.format(i))
        print(D_loss_curr)
        print(G_loss_curr)
        print()
end_time = time.time()
delta_t = end_time - start_time
print(timedelta(seconds=int(round(delta_t))))
