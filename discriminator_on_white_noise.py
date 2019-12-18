import os

import scipy.io as sio
import tensorflow as tf
from matplotlib import pyplot as plt, patches
from tqdm import tqdm

from utils.dataset import *
from utils.dir_gen import *

__NOISE_DIM__ = 2


def discriminator(inputs, variable_scope='discriminator'):
    with tf.variable_scope(variable_scope):
        current = inputs
        current = tf.layers.dense(inputs=current, units=50,
                                  kernel_initializer=tf.initializers.random_normal(stddev=1e-2),
                                  activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='fc1')
        current = tf.layers.dense(inputs=current, units=1,
                                  kernel_initializer=tf.initializers.random_normal(stddev=1e-2),
                                  activation=None, reuse=tf.AUTO_REUSE, name='fc2')

        return current, tf.nn.sigmoid(current)


def generator(inputs, variable_scope='generator'):
    with tf.variable_scope(variable_scope):
        current = inputs
        current = tf.layers.dense(inputs=current, units=10,
                                  kernel_initializer=tf.initializers.random_normal(stddev=1e-1),
                                  activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='fc1')
        current = tf.layers.dense(inputs=current, units=50,
                                  kernel_initializer=tf.initializers.random_normal(stddev=1e-1),
                                  activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='fc2')
        current = tf.layers.dense(inputs=current, units=2,
                                  kernel_initializer=tf.initializers.random_normal(stddev=1e-1),
                                  activation=None, reuse=tf.AUTO_REUSE, name='fc3')
        return current


class Toy:
    def __init__(self):
        self.learning_rate = 1e-4
        self.sigma = 0.5
        self.batch_size = 100

        self.input_noise = tf.placeholder(shape=(None, __NOISE_DIM__), dtype=tf.float32, name='input_noise')
        self.input_data = tf.placeholder(shape=(None, 2), dtype=tf.float32, name='input_data')
        self.generated_data = generator(self.input_noise)
        _, self.logits_real = discriminator(self.input_data)
        _, self.logits_fake = discriminator(self.generated_data)

        self.d_loss = -tf.reduce_mean(tf.log(self.logits_real) + tf.log(1 - self.logits_fake))
        self.g_loss = -tf.reduce_mean(tf.log(self.logits_fake))

        self.d_var = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        self.g_var = [var for var in tf.trainable_variables() if var.name.startswith('generator')]

        self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.d_grads_and_vars = self.d_optimizer.compute_gradients(self.d_loss, var_list=self.d_var)
        self.d_train = self.d_optimizer.apply_gradients(self.d_grads_and_vars)

        self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.g_grads_and_vars = self.g_optimizer.compute_gradients(self.g_loss, var_list=self.g_var)
        self.g_train = self.g_optimizer.apply_gradients(self.g_grads_and_vars)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            save_path = dir_gen('saves')
            d = Dataset(size=3200)
            sio.savemat(os.path.join(save_path, 'data.mat'), {'data': d.data})
            z_visulisation = np.random.normal(size=(self.batch_size, __NOISE_DIM__), scale=self.sigma)
            xv, yv = np.meshgrid(np.linspace(-2, 2, num=100), np.linspace(-2, 2, num=100))
            a = [np.reshape(xv, (-1,)), np.reshape(yv, (-1,))]
            a = np.asarray(a)
            a = np.transpose(a)
            for step in tqdm(range(1000000)):
                for _ in range(5):
                    _, loss_d = sess.run([self.d_train, self.d_loss],
                                         feed_dict={
                                             self.input_noise: np.random.normal(size=(self.batch_size, __NOISE_DIM__),
                                                                                scale=self.sigma),
                                             self.input_data: d.next_batch(self.batch_size)})
                _, loss_g = sess.run([self.g_train, self.g_loss],
                                     feed_dict={
                                         self.input_noise: np.random.normal(size=(self.batch_size, __NOISE_DIM__),
                                                                            scale=self.sigma)})
                # print(step, loss_d, loss_g)
                if step % 100 == 0:
                    generated_data, logits_real = sess.run([self.generated_data, self.logits_real],
                                                           feed_dict={self.input_noise: z_visulisation,
                                                                      self.input_data: a})
                    mat_name = '%d.mat' % step
                    sio.savemat(os.path.join(save_path, mat_name),
                                {'generared': generated_data, 'loss_d': loss_d, 'loss_g': loss_g,
                                 'logits_real': logits_real})

                    # fig, ax = plt.subplots(1)
                    # plt.plot(d.data[:, 0], d.data[:, 1], 'bo', markersize=3)
                    # plt.plot(generated_data[:, 0], generated_data[:, 1], 'rx', markersize=3)
                    # fig_name = '%d.jpg' % (step)
                    # fig_path = os.path.join(save_path, fig_name)
                    # plt.axis('equal')
                    # plt.xlim(-22, 22)
                    # plt.ylim(-22, 22)
                    # rect_g = patches.Rectangle((-20, -15), 1, loss_g / 1.7 * 5, linewidth=2, edgecolor='r',
                    #                            facecolor='r')
                    # rect_d = patches.Rectangle((-15, -15), 1, loss_d / 1.7 * 5, linewidth=2, edgecolor='b',
                    #                            facecolor='b')
                    # # Add the patch to the Axes
                    # ax.add_patch(rect_g)
                    # ax.add_patch(rect_d)
                    # # fig.set_size_inches(6.4, 4.8)
                    # plt.savefig(fig_path)
                    # # plt.show()
                    # plt.close()


if __name__ == '__main__':
    model = Toy()
    model.train()
