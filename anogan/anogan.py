import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

# Helper functions
def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def sample_z(num):
    return np.random.uniform(-1.0, 1.0, size=(num, 100))

def sample_training_data(num):
    REAL_MEAN, REAL_STD = 4., 1.25
    return np.random.normal(loc=REAL_MEAN, scale=REAL_STD, size=(num, 100))


class AnoGAN:
    def __init__(self, name='AnoGAN', training=True, D_lr=2e-4, G_lr=2e-4, in_shape=[100,], z_dim=100):
        self.name = name
        self.shape = in_shape
        self.beta1 = 0.5
        self.z_dim = z_dim
        self.D_lr = D_lr
        self.G_lr = D_lr
        self.args = vars(self).copy()

        if training:
            self._build_train_graph()
        else:
            self._build_gen_graph()

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])

            G = self._generator(z)
            D_real_prob, D_real_logits = self._discriminator(X)
            D_fake_prob, D_fake_logits = self._discriminator(G, reuse=True)

            G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_logits), logits=D_fake_logits)
            D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_logits), logits=D_real_logits)
            D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_logits), logits=D_fake_logits)
            D_loss = D_loss_real + D_loss_fake

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/G/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/G/')

            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.GradientDescentOptimizer(learning_rate=self.D_lr).\
                    minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.GradientDescentOptimizer(learning_rate=self.G_lr).\
                    minimize(G_loss, var_list=G_vars)

            
            # Summary: losses
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('D_loss/real', D_loss_real),
                tf.summary.scalar('D_loss/fake', D_loss_fake)
            ])

            # Summary: samples and stuff
            tf.summary.histogram('fake_sample', G)
            tf.summary.histogram('real_prob', D_real_prob)
            tf.summary.histogram('fake_prob', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G

    def _build_gen_graph(self):
        pass

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            W_in = slim.fully_connected(z, 100, activation_fn=tf.nn.relu)
            W_hid = slim.fully_connected(W_in, 200, activation_fn=tf.nn.relu)
            W_out = slim.fully_connected(W_hid, 100, activation_fn=None)

            return W_out

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):

            with slim.arg_scope([slim.fully_connected], activation_fn=lrelu):
                W_in = slim.fully_connected(X, 100)
                W_hid = slim.fully_connected(W_in, 50)

                logits = slim.fully_connected(W_hid, 1, activation_fn=None)
                prob = tf.sigmoid(logits)

                return prob, logits

    def train(self, batch_size=100, epochs=30000, print_interval=500):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter('./train', sess.graph)

            for i in range(epochs):
                z_ = sample_z(num=batch_size)
                real_ = sample_training_data(num=batch_size)

                _, summary = sess.run([self.D_train_op, self.all_summary_op], feed_dict={self.X: real_, self.z: z_})
                sess.run(self.G_train_op, feed_dict={self.z: z_})

                if i % print_interval == 0:
                    train_writer.add_summary(summary, i)

            z_ = sample_z(num=1)
            fake_samples = sess.run(self.fake_sample, feed_dict={self.z: z_})
            
            train_writer.close()

            return fake_samples