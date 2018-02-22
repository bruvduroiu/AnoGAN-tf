import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

REAL_MEAN = 4.
REAL_STD  = 1.25

# Helper functions
def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def sample_z(num):
    return np.random.uniform(-1.0, 1.0, size=(num, 100))


class AnoGAN:

    def __init__(self, name='AnoGAN', training=True, D_lr=2e-4, G_lr=2e-4, in_shape=[100,], z_dim=100):
        self.name = name
        self.shape = in_shape
        self.bn_params = {
            'decay': 0.99,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': training
        }
        self.beta1 = 0.5
        self.z_dim = z_dim
        self.D_lr = D_lr
        self.G_lr = G_lr
        self.args = vars(self).copy()

        if training == True:
            self._build_train_graph()
        else:
            self._build_gen_graph()

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            D_real_prob, D_real_logits = self._discriminator(X)
            D_fake_prob, D_fake_logits = self._discriminator(G, reuse=True)

            G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_logits), logits=D_fake_logits)
            D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_logits), logits=D_real_logits)
            D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_logits), logits=D_fake_logits)
            D_loss = D_loss_real + D_loss_fake

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/G/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/G/')

            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1).\
                    minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1).\
                    minimize(G_loss, var_list=G_vars)
            

            # Pre-step summaries
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('D_loss/real', D_loss_real),
                tf.summary.scalar('D_loss/fake', D_loss_fake)
            ])

            tf.summary.histogram('fake_sample', G)
            tf.summary.histogram('real_probs', D_real_prob)
            tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            # Accessible points∂
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 100, activation_fn=tf.nn.relu,
                                       normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params)
            
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.fully_connected(net, 200)
                net = slim.fully_connected(net, 100, activation_fn=None, normalizer_fn=None)

                return net

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            net = X

            with slim.arg_scope([slim.fully_connected], activation_fn=lrelu, normalizer_fn=slim.batch_norm,
                                normalizer_params=self.bn_params):
                net = slim.fully_connected(net, 100, normalizer_fn=None)
                net = slim.fully_connected(net, 50)
                
                logits = slim.fully_connected(net, 1, activation_fn=None)
                prob = tf.sigmoid(logits)

                return prob, logits

    def train(self, batch_size=100, epochs=30000, print_interval=500):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i in range(epochs):
                z_in = sample_z(num=batch_size)
                real_in = np.random.normal(loc=REAL_MEAN, scale=REAL_STD, size=(batch_size, 100))

                _, summary = sess.run([self.D_train_op, self.summary_op], feed_dict={self.X: real_in, self.z: z_in})
                _, global_step = sess.run([self.G_train_op, self.global_step], feed_dict={self.z: z_in})

            z_ = sample_z(num=1)

            fake_samples = sess.run(self.fake_sample, feed_dict={self.z: z_})

            return fake_samples

    def generate_sample(self, num_sample=1):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            z_ = sample_z(num=1)

            fake_samples = sess.run(self.fake_sample, feed_dict={self.z: z_})

            return fake_samples
