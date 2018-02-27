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
    REAL_MEAN = np.array([3.,4.])
    REAL_COV = np.array([[1.,0.],[0.,1.]])
    return np.random.multivariate_normal(REAL_MEAN, REAL_COV, size=(num,100))

def sample_test_data(num):
    test_data = sample_training_data(num)
    test_data[0][1] = [10,10]
    return test_data


def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongisde the specified axis.
    
    # Arguments
        x: A tensor or variable
        axis: An integer, the axis to compute the variance
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable
        axis: An integer, the axis to compute the variance
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.
    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


class AnoGAN:
    def __init__(self, name='AnoGAN', training=True, D_lr=2e-4, G_lr=2e-4, in_shape=[100,2], z_dim=100):
        self.name = name
        self.shape = in_shape
        self.beta1 = 0.5
        self.z_dim = z_dim
        self.D_lr = D_lr
        self.G_lr = D_lr
        self.args = vars(self).copy()
        self.sess = tf.Session()

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

            G_mean = tf.reduce_mean(G)
            G_std = reduce_std(G)

            G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_logits), logits=D_fake_logits)
            D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_logits), logits=D_real_logits)
            D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_logits), logits=D_fake_logits)
            D_loss = D_loss_real + D_loss_fake

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/G/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/G/')

            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1).\
                    minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1).\
                    minimize(G_loss, var_list=G_vars)

            
            # Summary: losses
            tf.summary.scalar('G_loss', G_loss)
            tf.summary.scalar('D_loss', D_loss)
            tf.summary.scalar('D_loss/real', D_loss_real)
            tf.summary.scalar('D_loss/fake', D_loss_fake)
            tf.summary.scalar('G_mean', G_mean)
            tf.summary.scalar('G_std', G_std)

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

    def _sampler(self, z, y=None, batch_size=1):
        with tf.variable_scope('G', reuse=True) as scope:
            tf.get_variable_scope().reuse_variables()

            net = z
            net = slim.fully_connected(net, 200, activation_fn=tf.nn.relu)
            net = slim.fully_connected(net, 400, activation_fn=tf.nn.relu)
            net = slim.fully_connected(net, 200, activation_fn=None)
            net = tf.reshape(net, [-1, 100, 2])

            return net

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            net = z
            net = slim.fully_connected(net, 200, activation_fn=tf.nn.relu)
            net = slim.fully_connected(net, 400, activation_fn=tf.nn.relu)
            net = slim.fully_connected(net, 200, activation_fn=None)
            net = tf.reshape(net, [-1, 100, 2])

            return net

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            net = X
            net = tf.reshape(net, [tf.shape(net)[0], 100*2])
            with slim.arg_scope([slim.fully_connected], activation_fn=lrelu):
                net = slim.fully_connected(net, 200)
                net = slim.fully_connected(net, 50)

                logits = slim.fully_connected(net, 1, activation_fn=None)
                prob = tf.sigmoid(logits)

                return prob, logits

    def _discriminator_feature_match(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            net = X
            net = tf.reshape(net, [tf.shape(net)[0], 100*2])
            with slim.arg_scope([slim.fully_connected], activation_fn=lrelu):
                net = slim.fully_connected(net, 200)
                net = slim.fully_connected(net, 50)

                return net

    def anomaly_detector(self, lambda_ano=0.1, reuse=True):
        with tf.variable_scope(self.name):
            self.test_inputs = tf.placeholder(tf.float32, shape=[1] + self.shape, name='test_scatter')
            test_inputs = self.test_inputs

            with tf.variable_scope('AnoD'):
                self.ano_z = tf.get_variable('ano_z', shape=[1, self.z_dim], dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(-1, 1, dtype=tf.float32))

            self.ano_G = self._sampler(self.ano_z, None, batch_size=1)

            # Residual loss
            self.res_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(test_inputs, self.ano_G))))

            # Discriminator loss
            d_feature_z = self._discriminator_feature_match(self.ano_G, reuse=True)
            d_feature_in = self._discriminator_feature_match(test_inputs, reuse=True)
            self.dis_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(d_feature_in, d_feature_z))))

            self.anomaly_score = (1 - lambda_ano) * self.res_loss + lambda_ano * self.dis_loss

            ano_z_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.name + '/AnoD/')

            ano_z_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/AnoD/')

            with tf.control_dependencies(ano_z_update_ops):
                ano_z_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1).\
                    minimize(self.anomaly_score, var_list=ano_z_vars)

            self.ano_z_train_op = ano_z_train_op

    def train(self, batch_size=100, epochs=30000, print_interval=500):
        self.sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('./train', self.sess.graph)

        for i in range(epochs):
            z_ = sample_z(num=batch_size)
            real_ = sample_training_data(num=batch_size)

            _, summary = self.sess.run([self.D_train_op, self.all_summary_op], feed_dict={self.X: real_, self.z: z_})
            self.sess.run(self.G_train_op, feed_dict={self.z: z_})

            if i % print_interval == 0:
                train_writer.add_summary(summary, i)
                print('Epoch: {:05d}'.format(i))

        z_ = sample_z(num=1)
        fake_samples = self.sess.run(self.fake_sample, feed_dict={self.z: z_})
        
        train_writer.close()

        return fake_samples

    def train_anomaly_detector(self, epochs=3000, print_interval=100):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.ano_z.initializer)
        test_data = sample_test_data(num=1)

        for epoch in range(epochs):
            _, ano_score, res_loss = self.sess.run([self.ano_z_train_op, self.anomaly_score, self.res_loss], feed_dict={self.test_inputs: test_data})

            if epoch % print_interval == 0:
                print("Epoch: [{:05d}], anomaly score: {:.8f}, res loss: {:.8f}".format(epoch, ano_score, res_loss))

        samples = self.sess.run(self.ano_G)
        return samples, test_data