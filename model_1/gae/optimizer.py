import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerVAE(object):
    def __init__(self, preds, labels, num_nodes, z_std, z_mean, lr=1e-3):
        self.preds = preds
        self.labels = tf.cast(labels, tf.float32)
        # in case preds is zeros(which will cause loss equal to inf)
        self.location = tf.where(tf.logical_and(tf.greater(self.labels, 0), tf.greater(preds, 0)))
        self.positive_pred = tf.gather_nd(preds, self.location)

        self.weight_decay_losses = tf.add_n(tf.get_collection('weight_decay_losses'))
        self.cost = -tf.reduce_mean(tf.log(self.positive_pred))
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * z_std - tf.square(z_mean) -
                                                                   tf.square(tf.exp(z_std)), 1))
        self.loss = self.cost + self.weight_decay_losses - self.kl
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # Adam Optimizer

            self.opt_op = self.optimizer.minimize(self.cost)
            self.grads_vars = self.optimizer.compute_gradients(self.cost)


class OptimizerAE(object):
    def __init__(self, preds, labels, lr=1e-3):

        self.preds = preds
        self.labels = tf.cast(labels, tf.float32)
        # self.location = tf.where(tf.greater(self.labels, 0))
        # in case preds is zeros(which will cause loss equal to inf)
        self.location = tf.where(tf.logical_and(tf.greater(self.labels, 0), tf.greater(preds, 0)))
        self.positive_pred = tf.gather_nd(preds, self.location)

        self.weight_decay_losses = tf.add_n(tf.get_collection('weight_decay_losses'))
        self.cost = -tf.reduce_mean(tf.log(self.positive_pred))
        self.loss = self.cost + self.weight_decay_losses
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # Adam Optimizer

            self.opt_op = self.optimizer.minimize(self.cost)
            self.grads_vars = self.optimizer.compute_gradients(self.cost)
