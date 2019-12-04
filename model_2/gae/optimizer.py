import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


# class OptimizerAE(object):
#     def __init__(self, preds, labels, pos_weight, norm):
#         preds_sub = preds
#         labels_sub = labels
#
#         self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
#
#         self.opt_op = self.optimizer.minimize(self.cost)
#         self.grads_vars = self.optimizer.compute_gradients(self.cost)
#
#         self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
#                                            tf.cast(labels_sub, tf.int32))
#         self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


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
