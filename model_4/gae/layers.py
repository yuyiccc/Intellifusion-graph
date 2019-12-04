from gae.initializations import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim,
                 output_dim,
                 adj,
                 weight_decay=0.,
                 weight_init='truncated_normal',
                 act=tf.nn.relu,
                 if_bn=True,
                 **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = get_weight_init_by_name(init_name='truncated_normal',
                                                           input_dim=input_dim,
                                                           output_dim=output_dim,
                                                           name=weight_init)
            # self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        weight_decay_loss = tf.multiply(tf.nn.l2_loss(self.vars['weights']), weight_decay, name='weight_loss')
        tf.add_to_collection('weight_decay_losses', weight_decay_loss)
        self.adj = adj
        self.act = act
        self.if_bn = if_bn

    def _call(self, inputs):
        x = inputs
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        if self.if_bn:
            x = tf.layers.batch_normalization(x, training=True)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim,
                 output_dim,
                 adj,
                 weight_decay=0.,
                 weight_init='truncated_normal',
                 if_bn=True,
                 act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = get_weight_init_by_name(init_name=weight_init,
                                                           input_dim=input_dim,
                                                           output_dim=output_dim,
                                                           name="weights")
            # self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")

        weight_decay_loss = tf.multiply(tf.nn.l2_loss(self.vars['weights']), weight_decay, name='weight_loss')
        tf.add_to_collection('weight_decay_losses', weight_decay_loss)
        self.adj = adj
        self.act = act
        self.if_bn = if_bn
        self.issparse = True

    def _call(self, inputs):
        x = inputs
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        if self.if_bn:
            x = tf.layers.batch_normalization(x, training=True)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, num_nodes,
                 popularity_weight,
                 productivity_weight,
                 act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.popularity_weight = popularity_weight
        self.productivity_weight = productivity_weight
        self.num_nodes = num_nodes
        self.act = act

    def _call(self, inputs):
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x, popular, product = self.popularity_softmax_with_weight(x)
        # x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs, popular, product

    # model_1
    def popularity_softmax_with_weight(self, logits):
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)

        po_weights = tf.sigmoid(self.popularity_weight)
        pro_weights = tf.sigmoid(self.productivity_weight)
        x = tf.exp(logits) * po_weights
        out = (x / tf.reduce_sum(x, axis=1, keepdims=True)) * pro_weights
        return out, po_weights, pro_weights