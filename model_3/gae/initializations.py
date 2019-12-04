import tensorflow as tf
import numpy as np


def get_weight_init_by_name(init_name, input_dim, output_dim, std=5e-2, name=""):
    if init_name == 'glorot':
        return weight_variable_glorot(input_dim, output_dim, name=name)
    elif init_name == 'truncated_normal':
        return weight_variable_truncated_normal(input_dim, output_dim, std=5e-2, name=name)
    else:
        print('no such way!!!')

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def weight_variable_truncated_normal(input_dim, output_dim, std=5e-2, name=""):
    """
    initialization.
    """
    variable = tf.get_variable(name, [input_dim, output_dim], dtype=tf.float32,
                                      initializer=tf.initializers.truncated_normal(mean=0.0, stddev=std))
    return variable