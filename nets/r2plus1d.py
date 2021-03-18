import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers

add_arg_scope = contrib_framework.add_arg_scope
layers = contrib_layers


def center_initializer():

    def _initializer(shape, dtype=tf.float32, partition_info=None):  # pylint: disable=unused-argument
        """Initializer op."""

        if dtype != tf.float32 and dtype != tf.bfloat16:
            raise ValueError(
                'Input tensor data type has to be tf.float32 or tf.bfloat16.')
        if len(shape) != 5:
            raise ValueError('Input tensor has to be 5-D.')
        if shape[3] != shape[4]:
            raise ValueError('Input and output channel dimensions must be the same.')
        if shape[1] != 1 or shape[2] != 1:
            raise ValueError('Spatial kernel sizes must be 1 (pointwise conv).')
        if shape[0] % 2 == 0:
            raise ValueError('Temporal kernel size has to be odd.')

        center_pos = int(shape[0] / 2)
        init_mat = np.zeros(
            [shape[0], shape[1], shape[2], shape[3], shape[4]], dtype=np.float32)
        for i in range(0, shape[3]):
            init_mat[center_pos, 0, 0, i, i] = 1.0

        init_op = tf.constant(init_mat, dtype=dtype)
        return init_op

    return _initializer


@add_arg_scope
def conv3d_spatiotemporal(inputs,
                          num_outputs,
                          kernel_size,
                          stride=1,
                          padding='SAME',
                          activation_fn=tf.nn.relu,
                          normalizer_fn=None,
                          normalizer_params=None,
                          weights_regularizer=None,
                          separable=False,
                          data_format='NDHWC',
                          scope=''):
    assert len(kernel_size) == 3
    if separable and kernel_size[0] != 1:
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        temporal_kernel_size = [kernel_size[0], 1, 1]
        if isinstance(stride, list) and len(stride) == 3:
            spatial_stride = [1, stride[1], stride[2]]
            temporal_stride = [stride[0], 1, 1]
        else:
            spatial_stride = [1, stride, stride]
            temporal_stride = [stride, 1, 1]
        net = layers.conv3d(
            inputs,
            num_outputs,
            spatial_kernel_size,
            stride=spatial_stride,
            padding=padding,
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_regularizer=weights_regularizer,
            data_format=data_format,
            scope=scope)
        net = layers.conv3d(
            net,
            num_outputs,
            temporal_kernel_size,
            stride=temporal_stride,
            padding=padding,
            scope=scope + '/temporal',
            activation_fn=None,
            normalizer_fn=None,
            data_format=data_format,
            weights_initializer=center_initializer())
        return net
    else:
        return layers.conv3d(
            inputs,
            num_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            activation_fn=None,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_regularizer=weights_regularizer,
            data_format=data_format,
            scope=scope)


def resnet_arg_scope(training=True, w_reg=1e-4):
    batch_norm_params = {
        'is_training': training,
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'center': True,
        'fused': False,
    }
    with slim.arg_scope([conv3d_spatiotemporal],
                        weights_regularizer=slim.l2_regularizer(w_reg),
                        separable=True,
                        normalizer_fn=tf.identity):
        with slim.arg_scope([slim.dropout], is_training=training):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as sc:
                return sc


def block2plus1d(input, num_filters, stride=1, use_final_relu=True):
    num_filters_in = input.get_shape().as_list()[1]

    # residual
    residual = conv3d_spatiotemporal(input, num_filters, kernel_size=[3, 3, 3], stride=[stride, stride, stride],
                                     scope='conv1')
    residual = slim.batch_norm(residual, scope='bn_1')
    residual = tf.nn.relu(residual)

    residual = conv3d_spatiotemporal(residual, num_filters, kernel_size=[3, 3, 3], stride=[1, 1, 1], scope='conv2')
    residual = slim.batch_norm(residual, scope='bn_2')

    # identity
    shortcut = input
    if stride != 1 or num_filters_in != num_filters:
        shortcut = conv3d_spatiotemporal(input, num_filters, kernel_size=[1, 1, 1], stride=[stride, stride, stride],
                                         scope='shortcut')
        shortcut = slim.batch_norm(shortcut, scope='bn_3')

    out = shortcut + residual
    if use_final_relu:
        out = tf.nn.relu(out)

    return out


def r2plus1d_18(net, num_out, reuse=tf.AUTO_REUSE, training=True, scope='resnet',
                module_sizes=(2, 2, 2, 2), filter_sizes=(64, 128, 256, 512), *args, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope(resnet_arg_scope(training=training)):
            feats = {}

            net = conv3d_spatiotemporal(net, 64, kernel_size=[3, 7, 7], stride=[1, 2, 2], scope='conv0')
            net = slim.batch_norm(net, scope='bn_0')
            net = tf.nn.relu(net)

            print('Shape conv_1: {}'.format(net.get_shape().as_list()))
            feats['conv_1'] = net

            block_id = 0
            for i, blocks_in_module in enumerate(module_sizes):
                for j in range(blocks_in_module):
                    block_id += 1
                    stride = 2 if j == 0 and i > 0 else 1
                    with tf.variable_scope("res%d.%d" % (i, j)):
                        print('Block {}'.format(block_id))
                        net = block2plus1d(net, filter_sizes[i], stride)
                        print('Shape {} {}: {}'.format(i, j, net.get_shape().as_list()))
                        feats['block_{}'.format(block_id)] = net
                feats['conv_{}'.format(i+2)] = net
                print('Shape conv_{}: {}'.format(i+2, net.get_shape().as_list()))

            net = tf.nn.relu(net)
            net = tf.reduce_mean(net, [1, 2, 3])
            feats['pre_logit'] = net
            print('Shape pre_logit: {}'.format(net.get_shape().as_list()))
            net = slim.batch_norm(net, scope='bn_last')

            logits = slim.fully_connected(net, num_out, activation_fn=None,
                                          weights_initializer=tf.random_normal_initializer(stddev=1e-3))
            return logits, feats
