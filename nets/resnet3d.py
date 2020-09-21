import tensorflow as tf
import tensorflow.contrib.slim as slim


def resnet_arg_scope(training=True, w_reg=1e-4):
    batch_norm_params = {
        'is_training': training,
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'center': True,
        'fused': False,
    }
    with slim.arg_scope([slim.conv3d],
                        weights_regularizer=slim.l2_regularizer(w_reg),
                        activation_fn=None,
                        normalizer_fn=tf.identity):
        with slim.arg_scope([slim.dropout], is_training=training):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as sc:
                return sc


def block2d(input, num_filters, stride=1, use_final_relu=True):
    num_filters_in = input.get_shape().as_list()[1]

    # residual
    residual = slim.conv3d(input, num_filters, kernel_size=(1, 3, 3), stride=(1, stride, stride), scope='conv1')
    residual = slim.batch_norm(residual, scope='bn_1')
    residual = tf.nn.relu(residual)

    residual = slim.conv3d(residual, num_filters, kernel_size=(1, 3, 3), stride=1, scope='conv2')
    residual = slim.batch_norm(residual, scope='bn_2')

    # identity
    shortcut = input
    if stride != 1 or num_filters_in != num_filters:
        shortcut = slim.conv3d(input, num_filters, kernel_size=(1, 1, 1), stride=(1, stride, stride), scope='shortcut')
        shortcut = slim.batch_norm(shortcut, scope='bn_3')

    out = shortcut + residual
    if use_final_relu:
        out = tf.nn.relu(out)

    return out


def block3d(input, num_filters, stride=1, use_final_relu=True):
    num_filters_in = input.get_shape().as_list()[1]

    # residual
    residual = slim.conv3d(input, num_filters, kernel_size=(3, 3, 3), stride=(stride, stride, stride), scope='conv1')
    residual = slim.batch_norm(residual, scope='bn_1')
    residual = tf.nn.relu(residual)

    residual = slim.conv3d(residual, num_filters, kernel_size=(3, 3, 3), stride=1, scope='conv2')
    residual = slim.batch_norm(residual, scope='bn_2')

    # identity
    shortcut = input
    if stride != 1 or num_filters_in != num_filters:
        shortcut = slim.conv3d(input, num_filters, kernel_size=(1, 1, 1), stride=(stride, stride, stride),
                               scope='shortcut')
        shortcut = slim.batch_norm(shortcut, scope='bn_3')

    out = shortcut + residual
    if use_final_relu:
        out = tf.nn.relu(out)

    return out


def resnet3d_18(net, num_out, reuse=tf.AUTO_REUSE, training=True, scope='resnet', blocks=('2d', '2d', '3d', '3d'),
                module_sizes=(2, 2, 2, 2), filter_sizes=(64, 128, 256, 512), *args, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope(resnet_arg_scope(training=training)):
            feats = {}

            net = slim.conv3d(net, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), scope='conv0')
            net = slim.batch_norm(net, scope='bn_0')
            net = tf.nn.relu(net)
            net = slim.max_pool3d(net, kernel_size=(1, 3, 3), stride=(1, 2, 2))

            print('Shape conv_1: {}'.format(net.get_shape().as_list()))
            feats['conv_1'] = net

            block_id = 0
            for i, blocks_in_module in enumerate(module_sizes):
                for j in range(blocks_in_module):
                    block_id += 1
                    stride = 2 if j == 0 and i > 0 else 1
                    with tf.variable_scope("res%d.%d" % (i, j)):
                        print('Block {}'.format(block_id))
                        if blocks[i] == '2d':
                            print('2D block')
                            net = block2d(net, filter_sizes[i], stride)
                        elif blocks[i] == '3d':
                            print('3D block')
                            net = block3d(net, filter_sizes[i], stride)
                        else:
                            net = None
                        print('Shape {} {}: {}'.format(i, j, net.get_shape().as_list()))
                        feats['block_{}'.format(block_id)] = net
                feats['conv_{}'.format(i+2)] = net
                print('Shape conv_{}: {}'.format(i+2, net.get_shape().as_list()))

            net = tf.reduce_mean(net, [1, 2, 3])
            feats['pre_logit'] = net
            print('Shape pre_logit: {}'.format(net.get_shape().as_list()))
            logits = slim.fully_connected(net, num_out, activation_fn=None,
                                          weights_initializer=tf.random_normal_initializer(stddev=1e-3))
            return logits, feats


