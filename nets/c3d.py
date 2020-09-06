import tensorflow as tf
import tensorflow.contrib.slim as slim


def c3d_argscope(activation=tf.nn.relu, kernel_size=3, padding='SAME', training=True):
    norm_params = {
        'is_training': training,
        'decay': 0.975,
        'epsilon': 0.001,
        'center': True,
        'scale': True,
        'fused': True,
    }
    normalizer_fn = slim.batch_norm
    with slim.arg_scope([slim.conv3d],
                        kernel_size=kernel_size,
                        padding=padding,
                        activation_fn=activation,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=norm_params):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=activation,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=norm_params):
            with slim.arg_scope([slim.max_pool3d], padding=padding):
                with slim.arg_scope([slim.batch_norm], **norm_params):
                    with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                        return arg_sc


def c3d(net,
        reuse=None,
        is_training=True,
        scope='c3d',
        use_fc=True):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        with slim.arg_scope(c3d_argscope(activation=tf.nn.relu, kernel_size=3, padding='SAME', training=is_training)):
            end_points = {}

            net = slim.conv3d(net, 64, scope='conv_1')
            print('conv_1 feats: {}'.format(net.get_shape().as_list()))
            end_points['conv_1'] = net

            net = slim.max_pool3d(net, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            net = slim.conv3d(net, 128, scope='conv_2')
            print('conv_2 feats: {}'.format(net.get_shape().as_list()))
            end_points['conv_2'] = net

            net = slim.max_pool3d(net, kernel_size=2, stride=2)
            net = slim.conv3d(net, 256, scope='conv_3_1')
            net = slim.conv3d(net, 256, scope='conv_3_2')
            print('conv_3 feats: {}'.format(net.get_shape().as_list()))
            end_points['conv_3'] = net

            net = slim.max_pool3d(net, kernel_size=2, stride=2)
            net = slim.conv3d(net, 512, scope='conv_4_1')
            net = slim.conv3d(net, 512, scope='conv_4_2')
            print('conv_4 feats: {}'.format(net.get_shape().as_list()))
            end_points['conv_4'] = net

            net = slim.max_pool3d(net, kernel_size=2, stride=2)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
            net = slim.conv3d(net, 512, scope='conv_5_1', padding='VALID')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
            net = slim.conv3d(net, 512, scope='conv_5_2', padding='VALID')
            print('conv_5 feats: {}'.format(net.get_shape().as_list()))
            end_points['conv_5'] = net

            net = slim.max_pool3d(net, kernel_size=2, stride=2)

            net = tf.reshape(net, [net.get_shape().as_list()[0], -1])
            # net = slim.flatten(net)
            print('flattened feats: {}'.format(net.get_shape().as_list()))

            net = slim.fully_connected(net, 4096, scope='fc_1')
            print('fc_1 feats: {}'.format(net.get_shape().as_list()))
            end_points['fc_1'] = net

            net = slim.fully_connected(net, 4096, scope='fc_2')
            print('fc_2 feats: {}'.format(net.get_shape().as_list()))
            end_points['fc_2'] = net

        return net, end_points


def c3d_small(net,
              reuse=None,
              is_training=True,
              scope='c3d',
              use_fc=True):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        with slim.arg_scope(c3d_argscope(activation=tf.nn.relu, kernel_size=3, padding='SAME', training=is_training)):
            end_points = {}

            net = slim.conv3d(net, 64, scope='conv_1')
            print('conv_1 feats: {}'.format(net.get_shape().as_list()))
            end_points['conv_1'] = net

            net = slim.max_pool3d(net, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            net = slim.conv3d(net, 128, scope='conv_2')
            print('conv_2 feats: {}'.format(net.get_shape().as_list()))
            end_points['conv_2'] = net

            net = slim.max_pool3d(net, kernel_size=2, stride=2)
            net = slim.conv3d(net, 256, scope='conv_3')
            print('conv_3 feats: {}'.format(net.get_shape().as_list()))
            end_points['conv_3'] = net

            net = slim.max_pool3d(net, kernel_size=2, stride=2)
            net = slim.conv3d(net, 256, scope='conv_4')
            print('conv_4 feats: {}'.format(net.get_shape().as_list()))
            end_points['conv_4'] = net

            net = slim.max_pool3d(net, kernel_size=2, stride=2)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
            net = slim.conv3d(net, 256, scope='conv_5', padding='VALID')
            print('conv_5 feats: {}'.format(net.get_shape().as_list()))
            end_points['conv_5'] = net

            net = slim.max_pool3d(net, kernel_size=2, stride=2)
            end_points['maxpool_5'] = net

            net = tf.reshape(net, [net.get_shape().as_list()[0], -1])
            print('flattened feats: {}'.format(net.get_shape().as_list()))

            if use_fc:
                net = slim.fully_connected(net, 2048, scope='fc_1')
                print('fc_1 feats: {}'.format(net.get_shape().as_list()))
                end_points['fc_1'] = net

                net = slim.fully_connected(net, 2048, scope='fc_2')
                print('fc_2 feats: {}'.format(net.get_shape().as_list()))
                end_points['fc_2'] = net

        return net, end_points


def c3d_video(net,
              num_classes=1000,
              reuse=None,
              is_training=True,
              scope='c3d',
              version='large',
              use_fc=True):
    print('Using C3D {}'.format(version))
    print('C3D input: {}'.format(net.get_shape().as_list()))
    if version == 'large':
        vid_feats, end_points = c3d(net, reuse, is_training, scope, use_fc)
    elif version == 'small':
        vid_feats, end_points = c3d_small(net, reuse, is_training, scope, use_fc)
    else:
        raise ValueError('C3D network version {} not valid!'.format(version))

    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        with slim.arg_scope(c3d_argscope(training=is_training)):
            preds = slim.fully_connected(vid_feats, num_classes, scope='fc_3', activation_fn=None, normalizer_fn=None)
            end_points['fc_3'] = preds
            return preds, end_points
