import tensorflow as tf
from nets import c3d
slim = tf.contrib.slim


class SLC3D:
    def __init__(self, scope, net_args={}, tag='default', feat_id='maxpool_5'):
        self.scope = scope
        self.name = 'C3D_SL_{}'.format(tag)
        self.net = c3d.c3d_video
        self.net_args = net_args
        self.feat_id = feat_id

    def model(self, input, num_classes, reuse=tf.compat.v1.AUTO_REUSE, training=True):
        preds, feats = self.net(input, num_classes, reuse, training, scope=self.scope, **self.net_args)
        return preds

    def feats(self, input, num_classes, reuse=tf.compat.v1.AUTO_REUSE, training=True):
        preds, feats = self.net(input, num_classes, reuse, training, scope=self.scope, **self.net_args)
        return feats[self.feat_id]

    def loss(self, scope, preds_train, labels_train, tower=0, summary=True):
        # Define the loss
        loss = tf.compat.v1.losses.softmax_cross_entropy(labels_train, preds_train, scope=scope)

        if summary:
            tf.compat.v1.summary.scalar('losses/softmax_cross_entropy_{}'.format(tower), loss)

            # Compute accuracy
            predictions = tf.argmax(preds_train, 1)
            tf.compat.v1.summary.scalar('accuracy/train_accuracy',
                                        slim.metrics.accuracy(predictions, tf.argmax(labels_train, 1)))

        return loss
