import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from train.VideoBaseTrainer import VideoBaseTrainer
from utils import get_variables_to_train, montage_tf

slim = tf.contrib.slim


class VideoSSLTrainer(VideoBaseTrainer):

    def __init__(self, skip_pred=True, *args, **kwargs):
        VideoBaseTrainer.__init__(self, *args, **kwargs)
        self.skip_pred = skip_pred

    def build_model(self, batch_queue, opt, scope, tower_id):
        bs = self.batch_size
        vids_train, skip_label, example = batch_queue.get_next()
        vids_train.set_shape((bs,) + self.pre_processor.out_shape)
        print('vids_train : {}'.format(vids_train.get_shape().as_list()))
        tf.compat.v1.summary.histogram('skip_label', skip_label)

        # Perform common augmentations
        vids_train = self.pre_processor.augment_train(vids_train)
        vids_transformed = tf.unstack(vids_train, axis=1)
        transforms = ['orig'] + self.pre_processor.transforms

        # Make summaries
        for v, transform in zip(vids_transformed, transforms):
            tf.compat.v1.summary.image('imgs/{}'.format(transform),
                                       montage_tf(tf.concat(tf.unstack(v, axis=1), 0), 16, bs),
                                       max_outputs=1)

        # Construct net input
        num_transform_classes = len(vids_transformed)
        vids_train = tf.concat(vids_transformed, 0)
        labels_train = tf.concat([i * tf.ones((bs,), dtype=tf.int32) for i in range(num_transform_classes)], 0)
        labels_train = tf.one_hot(labels_train, num_transform_classes)

        num_skip_classes = self.pre_processor.n_speeds
        skip_label = tf.one_hot(skip_label, num_skip_classes)
        skip_label = tf.concat([skip_label, skip_label], 0)

        if self.skip_pred:
            num_classes = num_transform_classes + num_skip_classes
        else:
            num_classes = num_transform_classes

        # Create the model
        preds = self.model.model(vids_train, num_classes)

        if self.skip_pred:
            preds_transform, preds_skip = tf.split(preds, [num_transform_classes, num_skip_classes], -1)
            preds_skip = preds_skip[:2*bs]
            tf.compat.v1.summary.scalar('accuracy/skip_pred',
                                        slim.metrics.accuracy(tf.argmax(preds_skip, 1),
                                                              tf.argmax(skip_label, 1)))
        else:
            preds_transform = preds

        # Compute accuracy
        predictions_transform = tf.argmax(preds_transform, 1)
        labels_transform = tf.argmax(labels_train, 1)
        tf.compat.v1.summary.scalar('accuracy/all_transforms',
                                    slim.metrics.accuracy(predictions_transform, labels_transform))

        for p, l, t in zip(tf.split(predictions_transform, len(transforms)),
                           tf.split(labels_transform, len(transforms)),
                           transforms):
            tf.compat.v1.summary.scalar('accuracy/{}'.format(t), slim.metrics.accuracy(p, l))

        # Compute losses
        loss_transform = self.model.loss(scope, preds_transform, labels_train, summary=False)
        tf.compat.v1.summary.scalar('losses/loss_transform', loss_transform)

        if self.skip_pred:
            loss_skip = self.model.loss(scope, preds_skip, skip_label, summary=False)
            tf.compat.v1.summary.scalar('losses/loss_skip', loss_skip)
            loss = loss_transform + loss_skip
        else:
            loss = loss_transform

        # Handle dependencies with update_ops (batch-norm)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        # Calculate the gradients for the batch of data on this tower.
        grads = opt.compute_gradients(loss, get_variables_to_train(self.train_scopes))

        self.summaries += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES, scope)
        return loss, grads, {}
