import numpy as np
import tensorflow as tf
from utils import get_checkpoint_path


class VideoBaseTester:
    def __init__(self, model, dataset, batch_size, pre_processor, min_num_frames=None, flip_aug=False):
        self.model = model
        self.pre_processor = pre_processor
        self.batch_size = batch_size
        self.dataset = dataset
        self.min_num_frames = min_num_frames
        self.flip_aug = flip_aug
        self.num_eval_steps = self.dataset.num_samples // self.batch_size
        print('Number of evaluation steps: {}'.format(self.num_eval_steps))

    def get_data_queue_multi_crop(self, shuffle=False):
        data = self.dataset.get_dataset()
        if shuffle:
            data = data.shuffle(buffer_size=1024)
        data = data.map(self.pre_processor.process_test_full, num_parallel_calls=1)
        data = data.prefetch(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(data)
        return iterator

    def make_test_summaries(self, names_to_values):
        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.items():
            op = tf.compat.v1.summary.scalar(metric_name, metric_value)
            summary_ops.append(op)
        return summary_ops

    def test_classifier_multi_crop(self, ckpt_dir):
        print('Restoring from: {}'.format(ckpt_dir))
        self.batch_size = 1

        g = tf.Graph()
        with g.as_default():
            tf.compat.v1.random.set_random_seed(123)

            # Get test batches
            batch_queue = self.get_data_queue_multi_crop()
            vids_test, labels_test, ex_test = batch_queue.get_next()
            vids_test = tf.reshape(vids_test, (-1,) + self.pre_processor.out_shape)

            # Center crop the videos
            vids_test = self.pre_processor.center_crop(vids_test)
            if self.flip_aug:
                vids_test = tf.concat([vids_test, tf.reverse(vids_test, [-2])], 0)

            # Get predictions
            preds_test = tf.map_fn(lambda x: self.model.model(tf.expand_dims(x, 0),
                                                              self.dataset.num_classes,
                                                              training=False),
                                   vids_test)

            preds_test = tf.reduce_mean(tf.nn.softmax(preds_test), 0, keep_dims=False)
            print('Preds test: {}'.format(preds_test.get_shape().as_list()))

            # preds_test = self.model.model(vids_test, self.dataset.num_classes, training=False)
            preds_test = tf.argmax(preds_test, 1)

            # Start running operations on the Graph.
            init = tf.compat.v1.global_variables_initializer()
            sess = tf.compat.v1.Session()
            sess.run(init)

            prev_ckpt = get_checkpoint_path(ckpt_dir)
            print('Restoring from previous checkpoint: {}'.format(prev_ckpt))
            saver = tf.train.Saver(tf.compat.v1.global_variables())
            saver.restore(sess, prev_ckpt)

            preds_list = []
            labels_list = []

            for step in range(self.dataset.num_samples):
                preds_np, labels_np = sess.run([preds_test, labels_test])
                preds_list.append(preds_np)
                labels_list.append(labels_np)
                if step % (self.dataset.num_samples // 10) == 0:
                    acc = np.mean(np.concatenate(preds_list, 0) == labels_list)
                    print('Evaluation step {}/{} Mini-Batch Acc: {}'
                          .format(step, self.dataset.num_samples, acc))

            print('Len preds: {}'.format(len(labels_list)))
            acc = np.mean(np.concatenate(preds_list, 0) == labels_list)
            print('Accuracy: {}'.format(acc))

            print('preds: {}'.format(np.concatenate(preds_list, 0).shape))

            return [acc]
