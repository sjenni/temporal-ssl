import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import sys
import os
import time
from datetime import datetime
from utils import remove_missing, get_variables_to_train, montage_tf, get_checkpoint_path, average_gradients
from constants import LOG_DIR

slim = tf.contrib.slim


class VideoBaseTrainer:
    def __init__(self, model, data_generator, pre_processor, num_epochs, batch_size, tag='default',
                 optimizer='AdamW', momentum=0.9, wd=1e-4, lr_policy='cosine', init_lr=1e-3, num_gpus=1,
                 train_scopes='encoder', exclude_scopes=(), alpha=1e-3, min_num_frames=None):
        self.model = model
        self.dataset = data_generator
        self.pre_processor = pre_processor
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.tag = tag
        self.opt_type = optimizer
        self.lr_policy = lr_policy
        self.init_lr = init_lr
        self.momentum = momentum
        self.wd = wd
        self.excl_gamma_wd = True
        self.excl_beta_wd = True
        self.excl_bias_wd = True
        self.num_gpus = num_gpus
        self.num_summary_steps = 80
        self.summaries = []
        self.global_step = None
        self.train_scopes = train_scopes
        self.exclude_scopes = exclude_scopes
        self.alpha = alpha
        self.min_num_frames = min_num_frames
        self.num_train_steps = int((self.dataset.num_samples * self.num_epochs) / (self.batch_size * self.num_gpus))
        print('Number of training steps: {}'.format(self.num_train_steps))

    def get_data_queue(self):
        data = self.dataset.get_dataset()
        if self.min_num_frames is not None:
            data = data.filter(lambda x: x['num_frames'] > self.min_num_frames-1)
        data = data.repeat()
        data = data.shuffle(buffer_size=min(self.dataset.num_samples, 1000))
        data = data.map(self.pre_processor.process_train, num_parallel_calls=8)
        data = data.batch(self.batch_size)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        iterator = tf.compat.v1.data.make_one_shot_iterator(data)
        return iterator

    def make_init_fn(self, chpt_path):
        if chpt_path is None:
            chpt_path = get_checkpoint_path(self.get_save_dir())
            if chpt_path is None:
                print('No checkpoint found for initialization')
                return None
            else:
                print('Initializing from previous checkpoint: {}'.format(chpt_path))
        else:
            print('Initializing from provided checkpoint: {}'.format(chpt_path))

        var2restore = slim.get_variables_to_restore(exclude=self.exclude_scopes)
        print('Variables to restore: {}'.format([v.op.name for v in var2restore]))
        var2restore = remove_missing(var2restore, chpt_path)
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(chpt_path, var2restore)
        sys.stdout.flush()

        # Create an initial assignment function.
        def init_fn(sess):
            sess.run(init_assign_op, init_feed_dict)

        return init_fn

    def get_save_dir(self):
        fname = '{}_{}_{}'.format(self.model.name, self.tag, self.dataset.name)
        return os.path.join(LOG_DIR, '{}/'.format(fname))

    def optimizer(self):
        decay = self.lr_decay_mult()
        lr = self.init_lr * decay
        wd = self.wd * decay
        opts = {
            'AdamW': tf.contrib.opt.AdamWOptimizer(wd, learning_rate=lr, beta1=self.momentum, beta2=0.99),
            'MomentumW': tf.contrib.opt.MomentumWOptimizer(wd, learning_rate=lr, momentum=self.momentum)
        }
        return opts[self.opt_type]

    def lr_decay_mult(self):
        boundaries = [np.int64(self.num_train_steps * 0.2 * i) for i in range(1, 4)]
        values = [0.1**i for i in range(4)]

        policies = {
            'cosine': tf.compat.v1.train.cosine_decay(1., self.global_step, self.num_train_steps, alpha=self.alpha),
            'step': tf.train.piecewise_constant(self.global_step, boundaries=boundaries, values=values),
            'const': 1.
        }
        return policies[self.lr_policy]

    def learning_rate(self):
        num_train_steps = self.num_train_steps
        boundaries = [np.int64(num_train_steps * 0.5), np.int64(num_train_steps * 0.75)]
        values = [self.init_lr, self.init_lr * 0.1, self.init_lr * 0.01]
        return tf.train.piecewise_constant(self.global_step, boundaries=boundaries, values=values)

    def make_summaries(self, grads, layers):
        self.summaries.append(tf.compat.v1.summary.scalar('lr_decay_mult', self.lr_decay_mult()))
        # Variable summaries
        for variable in slim.get_model_variables():
            self.summaries.append(tf.compat.v1.summary.histogram(variable.op.name, variable))
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                self.summaries.append(tf.compat.v1.summary.histogram('gradients/' + var.op.name, grad))
        # Add histograms for activation.
        if layers:
            for layer_id, val in layers.iteritems():
                self.summaries.append(tf.compat.v1.summary.histogram('activations/' + layer_id, val))

    def build_model(self, batch_queue, opt, scope, tower_id):
        vids_train, labels_train, ex_train = batch_queue.get_next()
        vids_train.set_shape((self.batch_size,) + self.pre_processor.out_shape)
        labels_train.set_shape((self.batch_size,))
        print('vids_train : {}'.format(vids_train.get_shape().as_list()))

        tf.compat.v1.summary.histogram('vids_train', vids_train)
        tf.compat.v1.summary.histogram('labels_train', labels_train)
        tf.compat.v1.summary.image('imgs/frames_train',
                                   montage_tf(tf.concat(tf.unstack(vids_train, axis=1), 0), 8, self.batch_size),
                                   max_outputs=1)

        # Augment the training examples
        vids_train = self.pre_processor.augment_train(vids_train)

        # Create the model
        preds = self.model.model(vids_train, self.dataset.num_classes)

        tf.compat.v1.summary.image('imgs/frames_train_augmented',
                                   montage_tf(tf.concat(tf.unstack(vids_train, axis=1), 0), 8, self.batch_size),
                                   max_outputs=1)

        # Compute losses
        loss = self.model.loss(scope, preds, self.dataset.format_labels(labels_train))

        # Handle dependencies with update_ops (batch-norm)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        # Calculate the gradients for the batch of data on this tower.
        grads = opt.compute_gradients(loss, get_variables_to_train(self.train_scopes))

        self.summaries += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES, scope)
        return loss, grads, {}

    def train_model(self, chpt_path=None):
        g = tf.Graph()
        with g.as_default():
            with tf.device('/cpu:0'):
                tf.compat.v1.random.set_random_seed(123)

                # Init global step
                self.global_step = tf.compat.v1.train.create_global_step()

                batch_queue = self.get_data_queue()
                opt = self.optimizer()

                # Calculate the gradients for each model tower.
                tower_grads = []
                loss = 0.

                with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
                    for i in range(self.num_gpus):
                        with tf.device('/gpu:%d' % i):
                            with tf.name_scope('tower_{}'.format(i)) as scope:
                                loss_, grads_, layers_ = self.build_model(batch_queue, opt, scope, i)
                                loss += loss_ / self.num_gpus

                            tower_grads.append(grads_)

                grad = average_gradients(tower_grads)

                # Make summaries
                self.make_summaries(grad, layers_)

                # Apply the gradients to adjust the shared variables.
                print('========================================WD VARS===============================================')
                wd_vars = get_variables_to_train(self.train_scopes)
                if self.excl_gamma_wd: wd_vars = [v for v in wd_vars if 'gamma' not in v.op.name]
                if self.excl_beta_wd: wd_vars = [v for v in wd_vars if 'beta' not in v.op.name]
                if self.excl_bias_wd: wd_vars = [v for v in wd_vars if 'biases' not in v.op.name]
                print('WD variables: {}'.format([v.op.name for v in wd_vars]))
                print('==============================================================================================')

                train_op = opt.apply_gradients(grad, global_step=self.global_step, decay_var_list=wd_vars)

                # Group all updates to into a single train op.
                train_op = control_flow_ops.with_dependencies([train_op], loss)

                # Create a saver.
                saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
                init_fn = self.make_init_fn(chpt_path)

                # Build the summary operation from the last tower summaries.
                summary_op = tf.compat.v1.summary.merge(self.summaries)

                # Build an initialization operation to run below.
                init = tf.compat.v1.global_variables_initializer()

                # Start running operations on the Graph.
                sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False), graph=g)
                sess.run(init)
                if init_fn:
                    init_fn(sess)

                summary_writer = tf.compat.v1.summary.FileWriter(self.get_save_dir(), sess.graph)
                init_step = sess.run(self.global_step)
                print('Start training at step: {}'.format(init_step))
                for step in range(init_step, self.num_train_steps):

                    start_time = time.time()
                    _, loss_value = sess.run([train_op, loss])

                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % (self.num_train_steps // 2000) == 0:
                        num_examples_per_step = self.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration
                        print('{}: step {}/{}, loss = {} ({} examples/sec; {} sec/batch)'
                              .format(datetime.now(), step, self.num_train_steps, loss_value,
                                      examples_per_sec, sec_per_batch))
                        sys.stdout.flush()

                    if step % (self.num_train_steps // 200) == 0:
                        print('Writing summaries...')
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)

                    # Save the model checkpoint periodically.
                    if step % (self.num_train_steps // 40) == 0 or (step + 1) == self.num_train_steps:
                        checkpoint_path = os.path.join(self.get_save_dir(), 'model.ckpt')
                        print('Saving checkpoint to: {}'.format(checkpoint_path))
                        saver.save(sess, checkpoint_path, global_step=step)
