import tensorflow as tf
import os

from constants import UCF101_TFDIR

slim = tf.contrib.slim


class UCF101:
    SPLITS_TO_SIZES = {'train_0': 9537, 'test_0': 3783,
                       'train_1': 9586, 'test_1': 3734,
                       'train_2': 9624, 'test_2': 3696}

    def __init__(self, split_name='train_0'):
        self.split_name = split_name
        self.reader = tf.TFRecordReader
        self.label_offset = 0
        self.is_multilabel = False
        self.data_dir = UCF101_TFDIR
        self.file_pattern = '%s-*'
        self.num_classes = 101
        self.name = 'UCF101'
        self.num_samples = self.SPLITS_TO_SIZES[split_name]
        self.min_nframe = 29

    def feature_description(self):
        keys_to_features = {
            'num_frames': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channels': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'frames': tf.io.VarLenFeature(tf.string),
        }
        return keys_to_features

    def get_trainset_labelled(self):
        return self.get_split('train')

    def get_trainset_unlabelled(self):
        return self.get_split('train')

    def get_testset(self):
        return self.get_split('validation')

    def get_num_train_labelled(self):
        return self.SPLITS_TO_SIZES['train']

    def get_num_train_unlabelled(self):
        return self.SPLITS_TO_SIZES['train']

    def get_num_test(self):
        return self.SPLITS_TO_SIZES['validation']

    def get_split_size(self, split_name):
        return self.SPLITS_TO_SIZES[split_name]

    def format_labels(self, labels):
        return slim.one_hot_encoding(tf.math.mod(labels, self.num_classes), self.num_classes)

    def get_split(self, split_name, data_dir=None):
        """Gets a dataset tuple with instructions for reading ImageNet.
        Args:
          split_name: A train/eval split name.
          data_dir: The base directory of the dataset sources.
        Returns:
          A `Dataset` namedtuple.
        Raises:
          ValueError: if `split_name` is not a valid train/eval split.
        """
        if split_name not in self.SPLITS_TO_SIZES:
            raise ValueError('split name %s was not recognized.' % split_name)

        if not data_dir:
            data_dir = self.data_dir

        tf_record_pattern = os.path.join(data_dir, self.file_pattern % split_name)
        data_files = tf.io.gfile.glob(tf_record_pattern)
        if not data_files:
            print('No files found for dataset at %s' % data_dir)
        print('data_files: {}'.format(data_files))

        raw_dataset = tf.data.TFRecordDataset(data_files)

        # Build the decoder
        feature_description = self.feature_description()

        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)

        parsed_dataset = raw_dataset.map(_parse_function)

        return parsed_dataset

    def get_dataset(self):
        return self.get_split(self.split_name)