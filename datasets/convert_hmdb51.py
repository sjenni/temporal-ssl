import os
import random
import tensorflow as tf
from datasets.dataset_utils import process_video_files


def _find_video_files(name, source_directory, label_dir, split, set_id):
    random.seed(123)
    print('Finding HMDB data for split: {}'.format(name))

    data_dir = os.path.join(source_directory, 'videos')
    class_list = sorted(['{}_test'.format(c) for c in os.listdir(data_dir)])
    class_list = [c[:-5] for c in class_list]

    label_files = sorted(os.listdir(os.path.join(source_directory, label_dir)))
    label_files = [os.path.join(source_directory, label_dir, lf) for lf in label_files if 'split{}'.format(split) in lf]

    fnames = []
    labels = []

    for c, l_file in enumerate(label_files):
        lfile_examples = [l.strip().split(' ') for l in tf.gfile.FastGFile(l_file, 'r').readlines()]
        examples = [ex[0] for ex in lfile_examples if ex[1] == '{}'.format(set_id)]

        class_dir = os.path.join(data_dir, class_list[c])
        vid_paths = [os.path.join(class_dir, v) for v in examples]

        fnames += vid_paths
        labels += [c]*len(vid_paths)

    shuffled_index = range(len(fnames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    fnames = [fnames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found {} videos across {} labels inside {}.'.format(len(fnames), len(class_list), data_dir))

    return fnames, labels


def _process_dataset(name, directory, num_shards, num_threads, output_directory, labels_dir, split, set_id):
    filenames, labels = _find_video_files(name, directory, labels_dir, split, set_id)

    process_video_files(name, filenames, labels, num_shards, num_threads, output_directory)


def run(source_directory, output_directory, train_shards=32, validation_shards=8,
        num_threads=8):
    assert not train_shards % num_threads, (
        'Please make the num_threads commensurate with train_shards')
    assert not validation_shards % num_threads, (
        'Please make the num_threads commensurate with '
        'validation_shards')
    print('Saving results to %s' % output_directory)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    labels_dir = 'testTrainMulti_7030_splits'
    for split in range(3):
        _process_dataset('train_{}'.format(split), source_directory, train_shards, num_threads, output_directory,
                         labels_dir, split+1, 1)
        _process_dataset('test_{}'.format(split), source_directory, validation_shards, num_threads, output_directory,
                         labels_dir, split+1, 2)

