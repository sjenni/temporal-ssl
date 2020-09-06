import os
import random
import tensorflow as tf

from datasets.dataset_utils import process_video_files


def _find_video_files(name, source_directory, label_file):
    random.seed(123)
    print('Finding UCF data for split: {}'.format(name))
    lfile_examples = [l.strip().split(' ')[0] for l in tf.gfile.FastGFile(label_file, 'r').readlines()]

    data_dir = os.path.join(source_directory, 'videos')
    class_list = sorted(os.listdir(data_dir))

    fnames = []
    labels = []

    for i, c in enumerate(class_list):
        class_dir = os.path.join(data_dir, c)
        vids = os.listdir(class_dir)
        vid_paths = [os.path.join(c, v) for v in vids]

        vid_paths = list(set(vid_paths) & set(lfile_examples))
        vid_paths = [os.path.join(data_dir, v) for v in vid_paths]

        fnames += vid_paths
        labels += [i]*len(vid_paths)

    shuffled_index = range(len(fnames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    fnames = [fnames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found {} videos across {} labels inside {}.'.format(len(fnames), len(class_list), data_dir))
    assert len(fnames) == len(lfile_examples), 'Number of examples do not match'

    return fnames, labels


def _process_dataset(name, directory, num_shards, num_threads, output_directory, label_file):
    filenames, labels = _find_video_files(name, directory, label_file)
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

    for split in range(3):
        train_file = os.path.join(source_directory, 'ucfTrainTestlist', 'trainlist0{}.txt'.format(split+1))
        test_file = os.path.join(source_directory, 'ucfTrainTestlist', 'testlist0{}.txt'.format(split+1))
        _process_dataset('train_{}'.format(split), source_directory, train_shards, num_threads, output_directory, train_file)
        _process_dataset('test_{}'.format(split), source_directory, validation_shards, num_threads, output_directory, test_file)

