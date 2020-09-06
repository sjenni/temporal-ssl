import argparse
import sys
import tensorflow as tf

from PreprocessorVideo import Preprocessor
from PreprocessorVideoSSL import PreprocessorTransform
from datasets.UCF101 import UCF101
from datasets.HMDB51 import HMDB51
from train.VideoSSLTrainer import VideoSSLTrainer
from train.VideoBaseTrainer import VideoBaseTrainer
from eval.VideoBaseTester import VideoBaseTester
from models.C3D import SLC3D
from utils import write_results, wait_for_new_checkpoint

# Basic model parameters as external flags.
FLAGS = None


def main(_):
    transforms = FLAGS.transforms.split(',')

    method_name = 'UCF_C3D_{}'.format(''.join(transforms))
    tag = '{}_{}'.format(method_name, FLAGS.tag)
    net_scope = 'features'
    preprocessor_ssl = PreprocessorTransform(seq_length=16, n_speeds=FLAGS.n_speed, crop_size=(112, 112),
                                             resize_shape=(128, 171), transforms=transforms)
    preprocessor = Preprocessor(seq_length=16, skip=FLAGS.frame_skip,
                                crop_size=(112, 112), resize_shape=(128, 171), num_test_seq=32)

    # Initialize the data generator
    dataset_train = UCF101('train_0')

    # Define the network and training
    model = SLC3D(scope=net_scope, tag=tag, net_args={'version': FLAGS.net_version})
    trainer = VideoSSLTrainer(model=model, data_generator=dataset_train, pre_processor=preprocessor_ssl,
                              num_epochs=FLAGS.n_eps_pre, batch_size=FLAGS.batch_size, tag='pre',
                              init_lr=FLAGS.pre_lr, momentum=FLAGS.momentum, wd=FLAGS.wd, skip_pred=FLAGS.skip_pred,
                              num_gpus=FLAGS.num_gpus, train_scopes=net_scope)
    trainer.train_model()
    ckpt = wait_for_new_checkpoint(trainer.get_save_dir(), last_checkpoint=None)

    for i in range(0, 3):
        # Transfer UCF
        transfer_dataset = UCF101('train_{}'.format(i))
        ftuner = VideoBaseTrainer(model=model, data_generator=transfer_dataset, pre_processor=preprocessor,
                                  num_epochs=FLAGS.n_eps_ftune, batch_size=FLAGS.batch_size_ftune,
                                  init_lr=FLAGS.ftune_lr, momentum=FLAGS.momentum, wd=FLAGS.wd,
                                  num_gpus=FLAGS.num_gpus, train_scopes=net_scope, tag='ftune_split{}'.format(i),
                                  exclude_scopes=['global_step', '{}/fc_3'.format(net_scope)])
        ftuner.train_model(ckpt)

        # Evaluate
        dataset_test = UCF101('test_{}'.format(i))
        tester = VideoBaseTester(model, dataset_test, FLAGS.batch_size, preprocessor)
        results = tester.test_classifier_multi_crop(ftuner.get_save_dir())
        write_results(results[0], '{}_ftune_split{}_{}'.format(tag, i, transfer_dataset.name), FLAGS)

        # Finetuning HMDB
        transfer_dataset = HMDB51('train_{}'.format(i))
        ftuner = VideoBaseTrainer(model=model, data_generator=transfer_dataset, pre_processor=preprocessor,
                                  num_epochs=FLAGS.n_eps_ftune, batch_size=FLAGS.batch_size_ftune,
                                  init_lr=FLAGS.ftune_lr, momentum=FLAGS.momentum, wd=FLAGS.wd,
                                  num_gpus=FLAGS.num_gpus, train_scopes=net_scope, tag='ftune_split{}'.format(i),
                                  exclude_scopes=['global_step', '{}/fc_3'.format(net_scope)])
        ftuner.train_model(ckpt)

        # Evaluate
        dataset_test = HMDB51('test_{}'.format(i))
        tester = VideoBaseTester(model, dataset_test, FLAGS.batch_size, preprocessor)
        results = tester.test_classifier_multi_crop(ftuner.get_save_dir())
        write_results(results[0], '{}_ftune_split{}_{}'.format(tag, i, transfer_dataset.name), FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--batch_size_ftune', type=int, default=24)
    parser.add_argument('--n_eps_pre', type=int, default=100,
                        help='Number of epochs for pre-training.')
    parser.add_argument('--n_eps_ftune', type=int, default=75,
                        help='Number of epochs for fine-tuning.')
    parser.add_argument('--pre_lr', type=float, default=3e-4,
                        help='Initial learning rate for pre-training.')
    parser.add_argument('--ftune_lr', type=float, default=5e-5,
                        help='Initial learning rate for the fine-tuning.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum (beta1 in the case of Adam).')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay.')
    parser.add_argument('--net_version', type=str, default='small')
    parser.add_argument('--frame_skip', type=int, default=4)
    parser.add_argument('--n_speed', type=int, default=4)
    parser.add_argument('--transforms', type=str, default='foba,shuffle,warp')
    parser.add_argument('--skip_pred', dest='skip_pred', action='store_true')
    parser.add_argument('--no_skip_pred', dest='skip_pred', action='store_false')
    parser.set_defaults(skip_pred=True)

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
