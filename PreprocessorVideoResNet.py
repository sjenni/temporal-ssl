import tensorflow as tf
from PreprocessorVideo import Preprocessor

slim = tf.contrib.slim


class PreprocessorResNet(Preprocessor):
    def __init__(self, rand_resize=(0.5, 1.), *args, **kwargs):
        Preprocessor.__init__(self, *args, **kwargs)
        self.rand_resize = rand_resize

    def random_resize_crop(self, vid):
        in_sz = vid.get_shape().as_list()
        rand_sz = tf.random_uniform((), self.rand_resize[0], self.rand_resize[1])
        resize_sh = tf.cast(tf.round(in_sz * rand_sz), tf.int32)
        vid = tf.compat.v1.image.resize_bilinear(vid, resize_sh[-3:-1])
        vid = tf.image.random_crop(vid, [in_sz[0], self.crop_size[0], self.crop_size[1], in_sz[-1]])
        return vid

    def augment_train(self, video):
        video = self.flip_lr(video)
        video = tf.map_fn(self.random_resize_crop, video)
        return video
