import tensorflow as tf
from PreprocessorVideoSSL import PreprocessorTransform

slim = tf.contrib.slim


class PreprocessorTransformResNet(PreprocessorTransform):
    def __init__(self,  *args, **kwargs):
        PreprocessorTransform.__init__(self, *args, **kwargs)

    def resize_crop(self, vid):
        in_sz = vid.get_shape().as_list()
        rand_sz = tf.random_uniform((), 0.5, 1.0)
        resize_sh = tf.cast(tf.round(in_sz * rand_sz), tf.int32)
        vid = tf.compat.v1.image.resize_bilinear(vid, resize_sh[-3:-1])
        vid = self.random_crop(vid)
        return vid
