import tensorflow as tf

slim = tf.contrib.slim


class Preprocessor:
    def __init__(self, seq_length, skip=1, crop_size=(256, 256), resize_shape=(256, 256), src_shape=(256, 256),
                 flip_prob=0.5, num_test_seq=None):
        self.crop_size = crop_size
        self.src_shape = src_shape
        self.resize_shape = resize_shape
        self.seq_length = seq_length
        self.skip = skip
        self.flip_prob = flip_prob
        self.load_shape = (seq_length,) + src_shape + (3,)
        self.out_shape = (seq_length,) + resize_shape + (3,)
        self.num_test_seq = num_test_seq

    def scale(self, image):
        image = tf.cast(image, tf.float32) / 255.
        image = image * 2. - 1.
        image = tf.clip_by_value(image, -1., 1.)
        return image

    def process_train(self, example):
        seq_length = self.seq_length

        # Randomly sample offset from the valid range.
        eff_skip = tf.minimum(tf.cast(tf.floor(example["num_frames"] / seq_length), tf.int64), self.skip)
        max_frame = example["num_frames"] - seq_length * eff_skip + 1
        random_offset = tf.random.uniform(shape=(), minval=0, maxval=max_frame, dtype=tf.int64, name='offset')
        offsets = tf.range(random_offset, random_offset + seq_length * eff_skip, delta=eff_skip)

        # Decode the encoded JPG images
        images = tf.map_fn(lambda i: tf.image.decode_jpeg(example["frames"].values[i]), offsets, dtype=tf.uint8)

        images.set_shape(self.load_shape)
        images = tf.reverse(images, axis=[-1])  # To RGB
        images = self.scale(images)
        images = self.resize(images)
        label = tf.cast(example["label"], tf.int64)

        return images, label, example

    def process_test_full(self, example):
        # Compute the offset to sample the sequence
        eff_skip = tf.minimum(tf.cast(tf.floor(example["num_frames"] / self.seq_length), tf.int64), self.skip)
        max_frame = example["num_frames"] - self.seq_length * eff_skip + 1
        if self.num_test_seq is None:
            start_inds = tf.range(0, max_frame)
        else:
            start_inds = tf.range(0, max_frame, tf.maximum(max_frame//self.num_test_seq, 1))
        inds_all_sub = tf.map_fn(lambda i: tf.range(i, i + self.seq_length * eff_skip, delta=eff_skip), start_inds,
                                 dtype=tf.int64)

        # Decode the encoded JPG images
        inds = tf.range(0, example["num_frames"])
        images = tf.map_fn(lambda i: tf.image.decode_jpeg(example["frames"].values[i]), inds, dtype=tf.uint8)
        images.set_shape(self.load_shape)
        images = tf.reverse(images, axis=[-1])  # To RGB
        images = self.scale(images)
        images = self.resize(images)

        label = tf.cast(example["label"], tf.int64)

        vids_all = tf.gather(images, inds_all_sub, axis=0, batch_dims=1)

        return vids_all, label, example

    def flip_lr(self, vid):
        flipped_vids = tf.map_fn(lambda x: flip_video(x, self.flip_prob), vid)
        return flipped_vids

    def random_crop(self, vid):
        in_sz = vid.get_shape().as_list()
        return tf.image.random_crop(vid, [in_sz[0], in_sz[1], self.crop_size[0], self.crop_size[1], in_sz[-1]])

    def center_crop(self, vid):
        in_sz = vid.get_shape().as_list()
        dh = tf.cast(tf.round((in_sz[2] - self.crop_size[0]) / 2), tf.int64)
        dw = tf.cast(tf.round((in_sz[3] - self.crop_size[1]) / 2), tf.int64)
        cropped_vid = vid[:, :, dh:dh + self.crop_size[0], dw:dw + self.crop_size[1], :]
        return cropped_vid

    def resize(self, vid):
        if not self.resize_shape == self.src_shape:
            print('Resizing videos from {} to {}'.format(self.src_shape, self.resize_shape))
            resized_video = tf.compat.v1.image.resize_bilinear(vid, self.resize_shape)
            return resized_video
        else:
            return vid

    def augment_train(self, video):
        video = self.flip_lr(video)
        video = self.random_crop(video)
        return video


def flip_video(vid, flip_prob):
    # print('vid before flip: {}'.format(vid.get_shape().as_list()))
    return tf.cond(tf.random.uniform(shape=(), minval=0.0, maxval=1.0) > flip_prob,
                   true_fn=lambda: vid,
                   false_fn=lambda: tf.reverse(vid, [2]))
