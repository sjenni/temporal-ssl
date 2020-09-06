import tensorflow as tf

slim = tf.contrib.slim


class PreprocessorTransform:
    def __init__(self, seq_length, n_speeds=1, crop_size=(256, 256), resize_shape=(256, 256), src_shape=(256, 256),
                 flip_prob=0.5, transforms=('foba', 'shuffle', 'warp'), augment_color=False, rand_grey=0.):
        self.crop_size = crop_size
        self.src_shape = src_shape
        self.resize_shape = resize_shape
        self.seq_length = seq_length
        self.n_speeds = n_speeds
        self.flip_prob = flip_prob
        self.transforms = transforms
        self.augment_color = augment_color
        self.rand_grey = rand_grey
        self.load_shape = (seq_length,) + src_shape + (3,)
        self.out_shape = (len(self.transforms) + 1, seq_length) + resize_shape + (3,)
        self.test_sequence = 'middle'  # Can be 'middle' or 'random'

    def color_scale_vid(self, vid):
        vid = tf.cast(vid, tf.float32) / 255.
        vid = tf.map_fn(self.color, vid)
        return vid

    def load_inds(self, example, inds):
        images = tf.map_fn(lambda i: tf.image.decode_jpeg(example["frames"].values[i]), inds, dtype=tf.uint8)
        images = tf.reverse(images, axis=[-1])  # To RGB
        images = tf.reshape(images, [-1, self.load_shape[1], self.load_shape[2], 3])
        return images

    def sub_sample_seq(self, vid, n_frames, skip):
        max_frame = n_frames - self.seq_length * skip + 1
        random_offset = tf.random.uniform(shape=(), minval=0, maxval=max_frame, dtype=tf.int64)
        inds = tf.range(random_offset, random_offset + self.seq_length * skip, delta=skip)
        sub_inds = tf.gather(vid, inds)
        return sub_inds

    def process_train(self, example):
        full_inds = tf.range(0, example["num_frames"])

        # Choose a speed
        max_speed = tf.minimum(tf.cast(log2(example["num_frames"] / self.seq_length), tf.int64), self.n_speeds-1)
        speed_label = tf.cond(tf.greater(max_speed, 0),
                              lambda: tf.random.uniform(shape=(), minval=0, maxval=max_speed+1, dtype=tf.int64),
                              lambda: tf.constant(0, tf.int64))
        eff_skip = 2 ** speed_label

        # Speed type video
        vid_ori_inds = self.sub_sample_seq(full_inds, example["num_frames"], eff_skip)
        vid_orig = self.load_inds(example, vid_ori_inds)
        vid_orig.set_shape(self.load_shape)
        vids_transformed = [vid_orig]

        # Periodic (forward/backward) transformation
        if 'foba' in self.transforms:
            sub_len = tf.minimum(example["num_frames"], (self.seq_length - 1) * eff_skip)
            crop_inds = tf.random_crop(full_inds, [sub_len])
            inds = tf.range(0, (self.seq_length - 2) * eff_skip, delta=eff_skip)
            start = tf.cond(tf.greater(eff_skip, 1),
                            lambda: tf.constant(1, tf.int64),
                            lambda: tf.constant(0, tf.int64))
            random_offset = tf.random.uniform(shape=(), minval=start, maxval=eff_skip + 1 - start, dtype=tf.int64)
            inds_foba = tf.concat([inds, random_offset + tf.reverse(inds, [0])], 0)
            inds_foba = tf.random_crop(inds_foba, [self.seq_length])
            vid_foba_inds = tf.gather(crop_inds, inds_foba)
            vid_foba = self.load_inds(example, vid_foba_inds)
            vid_foba.set_shape(self.load_shape)
            vids_transformed.append(vid_foba)

        # Random transformation
        if 'shuffle' in self.transforms:
            vid_shuffle_inds = self.sub_sample_seq(full_inds, example["num_frames"], 1)
            shuffle_inds = tf.random_shuffle(tf.range(0, self.seq_length))
            vid_shuffle_inds = tf.gather(vid_shuffle_inds, shuffle_inds)
            vid_shuffle = self.load_inds(example, vid_shuffle_inds)
            vid_shuffle.set_shape(self.load_shape)
            vids_transformed.append(vid_shuffle)

        # Warp transformation
        if 'warp' in self.transforms:
            sub_len = tf.cond(tf.greater(max_speed, 1),
                              lambda: tf.cast(self.seq_length * 2 ** max_speed, tf.int64),
                              lambda: example["num_frames"])
            max_offset = 2 ** max_speed
            off_sets = tf.random.uniform(shape=(self.seq_length,), minval=1, maxval=max_offset + 1, dtype=tf.int64)
            inds_warp_v3 = tf.cumsum(off_sets)

            # Special treatment for speed0
            inds_warp_v1 = tf.random_shuffle(tf.range(0, sub_len))
            inds_warp_v1 = inds_warp_v1[:self.seq_length]
            inds_warp_v1 = tf.sort(inds_warp_v1)
            inds_warp = tf.cond(tf.greater(max_speed, 1), lambda: inds_warp_v3, lambda: inds_warp_v1)

            vid_warp_inds = tf.random_crop(full_inds, [sub_len])
            vid_warp_inds = tf.gather(vid_warp_inds, inds_warp)
            vid_warp = self.load_inds(example, vid_warp_inds)
            vid_warp.set_shape(self.load_shape)
            vids_transformed.append(vid_warp)

        vids_transformed = [self.resize(self.color_scale_vid(v)) for v in vids_transformed]
        vids_transformed = tf.stack(vids_transformed)

        return vids_transformed, speed_label, example

    def flip_lr(self, vid):
        flipped_vids = tf.map_fn(lambda x: flip_video(x, self.flip_prob), vid)
        return flipped_vids

    def color(self, image, bright_max_delta=32. / 255., lower_sat=0.5, upper_sat=1.5):
        if self.augment_color:
            image = tf.cond(tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
                            true_fn=lambda: tf.image.random_saturation(
                                tf.image.random_brightness(image, max_delta=bright_max_delta),
                                lower=lower_sat, upper=upper_sat),
                            false_fn=lambda: tf.image.random_brightness(
                                tf.image.random_saturation(image, lower=lower_sat, upper=upper_sat),
                                max_delta=bright_max_delta))

        if self.rand_grey > 0.:
            image = tf.cond(tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > self.rand_grey,
                            true_fn=lambda: image,
                            false_fn=lambda: tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3]))

        # Scale to [-1, 1]
        image = tf.cast(image, tf.float32) * 2. - 1.
        image = tf.clip_by_value(image, -1., 1.)
        return image

    def random_crop(self, vid):
        in_sz = vid.get_shape().as_list()
        return tf.image.random_crop(vid, [in_sz[0], in_sz[1], in_sz[2], self.crop_size[0], self.crop_size[1], in_sz[-1]])

    def center_crop(self, vid):
        in_sz = vid.get_shape().as_list()
        dh = tf.cast(tf.round((in_sz[2] - self.crop_size[0]) / 2), tf.int64)
        dw = tf.cast(tf.round((in_sz[3] - self.crop_size[1]) / 2), tf.int64)
        cropped_vid = vid[:, :, dh:dh + self.crop_size[0], dw:dw + self.crop_size[1], :]
        return cropped_vid

    def resize(self, vid):
        if not self.resize_shape == self.src_shape:
            resized_video = tf.compat.v1.image.resize_bilinear(vid, self.resize_shape)
            return resized_video
        else:
            return vid

    def augment_train(self, video):
        video = self.flip_lr(video)
        video = self.random_crop(video)
        return video


def flip_video(vid, flip_prob):
    return tf.cond(tf.random.uniform(shape=(), minval=0.0, maxval=1.0) > flip_prob,
                   true_fn=lambda: vid,
                   false_fn=lambda: tf.reverse(vid, [3]))


def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator
