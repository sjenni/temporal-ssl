# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import cv2
import sys
import threading
from datetime import datetime

import numpy as np
import tensorflow as tf
from six.moves import urllib

LABELS_FILENAME = 'labels.txt'


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
      values: A scalar or list of values.
    Returns:
      a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def video_example(frames, label):
    frames = frames.astype(np.uint8)

    features = {}
    features['num_frames'] = _int64_feature(frames.shape[0])
    features['height'] = _int64_feature(frames.shape[1])
    features['width'] = _int64_feature(frames.shape[2])
    features['channels'] = _int64_feature(frames.shape[3])
    features['label'] = _int64_feature(label)

    # Compress the frames using JPG and store in as a list of strings in 'frames'
    encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes()) for frame in frames]
    features['frames'] = _bytes_list_feature(encoded_frames)

    return tf.train.Example(features=tf.train.Features(feature=features))


def get_video_capture_and_frame_count(path):
    assert os.path.isfile(path), "Couldn't find video file:" + path + ". Skipping video."
    cap = None
    if path:
        cap = cv2.VideoCapture(path)

    assert cap is not None, "Couldn't load video capture:" + path + ". Skipping video."

    # compute meta data of video
    if hasattr(cv2, 'cv'):
        frame_count = int(cap.get(cv2.cv.CAP_PROP_FRAME_COUNT))
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, frame_count


def get_next_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None

    return np.asarray(frame)


def repeat_image_retrieval(cap, file_path, video, take_all_frames, steps, frame,
                           prev_frame_none, frames_counter):
    stop = False

    if frame and prev_frame_none or steps <= 0:
        stop = True
        return stop, cap, video, steps, prev_frame_none, frames_counter

    if not take_all_frames:
        # repeat with smaller step size
        steps -= 1

    prev_frame_none = True
    print("reducing step size due to error for video: ", file_path)
    frames_counter = 0
    cap.release()
    cap = get_video_capture_and_frame_count(file_path)
    # wait for image retrieval to be ready
    time.sleep(2)

    return stop, cap, video, steps, prev_frame_none, frames_counter


def video_file_to_ndarray(file_path, n_frames_per_video, height, width,
                          n_channels):
    cap, frame_count = get_video_capture_and_frame_count(file_path)

    take_all_frames = False
    # if not all frames are to be used, we have to skip some -> set step size accordingly
    if n_frames_per_video == 'all':
        take_all_frames = True
        video = np.zeros((frame_count, height, width, n_channels), dtype=np.uint32)
        steps = frame_count
        n_frames = frame_count
    else:
        video = np.zeros((n_frames_per_video, height, width, n_channels),
                         dtype=np.uint32)
        steps = int(math.floor(frame_count / n_frames_per_video))
        n_frames = n_frames_per_video

    assert not (frame_count < 1 or steps < 1),\
        '{}  does not have enough frames: {}. Skipping video...'.format(file_path, frame_count)

    # variables needed
    image = np.zeros((height, width, n_channels),
                     dtype="uint8")
    frames_counter = 0
    prev_frame_none = False
    restart = True

    while restart:
        for f in range(frame_count):
            if math.floor(f % steps) == 0 or take_all_frames:
                frame = get_next_frame(cap)
                # unfortunately opencv uses bgr color format as default
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # special case handling: opencv's frame count sometimes differs from real frame count -> repeat
                if frame is None and frames_counter < n_frames:
                    stop, _, _1, _2, _3, _4 = repeat_image_retrieval(
                        cap, file_path, video, take_all_frames, steps, frame, prev_frame_none,
                        frames_counter)
                    if stop:
                        restart = False
                        break
                    else:
                        video[frames_counter, :, :, :].fill(0)
                        frames_counter += 1

                else:
                    if frames_counter >= n_frames:
                        restart = False
                        break

                    # iterate over channels
                    for k in range(n_channels):
                        resizedImage = cv2.resize(frame[:, :, k], (width, height))
                        image[:, :, k] = resizedImage

                    # assemble the video from the single images
                    video[frames_counter, :, :, :] = image
                    frames_counter += 1
            else:
                get_next_frame(cap)

    v = video.copy()
    cap.release()
    return v


def process_video_files_batch(thread_index, ranges, name, filenames,
                              labels, num_shards, output_directory):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            try:
                data = video_file_to_ndarray(file_path=filename,
                                             n_frames_per_video='all',
                                             height=256, width=256,
                                             n_channels=3)
                example = video_example(data, label)

                writer.write(example.SerializeToString())
                shard_counter += 1
                counter += 1
                if not counter % 1000:
                    print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                          (datetime.now(), thread_index, counter, num_files_in_thread))
                    sys.stdout.flush()
            except Exception as e:
                print(e)

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def process_video_files(name, filenames, labels, num_shards, num_threads, output_directory):
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, filenames, labels, num_shards, output_directory)
        t = threading.Thread(target=process_video_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()