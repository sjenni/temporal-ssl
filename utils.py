import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver
import time
import os


def write_results(acc, tag, params, fpath=None, results_dir='results'):
    if not fpath:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        fpath = os.path.join(results_dir, 'results.txt')
    with open(fpath, 'a') as f:
        line = 'Acc:{} Tag:{} Params: {} \n'.format(acc, tag, params)
        f.write(line)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def montage_tf(imgs, num_h, num_w):
    """Makes a montage of imgs that can be used in image_summaries.

    Args:
        imgs: Tensor of images
        num_h: Number of images per column
        num_w: Number of images per row

    Returns:
        A montage of num_h*num_w images
    """
    imgs = tf.unstack(imgs)
    img_rows = [None] * num_h
    for r in range(num_h):
        img_rows[r] = tf.concat(axis=1, values=imgs[r * num_w:(r + 1) * num_w])
    montage = tf.concat(axis=0, values=img_rows)
    return tf.expand_dims(montage, 0)


def remove_missing(var_list, model_path):
    reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    if isinstance(var_list, dict):
        var_dict = var_list
    else:
        var_dict = {var.op.name: var for var in var_list}
    available_vars = {}
    for var in var_dict:

        if reader.has_tensor(var):
            available_vars[var] = var_dict[var]
        else:
            logging.warning(
                'Variable %s missing in checkpoint %s', var, model_path)
    var_list = available_vars
    return var_list


def assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=False,
                              reshape_variables=False):
    """Returns a function that assigns specific variables from a checkpoint.

    Args:
        model_path: The full path to the model checkpoint. To get latest checkpoint
          use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
        var_list: A list of `Variable` objects or a dictionary mapping names in the
          checkpoint to the correspoing variables to initialize. If empty or None,
          it would return  no_op(), None.
        ignore_missing_vars: Boolean, if True it would ignore variables missing in
          the checkpoint with a warning instead of failing.
        reshape_variables: Boolean, if True it would automatically reshape variables
          which are of different shape then the ones stored in the checkpoint but
          which have the same number of elements.

    Returns:
        A function that takes a single argument, a `tf.Session`, that applies the
        assignment operation.

    Raises:
        ValueError: If the checkpoint specified at `model_path` is missing one of
                    the variables in `var_list`.
    """
    if ignore_missing_vars:
        var_list = remove_missing(var_list, model_path)

    saver = tf_saver.Saver(var_list, reshape=reshape_variables)

    def callback(session):
        saver.restore(session, model_path)

    return callback


def get_variables_to_train(trainable_scopes=None):
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if trainable_scopes is None:
        variables_to_train = tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]

        variables_to_train = []
        for scope in scopes:
            variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)

    print('Variables to train: {}'.format([v.op.name for v in variables_to_train]))

    return variables_to_train


def get_checkpoint_path(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if not ckpt:
        print("No checkpoint in {}".format(checkpoint_dir))
        return None
    return ckpt.model_checkpoint_path


def wait_for_new_checkpoint(checkpoint_dir,
                            last_checkpoint=None,
                            seconds_to_sleep=1,
                            timeout=None):
    """Waits until a new checkpoint file is found.
    Args:
      checkpoint_dir: The directory in which checkpoints are saved.
      last_checkpoint: The last checkpoint path used or `None` if we're expecting
        a checkpoint for the first time.
      seconds_to_sleep: The number of seconds to sleep for before looking for a
        new checkpoint.
      timeout: The maximum amount of time to wait. If left as `None`, then the
        process will wait indefinitely.
    Returns:
      a new checkpoint path, or None if the timeout was reached.
    """
    logging.info('Waiting for new checkpoint at %s', checkpoint_dir)
    stop_time = time.time() + timeout if timeout is not None else None
    while True:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt is None:
            checkpoint_path = None
        else:
            checkpoint_path = ckpt.model_checkpoint_path
            ckpt_id = checkpoint_path.split('/')[-1]
            checkpoint_path = os.path.join(checkpoint_dir, ckpt_id)

        # checkpoint_path = tf_saver.latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None or checkpoint_path == last_checkpoint:
            if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
                return None
            time.sleep(seconds_to_sleep)
        else:
            logging.info('Found new checkpoint at %s', checkpoint_path)

            return checkpoint_path
