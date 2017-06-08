from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import numpy as np

from util import log
from pprint import pprint

from input_ops import create_input_ops, check_data_id

import os
import time
import numpy as np
import tensorflow as tf
import h5py

class PoseEvalManager(object):

    def __init__(self):
        # collection of batches (not flattened)
        self._ids = []
        self._predictions = []
        self._groundtruths = []

    def add_batch(self, id, prediction, groundtruth):
        assert prediction.shape == groundtruth.shape
        self._ids.append(id)
        self._predictions.append(prediction)
        self._groundtruths.append(groundtruth)

    def dump_result(self, filename):
        log.infov("Dumping prediction result into %s ...", filename)
        f.h5py.File(filename, 'w')
        f['image'] = np.concatenate(self._predictions)
        log.info("Dumping prediction done.")

class Evaler(object):

    @staticmethod
    def get_model_class(model_name):
        if model_name == 'MLP':
            from model import Model
        else:
            return ValueError(model_name)
        return Model


    def __init__(self,
                 config,
                 dataset):
        self.config = config
        self.train_dir = config.train_dir
        self.output_file = config.output_file
        log.info("self.train_dir = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        self.dataset = dataset

        check_data_id(dataset, config.data_id)
        _, self.batch = create_input_ops(dataset, self.batch_size,
                                         data_id=config.data_id,
                                         is_training=False,
                                         shuffle=False)

        # --- create model ---
        Model = self.get_model_class(config.model)
        self.model = Model(config)

        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.step_op = tf.no_op(name='step_no_op')

        tf.set_random_seed(1234)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = tf.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver(max_to_keep=100)

        self.checkpoint_path = config.checkpoint_path
        if self.checkpoint_path is None and self.train_dir:
            self.checkpoint_path = tf.train.latest_checkpoint(self.train_dir)
        if self.checkpoint_path is None:
            log.warn("No checkpoint is given. Just random initialization :-)")
            self.session.run(tf.global_variables_initializer())
        else:
            log.info("Checkpoint path : %s", self.checkpoint_path)

    def eval_run(self):
        # load checkpoint
        if self.checkpoint_path:
            self.saver.restore(self.session, self.checkpoint_path)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")

        log.info("# of examples = %d", len(self.dataset))
        length_dataset = len(self.dataset)

        max_steps = self.config.max_steps
        log.info("max_steps = %d", max_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session,
                                               coord=coord, start=True)

        evaler = PoseEvalManager()
        try:
            for s in xrange(max_steps):
                step, step_time, batch_chunk, prediction_pred, prediction_gt = \
                    self.run_single_step(self.batch)
                self.log_step_message(s, step_time)
                evaler.add_batch(batch_chunk['id'], prediction_pred, prediction_gt)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warn(str(e)) # just simply ignore as of now

        if self.config.output_file:
            evaler.dump_result(self.config.output_file)

    def run_single_step(self, batch):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        [step, all_preds, all_targets, _] = self.session.run(
            [self.global_step, self.model.all_preds, self.model.all_targets, self.step_op],
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )

        _end_time = time.time()

        return step, (_end_time - _start_time), batch_chunk, all_preds, all_targets

        def log_step_message(self, step, step_time, is_train=False):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time,
                         )
               )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'SVHN', 'CIFAR10'])
    parser.add_argument('--max_steps', type=int, default=None)
    config = parser.parse_args()

    if config.dataset == 'mnist':
        import mnist_dataset as dataset
    elif config.dataset == 'SVHN':
        import datasets.svhn as dataset
    elif config.dataset == 'CIFAR10':
        import datasets.cifar10 as dataset
    else:
        raise ValueError(config.dataset)

    config.data_info = dataset.get_data_info()
    config.conv_info = dataset.get_conv_info()
    config.deconv_info = dataset.get_deconv_info()
    dataset_train, dataset_test = dataset.create_default_splits()

    evaler = Evaler(config, dataset_test)

    log.warning("dataset: %s", config.dataset)
    evaler.eval_run()

if __name__ == '__main__':
    main()
