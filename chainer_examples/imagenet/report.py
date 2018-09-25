# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import time

import chainer


class MetricsReport(chainer.training.extension.Extension):
    def __init__(self, parallelism, dataset_length):
        self._initialized = False
        self.parallelism = parallelism
        self.dataset_length = dataset_length
        self.epoch_start = None
        self.batch_start = None
        self.batch_size = None

    def __call__(self, trainer):

        now = time.time()

        # init times
        if not self._initialized:
            self.epoch_start = now - trainer.elapsed_time
            self.batch_start = self.epoch_start
            self.batch_size = trainer.updater.get_iterator('main').batch_size
            self._initialized = True

        observation = trainer.observation

        #if 'main/accuracy' in observation:
        #    print('Epoch: [{}] \t Training_accuracy_pct: {accuracy} \t Batch_speed: {image_persec:.3f} samples/sec'.format(trainer.updater.epoch, 
        #                      accuracy = (float(observation['main/accuracy'].data) * 100.0), image_persec = ((self.batch_size * self.parallelism) / (now - self.batch_start))))

        print('Epoch: [{}] \t Batch_speed: {image_persec:.3f} samples/sec'.format(trainer.updater.epoch, image_persec = 
                          (self.batch_size * self.parallelism) / (now - self.batch_start)))

        self.batch_start = now

        if trainer.updater.is_new_epoch:
            print('Epoch: [{}] \t Speed: {image_persec:.3f} samples/sec \t Time cost={time:.3f}'.format(trainer.updater.epoch, image_persec = (self.dataset_length / (now - self.epoch_start)), time = (now - self.epoch_start)))
            self.epoch_start = now

