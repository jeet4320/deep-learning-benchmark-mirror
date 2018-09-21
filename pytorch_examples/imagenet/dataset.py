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

import numpy as np
import torch.utils.data as data


class SyntheticDataset(data.Dataset):
    def __init__(self, shape, num_classes):
        self._shape = shape
        self._x = (np.random.random(shape) * 255.0).astype(np.float32)
        self._y = np.random.randint(0, num_classes, (shape[0],))

    def __getitem__(self, item):
        return self._x[item], self._y[item]

    def __len__(self):
        return self._shape[0]
