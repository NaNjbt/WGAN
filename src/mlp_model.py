# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore.nn as nn
from mindspore import context

# MLP Generator
class MlpG(nn.Cell):
    def __init__(self, isize, nz, nc, ngf):
        super(MlpG, self).__init__()
        main = nn.SequentialCell(
            # Z goes into a linear of size: ngf
            nn.Dense(nz, ngf),
            nn.ReLU(),
            nn.Dense(ngf, ngf),
            nn.ReLU(),
            nn.Dense(ngf, ngf),
            nn.ReLU(),
            nn.Dense(ngf, nc * isize * isize),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def construct(self, input):
        input = input.view(input.shape[0], input.shape[1])
        output = self.main(input)
        return output.view(output.shape[0], self.nc, self.isize, self.isize)


# MLP Descriminator
class MlpD(nn.Cell):
    def __init__(self, isize, nz, nc, ndf):
        super(MlpD, self).__init__()
        main = nn.SequentialCell(
            # Z goes into a linear of size: ndf
            nn.Dense(nc * isize * isize, ndf),
            nn.ReLU(),
            nn.Dense(ndf, ndf),
            nn.ReLU(),
            nn.Dense(ndf, ndf),
            nn.ReLU(),
            nn.Dense(ndf, 1),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def construct(self, input):
        input = input.view(input.shape[0], input.shape[1]*input.shape[2]*input.shape[3])
        output = self.main(input)
        output = output.mean(0)
        return output.view(1)
