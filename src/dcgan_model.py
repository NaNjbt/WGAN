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

# dcgan Decriminator
class DcganD(nn.Cell):
    def __init__(self, isize, nz, nc, ndf, n_extra_layers=0):
        super(DcganD, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.SequentialCell()
        main.append(nn.Conv2d(nc, ndf, 4, 2, 'pad', 1, has_bias=False))
        main.append(nn.LeakyReLU(0.2))

        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.append(nn.Conv2d(cndf, cndf, 3, 1, 'pad', 1, has_bias=False))
            main.append(nn.BatchNorm2d(cndf))
            main.append(nn.LeakyReLU(0.2))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2

            main.append(nn.Conv2d(in_feat, out_feat, 4, 2, 'pad', 1, has_bias=False))
            main.append(nn.BatchNorm2d(out_feat))
            main.append(nn.LeakyReLU(0.2))

            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.append(nn.Conv2d(cndf, 1, 4, 1, 'pad', 0, has_bias=False))
        self.main = main

    def construct(self, input):
        output = self.main(input)
        output = output.mean(0)
        return output.view(1)


# dcgan generator
class DcganG(nn.Cell):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super(DcganG, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.SequentialCell()
        # input is Z, going into a convolution
        
        main.append(nn.Conv2dTranspose(nz, cngf, 4, 1, 'pad', 0, has_bias=False))
        main.append(nn.BatchNorm2d(cngf))
        main.append(nn.ReLU())

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.append(nn.Conv2dTranspose(cngf, cngf // 2, 4, 2, 'pad', 1, has_bias=False))
            main.append(nn.BatchNorm2d(cngf // 2))
            main.append(nn.ReLU())

            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.append(nn.Conv2d(cngf, cngf, 3, 1, 'pad', 1, has_bias=False))
            main.append(nn.BatchNorm2d(cngf))
            main.append(nn.ReLU())

        main.append(nn.Conv2dTranspose(cngf, nc, 4, 2, 'pad', 1, has_bias=False))
        main.append(nn.Tanh())
        self.main = main

    def construct(self, input):
        output = self.main(input)
        return output
