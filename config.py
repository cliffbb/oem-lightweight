# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
from easydict import EasyDict

C = EasyDict()
config = C

# config root_dir and user when u first using
C.repo_name = 'OEM-LightweightModel'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]

# path config
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, "oem_lightweight"))
add_path(osp.join(C.root_dir, "fasterseg_api"))
add_path(osp.join(C.root_dir, "sparsemask_api"))

# image config
C.num_classes = 8
C.stats = {'mean': [0.4325, 0.4483, 0.3879], 'std': [0.0195, 0.0169, 0.0179]}
C.class_names = ["bareland", "rangeland", "developed space", "road", "tree", "water", "agriculture land", "buildings"]
C.class_colors = [[128, 0, 0],
                  [0, 255, 0],
                  [192, 192, 192],
                  [255, 255, 255],
                  [49, 139, 87],
                  [0, 0, 255],
                  [127, 255, 0],
                  [255, 0, 0]]

# eval config
C.use_tta = True # TTA evaluation
