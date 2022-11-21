import torch
import numpy as np

from config import config
from fasterseg_api.model_seg import Network_Multi_Path_Infer as Network
from sparsemask_api.sparse_mask_eval_mode import SparseMask


def fasterseg(arch, weights):
    # load arch
    state = torch.load(arch, map_location="cpu")
    model = Network([state["alpha_1_0"].detach(),
                     state["alpha_1_1"].detach(),
                     state["alpha_1_2"].detach()],
                    [None, state["beta_1_1"].detach(),
                     state["beta_1_2"].detach()],
                    [state["ratio_1_0"].detach(),
                     state["ratio_1_1"].detach(),
                     state["ratio_1_2"].detach()],
                    num_classes=config.num_classes,
                    layers=16,
                    Fch=12,
                    width_mult_list=[4. / 12, 6. / 12, 8. / 12, 10. / 12, 1., ],
                    stem_head_width=(8. / 12, 8. / 12),
                    ignore_skip=False)
    model.build_structure([2, 1])

    # load weights
    weights_dict = torch.load(weights, map_location="cpu")
    state = model.state_dict()
    weights_dict = {k: v for k, v in weights_dict.items() if k in state}
    state.update(weights_dict)
    model.load_state_dict(state)

    return dict(model=model, name="FasterSeg")


def sparsemask(mask, weights):
    # load arch
    mask = np.load(mask)
    model = SparseMask(mask,
                       backbone_name="mobilenet_v2",
                       depth=64,
                       in_channels=3,
                       num_classes=config.num_classes)

    # load weight
    weights_dict = torch.load(weights, map_location="cpu")
    weights_dict = {key.replace("module.", ""): value for key, value in weights_dict['state_dict'].items()}
    model.load_state_dict(weights_dict, strict=False)

    return dict(model=model, name="SparseMask")



