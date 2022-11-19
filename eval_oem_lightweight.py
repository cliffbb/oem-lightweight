import os
import torch
from thop import profile
from config import config
from evaluation_tools.evaluator import SegEvaluator
from evaluation_tools.utils import prepare_data
from oem_lightweight import sparsemask, fasterseg


def get_data_files(data_dir, test_split):
    import random
    with open(test_split, 'r') as f:
        test_files = f.readlines()

    img, label = random.choice(test_files).strip().split(" ")
    img = os.path.join(data_dir, img)
    label = os.path.join(data_dir, label)

    return img, label


data_dir = "/home/cliffbb/OpenEarthMap/LULC-RIKEN-Integrated/"
test_split = "/home/cliffbb/OpenEarthMap/LULC-RIKEN-Integrated/split_files/test_fns.txt"
img_file, label_file = get_data_files(data_dir, test_split)
data = prepare_data(img_file, label_file)


def main():
    # load network
    # model = sparsemask("models/SparseMask/mask_thres_0.001.npy", "models/SparseMask/checkpoint_63750.pth.tar")
    model = fasterseg("models/FasterSeg/arch_1.pt", "models/FasterSeg/weights1.pt")

    # check number of parameters and flops
    flops, params = profile(model["model"], inputs=(torch.randn(1, 3, 1024, 1024),), verbose=False)
    print("params = %fMB, FLOPs = %fGB" % (params / 1e6, flops / 1e9))

    # init evaluator
    evaluator = SegEvaluator(config, data, model)

    # preform evaluation
    result = evaluator.evaluate()
    # print("Prediction...\n", result)

    # compute class IoUs
    evaluator.compute_metric(result)

    # plot and save image
    evaluator.show_predicted_image(result, only_pred=False, save_img=True, save_dir="results")


if __name__ == "__main__":
    main()