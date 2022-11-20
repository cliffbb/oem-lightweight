import torch
from thop import profile
from config import config
from evaluation_tools.evaluator import SegEvaluator
from evaluation_tools.utils import prepare_data
from oem_lightweight import sparsemask, fasterseg


# get data files: image and label files
img_file = "demo_data/images/palu_8.tif"
label_file = "demo_data/labels/palu_8.tif"


def main():
    # load network
    model = sparsemask("models/SparseMask/mask_thres_0.001.npy", "models/SparseMask/checkpoint_63750.pth.tar")
    # model = fasterseg("models/FasterSeg/arch_1.pt", "models/FasterSeg/weights1.pt")

    # check number of parameters and flops
    flop, params = profile(model["model"], inputs=(torch.randn(1, 3, 1024, 1024),), verbose=False)
    print("Params = %fMB, FLOP = %fGB" % (params / 1e6, flop / 1e9))

    # prepare the data
    data = prepare_data(img_file, label_file)

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