import torch
from thop import profile

from config import config
from oem_lightweight.evaluator import SegEvaluator
from oem_lightweight.utils import prepare_data
from oem_lightweight.model import sparsemask, fasterseg


def main(args):
    # load network
    if args.model.lower() == "sparsemask":
        model = sparsemask(mask=args.arch, weights=args.pretrained_weights)
    elif args.model.lower() == "fasterseg":
        model = fasterseg(arch=args.arch, weights=args.pretrained_weights)

    # check number of parameters and flops
    flop, params = profile(model["model"], inputs=(torch.randn(1, 3, 1024, 1024),), verbose=False)
    print("Params = %fMB, FLOP = %fGB" % (params / 1e6, flop / 1e9))

    # prepare the data
    image = args.image_file
    label = args.label_file
    data = prepare_data(img_file=image, label_file=label)

    # init evaluator
    evaluator = SegEvaluator(config, data, model)

    # preform evaluation
    result = evaluator.evaluate()
    # print("Prediction...\n", result)

    # compute class IoUs
    evaluator.compute_metric(result)

    # plot and save image
    evaluator.show_predicted_image(result, only_pred=args.only_pred, save_img=args.save_image, save_dir=args.save_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OEM lightweight models demo")
    # model
    parser.add_argument("--model", type=str, choices=["fasterseg", "sparsemask"])
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--pretrained_weights", type=str, required=True)
    # demon data
    parser.add_argument("--image_file", type=str, default="demo_data/images/houston_16.tif")
    parser.add_argument("--label_file", type=str, default="demo_data/labels/houston_16.tif")
    # show and save prediction
    parser.add_argument("--only_pred", action="store_true", help="save and show only prediction")
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_dir", type=str, help="directory for saving predictions")

    args = parser.parse_args()
    main(args)