#!/usr/bin/env python3
# encoding: utf-8
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms

from utils import *


class SegEvaluator(object):
    def __init__(self, config, data, network):
        self.num_classes = config.num_classes
        self.class_colors = config.class_colors
        self.class_names = config.class_names
        self.use_tta = config.use_tta

        self.data = data
        self.stats = config.stats

        self.model = network["model"]
        self.model_name = network["name"]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def evaluate(self):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            image = self.data["img"]

            # for FasterSeg only
            if self.model_name == "FasterSeg":
                image = image[:, :, ::-1]

            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**self.stats)])
            img = transform(image.copy().astype(np.uint8))
            img = img[None, :, :, :].to(self.device)

            if self.use_tta:
                output = tta(self.model, img)  # use TTA
            else:
                output = self.model(img)

            # for SparseMask only
            if self.model_name == "SparseMask":
                h, w, _ = image.shape
                output = torch.nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)

            output = output[0]
            output = output.permute(1, 2, 0)
            output = output.cpu().numpy()
            output = output.argmax(2)

            return output

    def compute_metric(self, pred):
        label = self.data['label']
        hist, labeled, correct = hist_info(self.num_classes, pred, label)
        iu, mean_IU, _, _ = compute_score(hist, correct, labeled)
        print_iou(iu, self.class_names)
        # result_line = print_iou(iu, self.class_names)
        # return mean_IU, result_line

    def show_predicted_image(self, pred, only_pred=False, fig_size=None, save_img=True, save_dir=None):
        img = self.data["img"]
        label = self.data["label"]
        name = self.data["filename"]

        if only_pred:
            if fig_size is None:
                fig_size = (5, 5)
            comp_img = show_prediction(self.class_colors, -1, img, pred)
        else:
            if fig_size is None:
                fig_size = (16, 5)
            comp_img = show_img(self.class_colors, -1, img, label, pred)

        comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)    # comp_img[:,:,::-1]

        fig = plt.figure(figsize=fig_size)
        plt.imshow(cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.tight_layout()

        if not only_pred:
            ax = plt.gca()
            fig.set_transform(ax.transAxes)
            fig.set_clip_on(False)
            ax.text(0.16, -0.02, 'Image', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
            ax.text(0.5, -0.02, 'Segmentation', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
            ax.text(0.83, -0.02, 'Reference', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)

        # create a patch for every color and use for legend
        patches = [mpatches.Patch(facecolor=np.array(self.class_colors[i]) / 255., label=self.class_names[i], linewidth=3) for i in range(8)]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

        plt.yticks([])
        plt.xticks([])
        plt.show(block=True)

        if save_img:
            if save_dir is not None:
                ensure_dir(save_dir)
                cv2.imwrite(os.path.join(save_dir, name.replace(" ", "_") + ".png"), comp_img)
            else:
                cv2.imwrite(os.path.join(os.path.realpath('.'), name.replace(" ", "_") + ".png"), comp_img)


def tta(model, input):
    score = model(input)
    input_flip = input.flip(-1)
    score_flip = model(input_flip)
    score += score_flip.flip(-1)
    score = torch.exp(score)

    return score

