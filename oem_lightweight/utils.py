import numpy as np
import cv2
import os

np.seterr(divide='ignore', invalid='ignore')


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _set_img_color(colors, background, img, gt, show255=False, weight_foreground=0.55):
    origin = np.array(img)
    for i in range(len(colors)):
        if i != background:
            img[np.where(gt == i)] = colors[i]
    if show255:
        img[np.where(gt == 255)] = 0
    cv2.addWeighted(img, weight_foreground, origin, (1 - weight_foreground), 0, img)
    return img


def show_prediction(colors, background, img, pred):
    im = np.array(img, np.uint8)
    _set_img_color(colors, background, im, pred, weight_foreground=1)
    final = np.array(im)
    return final


def show_img(colors, background, img, gt, *pds):
    im1 = np.array(img, np.uint8)
    final = np.array(im1)
    # the pivot black bar
    pivot = np.zeros((im1.shape[0], 15, 3), dtype=np.uint8)

    for pd in pds:
        im = np.array(img, np.uint8)
        _set_img_color(colors, background, im, pd, weight_foreground=1)
        final = np.column_stack((final, pivot))
        final = np.column_stack((final, im))

    im = np.array(img, np.uint8)
    _set_img_color(colors, background, im, gt, True, weight_foreground=1)
    final = np.column_stack((final, pivot))
    final = np.column_stack((final, im))
    return final


def print_iou(iu, class_names=None):
    n = iu.size
    lines = []
    lines.append('Class IoU Results')
    lines.append('------------------------------')
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i + 1)
        else:
            cls = '%d %s' % (i + 1, class_names[i].capitalize())
        lines.append('%-16s\t%.2f%%' % (cls, iu[i] * 100))
    mean_IoU = np.nanmean(iu)
    lines.append('------------------------------')
    lines.append('%-16s\t%.2f%%' % ('mean_IoU', mean_IoU * 100,))
    line = "\n".join(lines)
    print(line)


############## voc cityscapes metric #####################
def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))

    return np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                       minlength=n_cl ** 2).reshape(n_cl, n_cl), labeled, correct


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    # freq = hist.sum(1) / hist.sum()
    # freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc


########### data utils ##############
def prepare_data(img_file, label_file):
    item_name = img_file.split("/")[-1].split(".")[0]
    item_name = item_name.replace("_", " ").replace("-", " ")

    img, label = _fetch_data(img_file, label_file)
    label = _encode_segmap(np.array(label, dtype=np.uint8))
    data_dict = dict(img=img, label=label, filename=str(item_name.capitalize()))

    return data_dict


def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
    # cv2: B G R => h w c
    img = np.array(cv2.imread(filepath, mode), dtype=dtype)
    if len(img.shape) == 3:
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    return img


def _fetch_data(img_file, label_file):
    img = _open_image(img_file)
    label = _open_image(label_file, cv2.IMREAD_GRAYSCALE, dtype='uint8')

    return img, label


def _encode_segmap(mask):
    void_classes = [0]
    valid_classes = [1, 2, 3, 4, 5, 6, 7, 8]
    ignore_index = 255
    class_map = dict(zip(valid_classes, range(len(valid_classes))))

    for c in void_classes:
        mask[mask == c] = ignore_index
    for c in valid_classes:
        mask[mask == c] = class_map[c]

    return mask
