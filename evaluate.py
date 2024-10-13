import numpy as np
import warnings
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from models import create_model
import glob
import os
from util.utils import display
from torchmetrics.functional import dice, jaccard_index, confusion_matrix
import tqdm

#original dataset_tvt
CLASSNAMES = [
    "grassland",  # 0
    "secondary vegetation",  # 1
    "perennial",  # 2
    "artificial",  # 3
    "cleared",  # 4
    "forest",  # 5
    "cropland",  # 6
    "water",  # 7
    "bare",  # 8
    "cloud",  # 9
    "active cropland",  # 10
    "bare soil"   # 11

]  # Class definition from polygon labels

#WEIGHTS = torch.tensor([.5, 4., 1., 1.5, 1., 2, 1.])   # class weights after putting all background classes to 0, now
#latest balanced weight for S1S2S3S4_tvt
WEIGHTS = torch.tensor([6.8, 1.2, 1.0, 2.1])

#classes_to_ignore = [3, 7, 8, 9, 11]  # put here the class indexes in CLASSNAMES that will be assigned to background
# #classes_to_ignore = [4, 7, 8, 9, 11]
#
#classes_to_merge = {10: 4}
#classes_to_merge = {10: 5}# put here the class indexes in CLASSNAMES that will be merged to the value class
classes_to_ignore = [0, 3, 4, 6, 7, 8, 9, 11, 10]  # put here the class indexes in CLASSNAMES that will be assigned to background

#classes_to_merge = {}
classes_to_merge = {10: 4}
#classes_to_merge = {10: 5}# put here the class ind
class_mapper = np.zeros(256, dtype=int)
j = 1
MODEL_CLASSNAMES = ["background"]
for i in range(len(CLASSNAMES)):
    if i in classes_to_ignore:
        continue
    elif i in classes_to_merge:
        class_mapper[i] = classes_to_merge[i]
        MODEL_CLASSNAMES[classes_to_merge[i]] = MODEL_CLASSNAMES[classes_to_merge[i]] + " + " + CLASSNAMES[i]
        continue
    else:
        class_mapper[i] = j
        MODEL_CLASSNAMES.append(CLASSNAMES[i])
    j += 1

class_mapper[255] = j

if WEIGHTS.shape[0] != len(MODEL_CLASSNAMES):
    raise ValueError("number of classes and class weights mismatch")
print(f"Class definition:")
print(0, f"background ({ ', '.join([CLASSNAMES[i] for i in classes_to_ignore])})", "- weight =", np.round(WEIGHTS[0].item(), decimals=1))
for i in range(1, len(MODEL_CLASSNAMES)):
    print(i, MODEL_CLASSNAMES[i], " - weight =", np.round(WEIGHTS[i].item(), decimals=1))
print(j, "IGNORE / INVALID")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DictAverager(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = {}

    def update(self, dict):
        for k in dict:
            if k not in self.vals:
                # declare
                self.vals[k] = []
            if isinstance(dict[k], int) or isinstance(dict[k], float):
                self.vals[k].append(dict[k])
            if isinstance(dict[k], list) or isinstance(dict[k], tuple):
                self.vals[k].extend(dict[k])

    def get_avg(self):
        output_dict = {}
        for k in self.vals:
            output_dict[k] = np.round(np.nanmean(self.vals[k]), decimals=4)
        return output_dict


def compute_precision(prediction, target):
    tp = torch.sum(torch.logical_and(prediction, target), dim=1)
    tpnfp = torch.sum(prediction, dim=1)
    tpnfp[tpnfp == 0] = 60000
    return torch.mean(tp / tpnfp)


def compute_recall(prediction, target):
    tp = torch.sum(torch.logical_and(prediction, target), dim=1)
    tpnfn = torch.sum(target, dim=1)
    return torch.mean(tp / tpnfn)


def compute_f1_score(prediction, target):
    tp = torch.sum(torch.logical_and(prediction, target), dim=1)
    fpnfn = torch.sum(torch.logical_xor(prediction, target), dim=1)
    return torch.mean(tp / (tp + 0.5 * (fpnfn)))


def evaluate_wetlands_classification(prediction, target, names):
    cm = confusion_matrix(target, prediction, normalize="true")
    df_cm = pd.DataFrame(cm, index=[i for i in names],
                         columns=[i for i in names])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    plt.xlabel('predicted', fontsize=18)
    plt.ylabel('true', fontsize=18)
    plt.show()
    return


def evaluate_peru_classification(prediction, target):
    cm = confusion_matrix(target, prediction, normalize="true")
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    plt.xlabel('predicted', fontsize=18)
    plt.ylabel('true', fontsize=18)
    plt.show()
    return


def compute_scalars(predicted_mask, target, suffix="raw"):
    res = {
        f"dice_{suffix}": dice(predicted_mask, target, num_classes=NUM_CLASSES, average="micro",
                               ignore_index=NUM_CLASSES - 1).cpu().item(),
        f"jaccard_{suffix}": jaccard_index(predicted_mask, target, num_classes=NUM_CLASSES, average="micro",
                                           task="multiclass", ignore_index=NUM_CLASSES - 1).cpu().item()
    }
    if suffix == "raw":
        jaccard = res["jaccard_raw"]
        Ddice = res["dice_raw"]
    dice_per_class = dice(predicted_mask, target, average=None, num_classes=NUM_CLASSES,
                          ignore_index=NUM_CLASSES - 1).cpu().tolist()
    jaccard_per_class = jaccard_index(predicted_mask, target, average=None, num_classes=NUM_CLASSES, task="multiclass",
                                      ignore_index=NUM_CLASSES - 1).cpu().tolist()
    cm = confusion_matrix(predicted_mask, target, num_classes=NUM_CLASSES, task="multiclass", ignore_index=NUM_CLASSES - 1)
    print(cm)
    for i, d in enumerate(dice_per_class):
        res[f"dice_{i}_{suffix}"] = d
    for i, d in enumerate(jaccard_per_class):
        res[f"jaccard_{i}_{suffix}"] = d
    return res


if __name__ == "__main__":

    from options.options import Options
    from data import create_dataset

    opt = Options().parse()  # get training options
    dataset = create_dataset(opt, name="peru_test",
                             mode="train")  # create a dataset given opt.train_dataset_mode and other options
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=int(opt.num_threads), pin_memory=True
    )
    print(len(loader))

    # for i, (im, target, rawimg, valid) in tqdm.tqdm(enumerate(loader), total=len(loader)):

    NUM_CLASSES = 8
    # browsing the frames for visual inspection
    # all_files = glob.glob(os.path.join(dataset.dataroot, "4_bands_original", "*.npy"))
    # all_files = glob.glob(os.path.join(dataset.dataroot, "*.npy"))
    # for f in all_files:
    #     array = np.load(f)
    #     array[1][array[1] == -1] = 7
    #     prediction = array[0]
    #     target = array[1]
    for i, (im, target, valid) in enumerate(loader):  # inner loop within one epoch        d = dataset[i*100]
        #print(torch.unique(im, return_counts=True))
        #display(target[7], hold=True)
        res = compute_scalars(im, target, suffix="raw")
        print(res)
        prec = compute_precision(im, target)
        recall = compute_recall(im, target)
        f1 = compute_f1_score(im, target)
        print(prec, recall, f1)

    # dummy_input = torch.Tensor(((0.1, 0.1, 0.1), (0.4, 0.1, 0.2)))
    # dummy_target = torch.Tensor(((1, 0, 1), (0, 1, 0)))

    # prec = compute_precision(dummy_input > 0.5, dummy_target)
    # recall = compute_recall(dummy_input > 0.5, dummy_target)
    # f1 = compute_f1_score(dummy_input > 0.5, dummy_target)
    # print(prec, recall, f1)
    #
    # prec2 = precision_score(dummy_target, dummy_input > 0.5, average="samples", zero_division=0)
    # recall2 = recall_score(dummy_target, dummy_input > 0.5, average="samples")
    # f12 = f1_score(dummy_target, dummy_input > 0.5, average="samples")
    # print(prec2, recall2, f12)
