import warnings
import shutil
import numpy as np
import torchvision
torchvision.set_image_backend('accimage')
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = 1000000000     # to avoid error "https://github.com/zimeon/iiif/issues/11"
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # to avoid error "https://github.com/python-pillow/Pillow/issues/1510"
import os
import os.path
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import cv2
from util.utils import display
import seaborn as sns
from scipy.stats import wasserstein_distance
import tqdm
import glob
import os


# CLASSNAMES = [
#     "artificial",  # 0
#     "cropland",  # 1
#     "secondary vegetation",  # 2
#     "cleared",  # 3
#     "perennial",  # 4
#     "forest",  # 5
#     "grassland",  # 6
#     "water",  # 7
#     "bare",  # 8
#     "cloud",  # 9
#     "active cropland",  # 10
#     "bare soil"   # 11
# ]  # Class definition from polygon labels

# #original dataset
# CLASSNAMES = [
#     "forest",  # 0
#     "grassland",  # 1
#     "secondary vegetation",  # 2
#     "perennial",  # 3
#     "artificial",  # 4
#     "cleared",  # 5
#     "cropland",  # 6
#     "water",  # 7
#     "bare",  # 8
#     "cloud",  # 9
#     "active cropland",  # 10
#     "bare soil"   # 11
# ]  # Class definition from polygon labels

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

# #WEIGHTS = torch.tensor([.5, 4., 1., 1.5, 1., 2, 1.])   # class weights after putting all background classes to 0, now
# #original_weight_4bands & 5bands
#WEIGHTS = torch.tensor([1., .5, 1., 1., 1., 3., 5.])
#original_weight_5bands_tvt
#WEIGHTS = torch.tensor([1., 1., 1., 1., 2.5, .5, 5.])
#WEIGHTS = torch.tensor([1., 1., 1., 1., 2.5, .5, 5.])
#S1S2_weight_5bands_tvt
#WEIGHTS = torch.tensor([1., 1., 1., 1.5, 2., .5, 3.])
#original_add_c weight
#WEIGHTS = torch.tensor([1.,.5, 1., 1., 1., 2., 3.])
#S1S2S3_weight_5bands_tvt
#WEIGHTS = torch.tensor([2., 1., 1., 1.5, 2., .5, 3.])
#latest balanced weight for S1
#WEIGHTS = torch.tensor([12.9, 1.3, 1.4, 2.4, 9.9, 0.4, 20.3])
#latest balanced weight for S1S2
#WEIGHTS = torch.tensor([5., 1.3, 1.1, 2.1, 6.2, 0.3, 9.9])
#latest balanced weight for S1S2S3
#WEIGHTS = torch.tensor([6.4, 1.1, 0.9, 2.0, 3.8, 0.3, 7.4])
#latest balanced weight for S1S2S3S4
#WEIGHTS = torch.tensor([5.2, 1.1, 0.9, 1.8, 3.7, 0.3, 6.4])
#latest balanced weight for S1_tvt
#WEIGHTS = torch.tensor([16.9, 1.5, 1.7, 2.8, 10.9, 0.4, 24.4])
#latest balanced weight for S1S2_tvt
#WEIGHTS = torch.tensor([7.6, 1.4, 1.2, 2.5, 6.2, 0.4, 11.8])
#latest balanced weight for S4_tvt
#WEIGHTS = torch.tensor([5.5, 1.2, 1.0, 2.0, 2.7, 0.5, 6.3])
#latest balanced weight for S3_tvt
#WEIGHTS = torch.tensor([5.8, 1.3, 1.0, 2.8, 4.1, 0.5, 6.4])
#latest balanced weight for S2_tvt
#WEIGHTS = torch.tensor([5.8, 1.4, 1.1, 2.2, 3.4, 0.5, 7.1])
#latest balanced weight for S1_tvt
#WEIGHTS = torch.tensor([7., 1.5, 1.5, 2.3, 4.2, 0.5, 12.1])
#latest balanced weight for S1_tvt
#WEIGHTS = torch.tensor([3.0, 1.0, 1.0, 1.8, 3.5, 0.5, 7.0])
#latest balanced weight for S1_tvt
#WEIGHTS = torch.tensor([12.6, 1.3, 1.5, 2.4, 10.1, 0.5, 20.])
#latest balanced weight for S1s2_tvt
#WEIGHTS = torch.tensor([5.1, 1.2, 1., 2.1, 5.8, 0.1, 9.1])
#latest balanced weight for S1s2s3_tvt
#WEIGHTS = torch.tensor([5.0, 1.2, 1., 2.0, 4.2, 0.5, 7.5])
#latest balanced weight for S1s2s3s4_tvt
#WEIGHTS = torch.tensor([4.8, 1.1, 1., 1.8, 3.7, 0.5, 6.7])
#latest balanced weight for S1s2s3s4_tvt
WEIGHTS = torch.tensor([0.6, 0.8, 1.8, 0.1])
#latest balanced weight for S1s3s4_tvt
#WEIGHTS = torch.tensor([0.8, 1.1, 2.0, 0.4])
#latest balanced weight for S1s3s4_tvt
#WEIGHTS = torch.tensor([0.8, 1.1, 2.0, 1.])

classes_to_ignore = [0, 3, 4, 6, 7, 8, 9, 11, 10]  # put here the class indexes in CLASSNAMES that will be assigned to background
#classes_to_ignore = [3, 7, 8, 9, 11]

#classes_to_merge = {}
classes_to_merge = {10: 4}
#classes_to_merge = {10: 5}# put here the class indexes in CLASSNAMES that will be merged to the value class
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


class PeruDataset(BaseDataset):
    def __init__(self, opt=None, mode="train"):
        #self.dataroot = "/mnt/warehouse/experiments/perennial_fallow_forest/a_lastex/input_data/test_temp"
        self.dataroot = "/mnt/warehouse/experiments/perennial_fallow_forest/a_lastex/input_data/sep22/all_4bands_classes"
        #self.dataroot = '/mnt/warehouse/experiments/perennial_fallow_forest/a_lastex/input_data/sep22/S1S3S4_4bands_classes'
        #self.dataroot = '/mnt/warehouse/experiments/perennial_fallow_forest/input_data/s1s2s3s4'
        #self.dataroot = "/home/wanting/input_data_4_bands_original"
        #self.split_root = os.path.join(self.dataroot, "all")
        self.split_root = os.path.join(self.dataroot, 'all')
        self.mode = mode
        super().__init__(opt)

        #normalizer for bands
        #5bands(4bands+DEM)
        # self.normalizer = transforms.Normalize(mean=(58.1, 109.0, 70.8, 135.6, 1028.0),
        #                                      std=(26.2, 46.1, 41.1, 45.4, 598.0))
        #4bands
        self.normalizer = transforms.Normalize(mean=(58.1, 109.0, 70.8, 135.6),
                                               std=(26.2, 46.1, 41.1, 45.4))
        # self.normalizer = transforms.Normalize(mean=(58.1, 109.0, 70.8, 135.6, 0, 0, 0, 0, 0, 0),
        #                                        std=(26.2, 46.1, 41.1, 45.4, 1, 1, 1, 1, 1, 1))
        #6bands(4bands+slope+aspect)
        # self.normalizer = transforms.Normalize(mean=(58.1, 109.0, 70.8, 135.6, 42.4, 176.2),
        #                                         std=(26.2, 46.1, 41.1, 45.4, 22.0, 104.9))


        self.cropper = A.Compose([
            #A.RandomCrop(width=opt.imsize, height=opt.imsize),
            A.RandomResizedCrop(width=opt.imsize, height=opt.imsize, scale=(.8, 1.05)),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
        ])
        self.augmenter = A.Compose([
            A.Downscale(0.5, 0.8, interpolation=cv2.INTER_LINEAR),
            A.RandomBrightnessContrast(),
            A.Sharpen(alpha=(.05, .1)),
            A.GaussianBlur(),
            #A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(),
        ])

        self.num_classes = len(MODEL_CLASSNAMES)

    def __getitem__(self, index):
        img, target, valid = self.get_frame()
        #display(valid)
        target[~valid] = 255
        cropped = self.cropper(image=img.transpose(1, 2, 0), mask=target)

        if self.mode == "train":
            transformed = self.augmenter(image=cropped["image"], mask=cropped["mask"])
            im = self.normalizer(transformed["image"].float())
            target = class_mapper[transformed["mask"]]
        elif self.mode == "val":
            im = self.normalizer(torch.from_numpy(cropped["image"].transpose(2, 0, 1)).float())
            target = class_mapper[cropped["mask"]]

        rawim = cropped["image"].transpose(2, 0, 1)
        valid = target != 7
        return im, target, rawim, valid



    def load_data(self):
        """ Loads dataset from disk. This method will save a .pkl dataset file to the DATAROOT folder the first time
        the dataset is used and directly read it next time, to avoid systematic parsing.
        """
        self.files = []
        all_files = sorted(glob.glob(os.path.join(self.split_root, "*.npy")))
        # self.source_paths = sorted(glob.glob(os.path.join(self.source_dir, "annotation_*.png")))
        dropped_frames = 0
        for f in all_files:
            frame = np.load(f)
            # only keep frames bigger than imsize
            if frame.shape[1] >= self.opt.imsize and frame.shape[2] >= self.opt.imsize:
                self.files.append(f)
            else:
                dropped_frames += 1
        warnings.warn(f"{dropped_frames} frame will be ignored because they are smaller than the requested patch size")
        return

    def __len__(self):
        # return len(self.raster_paths)
        if self.mode == "train":
            return 10000
        else:
            return 200

    def print_stats(self):
        train_files = glob.glob(os.path.join(self.dataroot, "train", "*.npy"))
        val_files = glob.glob(os.path.join(self.dataroot, "val", "*.npy"))
        test_files = glob.glob(os.path.join(self.dataroot, "test", "*.npy"))

        fig, axs = plt.subplots(3)
        classcounts = np.zeros(len(CLASSNAMES) - len(classes_to_ignore) + 1)
        for tf in train_files:
            target = np.load(tf)[-2].astype(int)
            valid = target[target != -1]
            unique, counts = np.unique(class_mapper[valid], return_counts=True)
            classcounts[unique] += counts
        for i, c in enumerate(classcounts):
            print(i, c)
        sns.barplot(x=np.arange(classcounts.shape[0]), y=classcounts, ax=axs[0]).set(title='Train set class distribution')

        classcounts = np.zeros(len(CLASSNAMES) - len(classes_to_ignore) + 1)
        for vf in val_files:
            target = np.load(vf)[-2].astype(int)
            valid = target[target != -1]
            unique, counts = np.unique(class_mapper[valid], return_counts=True)
            classcounts[unique] += counts
        for i, c in enumerate(classcounts):
            print(i, c)
        classcounts_train = classcounts / classcounts.sum()
        sns.barplot(x=np.arange(classcounts.shape[0]), y=classcounts, ax=axs[1]).set(title='Val set class distribution')

        classcounts = np.zeros(len(CLASSNAMES) - len(classes_to_ignore) + 1)
        for ttf in test_files:
            target = np.load(ttf)[-2].astype(int)
            valid = target[target != -1]
            unique, counts = np.unique(class_mapper[valid], return_counts=True)
            classcounts[unique] += counts
        for i, c in enumerate(classcounts):
            print(i, c)
        classcounts_train = classcounts / classcounts.sum()
        sns.barplot(x=np.arange(classcounts.shape[0]), y=classcounts, ax=axs[2]).set(title='Test set class distribution')
        plt.show()
        #exit()

    def find_train_val_split(self):
        all_files = glob.glob(os.path.join(self.dataroot, "val", "*.npy"))

        best_wd = 1000
        # load all classcounts in memory
        classcounts = []
        for f in all_files:
            target = np.load(f)[-2].astype(int)
            valid = target[target != -1]
            unique, counts = np.unique(class_mapper[valid], return_counts=True)
            cc = np.zeros(len(MODEL_CLASSNAMES))
            cc[unique] = counts
            classcounts.append(cc)
        classcounts = np.array(classcounts)
        for i in range(100000):
            shuffling_idxs = np.random.permutation(np.arange(classcounts.shape[0]))
            train_idxs, val_idxs = np.split(shuffling_idxs, [int(.5 * shuffling_idxs.shape[0])])
            train, val = classcounts[train_idxs], classcounts[val_idxs]
            traindist = train.sum(0)
            traindist = traindist / traindist.sum()
            valdist = val.sum(0)
            valdist = valdist / valdist.sum()
            wd = wasserstein_distance(traindist, valdist)
            if wd < best_wd:
                best_wd = wd
                print(wd)
                train_files = [all_files[j] for j in train_idxs]
                val_files = [all_files[j] for j in val_idxs]
        for tf in train_files:
            shutil.copy(tf, os.path.join(self.dataroot, "val1"))
        for tf in val_files:
            shutil.copy(tf, os.path.join(self.dataroot, "test"))
        #exit()


if __name__ == "__main__":
    from options.options import Options
    from data import create_dataset

    opt = Options().parse()  # get training options
    dataset = create_dataset(opt, name="peru", mode="train")  # create a dataset given opt.train_dataset_mode and other options

    #browsing the frames for visual inspection
    #all_files = glob.glob(os.path.join(dataset.dataroot, "4_bands_original", "*.npy"))
    # all_files = glob.glob(os.path.join(dataset.dataroot, "*.npy"))
    # for f in all_files:
    #     array = np.load(f)
    #     display(array[0], to_int=False, hold=True)
    #     display(array[1], to_int=False, hold=True)
    #     display(array[2], to_int=False)

    #dataset.find_train_val_split()
    dataset.print_stats()

    loader = torch.utils.data.DataLoader(
                    dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads), pin_memory=True
                )
    print(len(loader))

    for i, (im, target, rawimg, valid) in tqdm.tqdm(enumerate(loader), total=len(loader)):  # inner loop within one epoch        d = dataset[i*100]

        #print(torch.unique(im, return_counts=True))
        #display(target[7], hold=True)

        display(rawimg[0, :3], overlay=target[0], hold=True, title="raw img", to_int=False)
        display(im[0, 3], overlay=target[0], hold=True, title="nir")
        #display(im[0, 4], hold=True, title="yod", to_int=False)
        #display(im[0, 5], hold=True, title="aspect", to_int=False)
        #display(rawimg[0, :3], overlay=im[0, 4], title="chm+img", to_int=False)
        #print(im[1, 4])

        # display(im[1, :3])
        display(target[0])


    bigarray = []
    nbands = []
    for i, (im, target, rawimg, valid) in tqdm.tqdm(enumerate(loader)):
        print(im.shape)
        bigarray.append(im)

    bigarray = torch.cat(bigarray, dim=0)
    print(bigarray.shape)
    for b in range(bigarray.shape[1]):
        print(f"band {b} has min {torch.min(bigarray[:, b])}, max {torch.max(bigarray[:, b])}, mean {torch.mean(bigarray[:, b])}, std {torch.std(bigarray[:, b])}")
