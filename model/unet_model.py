import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from models.base_model import BaseModel
import torch.nn.functional as F
from models import create_model
import segmentation_models_pytorch as smp
import math
from layers.loss import FocalLoss, TverskyLoss, DiceLoss
from util.post_processing import DenseCRF
from torchmetrics.functional import dice, jaccard_index
import numpy as np
from torch.nn import CrossEntropyLoss
from data.peru_dataset import WEIGHTS
import ipdb

NUM_CLASSES = WEIGHTS.shape[0] + 1
CMAP = torch.Tensor([
    (255, 255, 255),    # 0, WHITE
    (0, 255, 0),        # 1, GREEN
    (0, 0, 255),        # 2, BLUE
    (255, 0, 0),        # 3, RED
    (255, 102, 102),    # 4, SALMON
    (102, 102, 255),    # 5, PURPLE
    (102, 255, 102),    # 6, LIGHT GREEN
    (255, 255, 0),      # 7, YELLOW
    (0, 255, 255),
    (255, 0, 255),
    (102, 102, 102),
    (255, 255, 204),
]).float() / 255


class UnetModel(BaseModel, torch.nn.Module):
    """
    Unet model relying on segmentation_models_pytorch library to allow loading pretrained segmentation model such as
    ResNet50 pretrained on ImageNet.
    """
    def __init__(self, opt, mode):
        BaseModel.__init__(self, opt)
        torch.nn.Module.__init__(self)
        self.mode = mode
        print(f"## Unet model")
        self.loss_names = ['pixel', 'region', 'total']
        self.scalar_names = ["jaccard", "dice"]
        self.model_names = ['unet']
        self.visual_names = ["visual_target", "visual_pred"]

        if opt.ckpt == "imagenet":
            self.unet = smp.Unet("resnet50", in_channels=3, encoder_weights='imagenet', classes=NUM_CLASSES-1)
            self.unet.encoder.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif opt.ckpt == "None":
            self.unet = smp.Unet("resnet50", in_channels=3, encoder_weights=None, classes=NUM_CLASSES-1)
            self.unet.encoder.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif opt.ckpt == "bigearthnet":
            self.unet = smp.Unet("resnet50", in_channels=4, encoder_weights='imagenet', classes=NUM_CLASSES-1)
            bigearthnet_ckpt_path = "/mnt/warehouse/experiments/perennial_fallow_forest/test/model/tfs_4bands/latest_net_feature_extractor.pth"
            #bigearthnet_ckpt_path = "/mnt/warehouse/experiments/perennial_fallow_forest/test/model/0328_1025/latest_net_unet.pth"

            bigearthnet_ckpt = torch.load(bigearthnet_ckpt_path)
            ipdb.set_trace()
            bigearthnet_ckpt = {x.replace("module.0", "conv1"): y for x, y in bigearthnet_ckpt.items()}
            bigearthnet_ckpt = {x.replace("module.1", "bn1"): y for x, y in bigearthnet_ckpt.items()}
            bigearthnet_ckpt = {x.replace("module.4", "layer1"): y for x, y in bigearthnet_ckpt.items()}
            bigearthnet_ckpt = {x.replace("module.5", "layer2"): y for x, y in bigearthnet_ckpt.items()}
            bigearthnet_ckpt = {x.replace("module.6", "layer3"): y for x, y in bigearthnet_ckpt.items()}
            bigearthnet_ckpt = {x.replace("module.7", "layer4"): y for x, y in bigearthnet_ckpt.items()}
            self.unet.encoder.load_state_dict(bigearthnet_ckpt, strict=False)
            #self.unet.encoder.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.unet.encoder.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        ipdb.set_trace()
        self.unet.to(self.device)

        self.criterion_region = DiceLoss(mode="multiclass", classes=list(range(NUM_CLASSES-1)), ignore_index=NUM_CLASSES-1, weight=WEIGHTS.to(self.device))
        self.criterion_pixel = CrossEntropyLoss(weight=WEIGHTS.to(self.device), ignore_index=NUM_CLASSES-1, reduction="none")

        self.optimizers["main"] = torch.optim.Adam(list(self.unet.parameters()), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)

        # self.crf = DenseCRF(max_iter=10, pos_xy_std=25, bi_xy_std=80, bi_rgb_std=15)
        self.crf = DenseCRF(max_iter=20, pos_xy_std=30, bi_xy_std=50, bi_rgb_std=9)  # picked 25/01 with parameter sweep

    def forward(self, img):
        self.heatmap = self.unet(img.to(self.device))
        self.probs = torch.softmax(self.heatmap, dim=1)
        self.predicted_mask = torch.argmax(self.probs, dim=1)
        self.visual_pred = CMAP[self.predicted_mask.detach().cpu()].permute(0, 3, 1, 2)
        return self.predicted_mask

    def __call__(self, img):
        print(img.shape)
        self.forward(img.float())
        # return self.probs, self.predicted_mask.unsqueeze(1)
        # refined_probs = np.zeros(
        #     (self.predicted_mask.shape[0], NUM_CLASSES-1, self.predicted_mask.shape[1], self.predicted_mask.shape[2]))
        # for i in range(self.predicted_mask.shape[0]):
        #     refined_probs[i] = self.crf.forward(rawimg[i, :3], self.probs[i])
        #refined_probs = self.crf.batch_forward(rawimg[:, :3], self.probs.cpu(), num_threads=16)
        #refined_mask = torch.argmax(torch.from_numpy(refined_probs), dim=1)
        # # print(torch.amax(refined_mask, dim=(1, 2)))
        #return torch.from_numpy(refined_probs), refined_mask.unsqueeze(1)
        print(self.probs.shape)
        return self.probs #, self.predicted_mask.unsqueeze(1)

    def compute_loss(self, target, valid):
        self.visual_target = CMAP[target.detach().cpu()].permute(0, 3, 1, 2)
        self.loss_region = self.criterion_region(self.heatmap, target)
        self.loss_pixel = (valid * self.criterion_pixel(self.heatmap, target)).mean([1, 2]).mean()
        self.loss_total = self.opt.alpha * self.loss_region + (1 - self.opt.alpha) * self.loss_pixel
        return self.loss_total

    def optimize_parameters(self, target, valid):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.compute_loss(target.to(self.device), valid.to(self.device))
        for name in self.optimizers:
            self.optimizers[name].zero_grad()
        self.loss_total.backward()
        for name in self.optimizers:
            self.optimizers[name].step()
        return

    def compute_scalars(self, predicted_mask, target, suffix="raw"):
        res = {
            f"dice_{suffix}": dice(predicted_mask, target, num_classes=NUM_CLASSES, average="weighted", ignore_index=NUM_CLASSES-1).cpu().item(),
            f"jaccard_{suffix}": jaccard_index(predicted_mask, target, num_classes=NUM_CLASSES, average="weighted", task="multiclass", ignore_index=NUM_CLASSES-1).cpu().item()
        }
        if suffix == "raw":
            self.jaccard = res["jaccard_raw"]
            self.dice = res["dice_raw"]
        dice_per_class = dice(predicted_mask, target, average=None, num_classes=NUM_CLASSES, ignore_index=NUM_CLASSES-1).cpu().tolist()
        jaccard_per_class = jaccard_index(predicted_mask, target, average=None, num_classes=NUM_CLASSES, task="multiclass", ignore_index=NUM_CLASSES-1).cpu().tolist()
        for i, d in enumerate(dice_per_class):
            res[f"dice_{i}_{suffix}"] = d
        for i, d in enumerate(jaccard_per_class):
            res[f"jaccard_{i}_{suffix}"] = d
        return res

    def val(self, img, target, rawimg, valid):
        with torch.no_grad():
            predicted_mask = self.forward(img)
            results = self.compute_scalars(predicted_mask, target.to(self.device))
            self.compute_loss(target.to(self.device), valid.to(self.device))
            results["loss_pixel"] = self.loss_pixel.cpu().item()
            results["loss_region"] = self.loss_region.cpu().item()
            results["loss_total"] = self.loss_total.cpu().item()

            # apply CRF
            # refined_probs = np.zeros((self.predicted_mask.shape[0], NUM_CLASSES-1, self.predicted_mask.shape[1], self.predicted_mask.shape[2]))
            # for i in range(self.predicted_mask.shape[0]):
            #     refined_probs[i] = self.crf.forward(rawimg[i, :3], self.probs[i])
            # refined_probs = self.crf.batch_forward(rawimg[:, :3], self.probs.cpu())
            #
            # refined_mask = torch.argmax(torch.from_numpy(refined_probs), dim=1)
            # results_after_crf = self.compute_scalars(refined_mask, target, suffix="crf")
            # results = {**results, **results_after_crf}
        return results

    def test(self, dataset):
        pass


if __name__ == '__main__':
    from options.options import Options
    from data import create_dataset
    from util.evaluate import DictAverager
    import random
    import imgaug
    import numpy as np

    opt = Options()
    opt = opt.parse()
    model = create_model(opt, mode='test')
    model.setup(opt)
    model.eval()

    best_dice = 0

    dataset = create_dataset(opt, name=opt.train_dataset, mode='val')
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads), pin_memory=True
    )

    averager = DictAverager()

    random.seed(42)
    imgaug.random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    for i, (im, target, rawimg, valid) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
        model.forward(im)
        results = model.val(im, target, rawimg, valid)
        averager.update(results)
    res_str = f"{averager.get_avg()}"
    print(res_str)

