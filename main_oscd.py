# refer to SeCo: https://github.com/ServiceNow/seasonal-contrast
from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser
import os
import sys
import torch
from torch.nn import BCEWithLogitsLoss
from torchvision.models import resnet
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics.classification import Accuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryJaccardIndex

from data_process.oscd_datamodule import ChangeDetectionDataModule
from models.segmentation import get_segmentation_model
import utils.vision_transformer as vits
from torchvision import models as torchvision_models

import utils.utils as utils
import utils.loss as loss_function

torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

class SiamSegment(LightningModule):

    def __init__(self, backbone, feature_indices, feature_channels, args):
        super().__init__()
        self.model = get_segmentation_model(backbone, feature_indices, feature_channels, args.arch)
        if 'BCE' in args.loss_function:
            self.criterion = BCEWithLogitsLoss()
        elif 'dice' in args.loss_function:
            self.criterion = loss_function.dice_bce_loss()

        self.prec = BinaryPrecision(multidim_average='global',threshold=0.5)
        self.rec = BinaryRecall(threshold=0.5)
        self.f1 = BinaryF1Score(threshold=0.5)
        self.iou = BinaryJaccardIndex(threshold=0.5)
        self.args = args

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1, iou = self.shared_step(batch)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/precision', prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/recall', rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        tensorboard.add_image('train/img_1', img_1[0], global_step)
        tensorboard.add_image('train/img_2', img_2[0], global_step)
        tensorboard.add_image('train/mask', mask[0], global_step)
        tensorboard.add_image('train/out', (pred[0]>=0.2)*1, global_step)
        # print((pred[0]>0.5)*1)
        return loss
        
    def validation_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1, iou = self.shared_step(batch)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/precision', prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recall', rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        assert(len(img_1)==len(img_2)==len(mask)==len(pred))
        for i in range(len(img_1)):
            print(str(i),':',self.cal_f1(pred[i],mask[i]))
            tensorboard.add_image('val/'+str(i)+'/img_1', img_1[i], global_step)
            tensorboard.add_image('val/'+str(i)+'/img_2', img_2[i], global_step)
            tensorboard.add_image('val/'+str(i)+'/mask', mask[i], global_step)
            tensorboard.add_image('val/'+str(i)+'/out'+str(self.cal_f1(pred[i],mask[i])), (pred[i]>=0.2)*1, global_step)
        return loss
    def cal_f1(self,pred, mask):
        f1 = self.f1(pred, mask.long())
        return format(f1, '.4f')
        
    def shared_step(self, batch):
        img_1, img_2, mask = batch
        out = self(img_1, img_2)
        pred = torch.sigmoid(out)

        prec = self.prec(pred, mask.long())
        rec = self.rec(pred, mask.long())
        f1 = self.f1(pred, mask.long())
        iou = self.iou(pred, mask.long())
        
        if 'BCE' in self.args.loss_function:
            loss = self.criterion(out, mask)
        elif 'dice' in self.args.loss_function:
            loss = self.criterion(pred, out, mask)
        return img_1, img_2, mask, pred, loss, prec, rec, f1, iou

    def configure_optimizers(self):
        params = set(self.model.parameters()).difference(self.model.encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]

if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--loss_function', type=str, default='BCEWithLogitsLoss')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--log_name', type=str)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--num_workers_train', type=int, default=0)
    parser.add_argument('--num_workers_val', type=int, default=0)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--trained_weights', type=str, default=None)
    args = parser.parse_args()
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
   
    datamodule = ChangeDetectionDataModule(args.data_path, args.batch_size,num_workers_train=args.num_workers_train,num_workers_val=args.num_workers_val)

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        pretrain_model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = pretrain_model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        pretrain_model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = pretrain_model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        # pretrain_model = torchvision_models.__dict__[args.arch](num_classes=768)
        pretrain_model = torchvision_models.__dict__[args.arch](num_classes=0)
        embed_dim = pretrain_model.fc.weight.shape[1]
        # pretrain_model.fc = nn.Identity()
    elif args.arch == 'supervised_resnet':
        pretrain_model = torchvision_models.__dict__['resnet50'](weights = 'IMAGENET1K_V1')
        embed_dim = 1000
        print(f"Pretrained {args.arch} built.")
    elif args.arch == 'supervised_wide_resnet':
        pretrain_model = torchvision_models.__dict__['wide_resnet50_2'](weights = 'IMAGENET1K_V1')
        embed_dim = 1000
        print(f"Pretrained {args.arch} built.")
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)

    # load weights to evaluate
    if 'supervised' not in args.arch:
        utils.load_pretrained_weights(pretrain_model, args.pretrained_weights, 'teacher', args.arch, args.patch_size)
        print(f"Pretrained Model {args.arch} built.")

    feature_indices=(0, 4, 5, 6, 7)
    feature_channels=(64, 256, 512, 1024, 2048)
    model = SiamSegment(pretrain_model, feature_indices=feature_indices, feature_channels=feature_channels,args = args)
    model.example_input_array = (torch.zeros((1, 3, 96, 96)), torch.zeros((1, 3, 96, 96)))

    logger = TensorBoardLogger(save_dir=args.output_dir, name=args.log_name)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        # every_n_epochs=2,
        monitor="val/f1",
        mode="max",
        filename='{val/f1:.4f}',
        save_weights_only=True
    )
    trainer = Trainer(accelerator='gpu', devices=args.gpus, enable_model_summary = False,check_val_every_n_epoch=1, logger=logger, log_every_n_steps=2, callbacks=[checkpoint_callback], max_epochs=args.epochs)
    if args.evaluate:
        trainer.validate(model, datamodule=datamodule, ckpt_path= args.trained_weights, verbose=True)
    else:
        trainer.fit(model, datamodule=datamodule)