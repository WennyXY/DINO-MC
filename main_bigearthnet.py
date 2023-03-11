# Based on SeCo: https://github.com/ServiceNow/seasonal-contrast
import os
import argparse
import json
from pathlib import Path
import datetime
import time
import sys

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from PIL import Image
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

from sklearn.metrics import average_precision_score

import utils.utils as utils
import utils.vision_transformer as vits
from optim_factory import create_optimizer_v2, optimizer_kwargs
from utils.data import random_subset, LMDBDataset, InfiniteDataLoader
from data_process.bigearthnet_dataset import Bigearthnet

def land_use_classify(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        pretrain_model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=args.num_labels)
        embed_dim = pretrain_model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        pretrain_model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=args.num_labels)
        embed_dim = pretrain_model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        if 'swin' in args.arch:
            pretrain_model = torchvision_models.__dict__[args.arch](num_classes=args.num_labels)
            embed_dim = pretrain_model.head.weight.shape[1]
        else:
            pretrain_model = torchvision_models.__dict__[args.arch](num_classes=args.num_labels)
            embed_dim = pretrain_model.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)

    pretrain_model.cuda()

    utils.load_pretrained_weights(pretrain_model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Pretrained Model {args.arch} built.")
    
    # ============ preparing data ... ============
    train_transforms = pth_transforms.Compose([
        pth_transforms.Resize((224, 224), interpolation=Image.Resampling.BICUBIC),
        pth_transforms.ToTensor()
    ])
    
    if args.lmdb:
        train_dataset = LMDBDataset(
            lmdb_file=os.path.join(args.data_path, 'train.lmdb'),
            transform=train_transforms
        )
        val_dataset = LMDBDataset(
            lmdb_file=os.path.join(args.data_path, 'val.lmdb'),
            transform=train_transforms
        )
    else:
        train_dataset = Bigearthnet(
            root=args.data_path,
            split='train',
            bands=None,
            transform=train_transforms
        )
        val_dataset = Bigearthnet(
            root=args.data_path,
            split='val',
            bands=None,
            transform=train_transforms
        )
    seed = 42
    if args.train_frac is not None and args.train_frac < 1:
        train_dataset = random_subset(train_dataset, args.train_frac, seed)
        val_dataset = random_subset(val_dataset, args.train_frac, seed)
 
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = InfiniteDataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = InfiniteDataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True
    )
    print(f"Data loaded with {len(train_dataset)} train and {len(val_dataset)} val imgs.")
    if 'adamw' in args.opt:
        print('using adamW optimizer')
        optimizer = torch.optim.AdamW(
            pretrain_model.parameters(),
            lr=args.lr
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)
    elif 'adam' in args.opt:
        print('using adam optimizer')
        optimizer = torch.optim.Adam(
            pretrain_model.parameters(),
            lr=args.lr
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)
    elif 'sgd' in args.opt:
        print('using sgd optimizer')
        optimizer = torch.optim.SGD(
            pretrain_model.parameters(),
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
            momentum=0.9,
            weight_decay=0, # we do not apply weight decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    
    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    
    # load weights to evaluate
    if args.evaluate:
        print('Pretrained weights found at ', args.pretrained_weights)
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "luc_checkpoint_best.pth.tar"),
            run_variables=to_restore,
            state_dict=pretrain_model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        test_stats = validate_network(val_loader, pretrain_model, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc']:.1f}%")
        return
    
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "luc_checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=pretrain_model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    
    start_time = time.time()
    print("Starting finetuning on bigearthnet!")
    print('start_epoch=',start_epoch, 'args.epochs=',args.epochs)
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(pretrain_model, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, pretrain_model, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(val_dataset)} test images: {test_stats['acc']:.1f}%")
            best_acc = max(best_acc, test_stats["acc"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
            if best_acc <= test_stats["acc"] and utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": pretrain_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": best_acc,
                }
                torch.save(save_dict, os.path.join(args.output_dir, "luc_checkpoint_best.pth.tar"))
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": pretrain_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            if epoch > 0:
                torch.save(save_dict, os.path.join(args.output_dir, "luc_checkpoint.pth.tar"))
    
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("Training of the supervised linear classifier on remote sensing images completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))

def train(pretrain_model, optimizer, loader, epoch, n, avgpool):
    pretrain_model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        batch_size = inp.shape[0]
        output = pretrain_model(inp)
        # compute cross entropy loss
        loss = nn.MultiLabelSoftMarginLoss()(output, target)
        # loss = nn.CrossEntropyLoss()(output, target)
        acc = average_precision_score(target.detach().cpu().numpy(), torch.sigmoid(output).detach().cpu(), average='micro') * 100.0
        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def validate_network(val_loader, pretrain_model, n, avgpool):
    pretrain_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = pretrain_model(inp)
        
        loss = nn.MultiLabelSoftMarginLoss()(output, target)
        # loss = nn.CrossEntropyLoss()(output, target)
        acc = average_precision_score(target.cpu(), torch.sigmoid(output).detach().cpu(), average='micro') * 100.0
        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc, losses=metric_logger.loss))
    # print(confusion_matrix)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=1e-5, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=19, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    
    parser.add_argument('--train_frac', type=float, default=1)
    # finetuning
    parser.add_argument('--tuning_mode', default=None, type=str,
                    help='Method of fine-tuning (default: None')
    # Optimizer parameters (ssf)
    parser.add_argument('--milestones', type=int, nargs='*', default=[60, 80])
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=2e-5,
                        help='weight decay (default: 2e-5)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--layer_decay', type=float, default=None,
                        help='layer-wise learning rate decay (default: None)')
    
    parser.add_argument('--lmdb', action='store_true')

    args = parser.parse_args()
    land_use_classify(args)