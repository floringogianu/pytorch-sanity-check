""" Adapted from:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import argparse
import os
import random
import shutil
import time
import datetime
import warnings
import sys
import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from datasets import iNaturalist
from utils import FPSMetric, AverageMetric, Logger, parse_cmd


best_acc1 = 0


def main():
    args = parse_cmd()

    # create a run label
    timestamp = f"{datetime.datetime.now():%Y%b%d-%H%M%S}"
    args.run_label = f"{timestamp}_inat_{args.arch}"

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    print(f"MAIN: world_size={args.world_size}")
    print(f"MAIN: dist_url={args.dist_url}")

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print(
        f"MAIN: the setup {'is' if args.distributed else 'is not'} distributed."
    )

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print("MAIN: spawn procs")
        mp.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)
        )
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # get data
    train_dataset = iNaturalist(root=args.data, mode="train")
    validation_dataset = iNaturalist(root=args.data, mode="validation")

    # monkeypatch the no of classes.
    args.num_classes = train_dataset.num_classes

    if args.gpu is not None:
        print("MAIN: use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    if args.pretrained:
        if args.arch.startswith("inception"):
            print("MAIN: iniatialising pre-trained Inception_v3")
            # tested on inception v3
            model = models.__dict__[args.arch](pretrained=True)
            model.aux_logits = False
        else:
            print(f"MAIN: iniatialising pre-trained {args.arch}")
            model = models.__dict__[args.arch](pretrained=True)
        # Replacing the final layer.
        # 2048 for Inception, 512 for ResNet, etc.
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, args.num_classes)
    else:
        print(f"MAIN: iniatialising {args.arch} from scratch.")
        model = models.__dict__[args.arch](num_classes=args.num_classes)
    if args.arch.startswith("resnet"):
        model.avgpool = nn.AdaptiveAvgPool2d(1)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    print("MAIN: Creating loaders...")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # configure logger
    train_log = Logger(label="train", path=f"./results/{args.run_label}")
    train_log.add_metrics(
        batch_time=AverageMetric(),
        data_time=AverageMetric(),
        fps=FPSMetric(),
        losses=AverageMetric(),
        top1=AverageMetric(),
        top5=AverageMetric(),
    )

    # save the current options
    with open(f'./results/{args.run_label}/cfg.yaml', 'w') as cfg:
        yaml.dump(args.__dict__, cfg, default_flow_style=False)


    print("MAIN: starting training.")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, train_log, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )


def train(train_loader, model, criterion, optimizer, epoch, log, args):
    print(f"MAIN: epoch={epoch} begins...")

    header = "Batch cnt.\tTrain sec/batch (avg)\tData Load sec/batch (avg)\t"

    # switch to train mode
    model.train()

    log.reset()
    end = time.time()
    print(f"\n{header}\n{'-'*len(header)}")

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        log.update(data_time=(time.time() - end))

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        log.update(
            losses=(loss.item(), input.size(0)),
            top1=(acc1[0], input.size(0)),
            top5=(acc5[0], input.size(0)),
        )

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        log.update(
            batch_time=(time.time() - end),
            fps=(input.size(0), time.time() - end),
        )

        if i % args.print_freq == 0:
            # stdout stats
            log.log(cb=stdout_cb)

        if i % args.save_log_freq == 0:
            # save to disk
            log.save()

        end = time.time()

    log.save()


def validate(val_loader, model, criterion, args):

    batch_time = AverageMetric()
    losses = AverageMetric()
    top1 = AverageMetric()
    top5 = AverageMetric()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )

        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
        )

    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def stdout_cb(metrics):
    print(
        "Batch [{batch_time.count:6d}]\t"
        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
        "FPS {fps.val:.3f} ({fps.avg:.3f})\t"
        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
        "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
            batch_time=metrics["batch_time"],
            data_time=metrics["data_time"],
            fps=metrics["fps"],
            loss=metrics["losses"],
            top1=metrics["top1"],
        )
    )


if __name__ == "__main__":
    main()
