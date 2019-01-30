""" Some utilities that could be usefull across all benchmarks.
"""
import os
import argparse
import pickle
import torchvision.models as models


def parse_cmd():
    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
        choices=model_names,
        help="model architecture: "
        + " | ".join(model_names)
        + " (default: resnet18)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--save-log-freq",
        default=200,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://224.66.41.62:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    return parser.parse_args()


class AverageMetric:
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


class FPSMetric:
    def __init__(self):
        self.reset()

    def update(self, units, time, n=1):
        self.time += time
        self.sum += units
        self.count += n
        self.val = units / time
        self.avg = self.sum / self.time

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.time = 0
        self.count = 0


class Logger:
    def __init__(self, label="train", path=None):
        self.label = label
        self.metrics = {}
        self.history = {}

        self.path = path
        if path is None:
            self.path = os.path.join(os.getcwd(), label)

        try:
            print("Creating directory %s." % str(self.path))
            os.makedirs(self.path)
        except FileExistsError:
            print(f"Warning! Directory {self.path} exists!")

    def add_metrics(self, **metrics):
        for metric_name, metric in metrics.items():
            if metric_name in self.metrics:
                print(f"LOG: warning, metric {metric_name} already registered!")
            self.metrics[metric_name] = metric
            self.history[metric_name] = []

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert k in self.metrics, print(
                f"{k} metric you are trying to update is not registered."
            )
            if isinstance(v, tuple):
                self.metrics[k].update(*v)
            else:
                self.metrics[k].update(v)

            # also update history
            self.history[k].append(
                {
                    "step_idx": self.metrics[k].count,
                    "value": self.metrics[k].val,
                }
            )

    def reset(self):
        for _, metric in self.metrics.items():
            metric.reset()

    def log(self, cb=None):
        if cb:
            cb(self.metrics)

    def save(self):
        filename = "%s.pkl" % self.label.replace(" ", "_").lower()
        path = os.path.join(self.path, filename)

        with open(path, "wb") as f:
            pickle.dump(self.history, f)
