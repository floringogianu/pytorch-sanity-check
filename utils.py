""" Some utilities that could be usefull across all benchmarks.
"""
import os
import pickle


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
