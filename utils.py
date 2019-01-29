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


class Logger:
    def __init__(self, label="log", path=None):
        self.label = label
        self.metrics = {}

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

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert k in self.metrics, print(
                f"{k} metric you are trying to update is not registered."
            )
            if isinstance(v, tuple):
                self.metrics[k].update(*v)
            else:
                self.metrics[k].update(v)

    def reset(self):
        for _, metric in self.metrics.items():
            metric.reset()
    
    def save(self):
        filename = '%s.pkl' % self.label.replace(" ", "_").lower()
        path = os.path.join(self.path, filename)

        try:
            with open(path, 'rb') as f:
                history = pickle.load(f)
        except FileNotFoundError:
            history = {k: [] for k in self.metrics.keys()}
        
        for metric_name, metric in self.metrics.items():
            history[metric_name].append({
                "step_idx": metric.count,
                "value": metric.value
            })


