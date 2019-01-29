# PyTorch Sanity Check
---

**Work in progress...**

Data is assumed to be the
[iNaturalist](https://github.com/visipedia/inat_comp) dataset for the
convolutional benchmark.

Most of the code here follows PyTorch
[examples](https://github.com/pytorch/examples).


## Give it a go:

For dev purposes:
```sh
python inaturalist.py ./data --workers 8 -b 12 --gpu 0
```

For filling up four `RTX 2080Ti` and training with `torch.DataParallel`:
```sh
python inaturalist.py ./data -a resnet50 --workers 32 -b 192
```

## Todo:

- [ ] Write a proper Readme. 
- [ ] Read from `yaml` config files the standardized benchmarks.
- [ ] Start adding additional models.
