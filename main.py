import torch
from datasets.inaturalist_image_folder import iNaturalist
from tqdm import tqdm

if __name__ == "__main__":
    dset = iNaturalist(root="./data/", mode="train")

    train_loader = torch.utils.data.DataLoader(
        dset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True
    )

    for i, (imgs, target, imgs_ids) in enumerate(train_loader):
        print(f"Batch: {i:5d}.")
        print("Shapes: ", imgs.shape, target.shape)
        print("Img ids: ", [idx.item() for idx in imgs_ids[:5]])
