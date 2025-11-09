
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import math
import random
from io import BytesIO


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def set_device():
    return torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")


#main classifier class
class SimpleTwoLayerMLP(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Reducing spatial size before flatten
        self.pool = nn.AdaptiveAvgPool2d((8, 8))  # from 224x224 -> 8x8

        in_dim = 8 * 8 * 3  # 192
        hidden = 128

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = self.pool(x)              # [B, 3, 8, 8]
        x = x.view(x.size(0), -1)     # [B, 192]
        return self.classifier(x)     # [B, 1]




MEAN = (0.2447, 0.2446, 0.2447)
STD = (0.1892, 0.1892, 0.1891)




def train_transforms(img_size: int = 224) -> transforms.Compose:
    #Training transformations with resizing, horizontal flip, and rotation.
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # Adds a slight random rotation
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(MEAN), torch.Tensor(STD)),
        ]
    )

def val_transforms(img_size: int = 224) -> transforms.Compose:
    #Validation transformations with resizing and normalization.
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(MEAN), torch.Tensor(STD)),
        ]
    )



def prepare_dataloader(dataset_path, transform, batch_size, is_train=True):
    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=is_train)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=4)
    return loader

def save_checkpoint(model, epoch, optimizer, best_acc, path):
    state = {
        'epoch': epoch + 1,
        'model': model.module.state_dict(),
        'best_accuracy': best_acc,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_correct, total = 0.0, 0.0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * running_correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * running_correct / total

def train(rank, world_size, train_path, test_path, save_path, total_epochs, batch_size):
    ddp_setup(rank, world_size)
    device = set_device()

    # Transforms and Dataloaders
    batch_size_per_gpu = batch_size // world_size
    train_loader = prepare_dataloader(train_path, train_transforms(img_size=224), batch_size_per_gpu, is_train=True)
    test_loader = prepare_dataloader(test_path, val_transforms(img_size=224), batch_size_per_gpu, is_train=False)

    model = SimpleTwoLayerMLP(num_classes=1)

    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0001,          
    betas=(0.9, 0.99),
    eps=1e-8,         
    weight_decay=0.006 
    )

    best_acc = 0.0
    for epoch in range(total_epochs):
        print(f"Epoch {epoch + 1}/{total_epochs}")
        train_loader.sampler.set_epoch(epoch)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test Accuracy: {test_acc:.2f}%")

        if rank == 0 and test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, epoch, optimizer, best_acc, save_path)

    destroy_process_group()

if __name__ == "__main__":
    import argparse
    from torch.multiprocessing import spawn

    parser = argparse.ArgumentParser(description="Distributed Training with Two-Layer MLP Model")
    parser.add_argument("train_path", type=str, help="Path to training dataset")
    parser.add_argument("test_path", type=str, help="Path to testing dataset")
    parser.add_argument("--save_path", type=str, default="./saved_models/twolayermlp_weight.pth", help="Path to save the best model")
    parser.add_argument("--total_epochs", type=int, default=10, help="Number of total epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for each GPU")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    spawn(
        train,
        args=(world_size, args.train_path, args.test_path, args.save_path, args.total_epochs, args.batch_size),
        nprocs=world_size,
    )

