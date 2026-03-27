import os
import torch
import struct
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset


train_img_dir = "./dataset/train-images.idx3-ubyte"
train_label_dir = "./dataset/train-labels.idx1-ubyte"
val_img_dir = "./dataset/t10k-images.idx3-ubyte"
val_label_dir = "./dataset/t10k-labels.idx1-ubyte"

def local_dataset_exists():
    files = [train_img_dir, train_label_dir, val_img_dir, val_label_dir]
    return all(os.path.exists(f) for f in files)

def download_dataset():
    train_set = datasets.MNIST(root="./data", train=True, download=True)
    val_set = datasets.MNIST(root="./data", train=False, download=True)
    return train_set, val_set

def read_idx_images(filename):
    with open(filename, 'rb') as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
    return torch.tensor(images, dtype=torch.float32) / 255.0

def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.tensor(labels, dtype=torch.long)

def cache_data(cache_dir="./dataset_cache"):
    train_cache_path = os.path.join(cache_dir, "train.pt")
    val_cache_path = os.path.join(cache_dir, "val.pt")
    
    # Check if the final cache already exists
    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        print(f"Cache already exists in {cache_dir}\n")
        return
        
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if the dataset exist locally
    if local_dataset_exists():
        print("Local raw files found. Reading and converting...\n")
        train_images = read_idx_images(train_img_dir)
        train_labels = read_idx_labels(train_label_dir)
        val_images = read_idx_images(val_img_dir)
        val_labels = read_idx_labels(val_label_dir)
        
    # If the dataset doesn't exist locally, download the dataset
    else:
        print("Local raw files not found. Downloading via torchvision...\n")
        # This will download the raw files into a temporary "./data" folder
        train_set, val_set = download_dataset()
        
        # Extract the images and labels
        # Normalize to [0, 1] 
        train_images = train_set.data.clone().detach().float() / 255.0
        train_labels = train_set.targets.clone().detach().long()
        
        val_images = val_set.data.clone().detach().float() / 255.0
        val_labels = val_set.targets.clone().detach().long()

    # Save the cached data
    print("Caching tensors...")
    torch.save((train_images, train_labels), train_cache_path)
    torch.save((val_images, val_labels), val_cache_path)
    print(f"Data successfully cached to {cache_dir}\n")


class MNISTCachedDataset(Dataset):
    # Loads the cached .pt file into memory
    def __init__(self, cache_file):
        super().__init__()
        
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Missing cache file: {cache_file}. Run cache_data() first")
            
        # Load the tuple of (images, labels) into RAM
        self.images, self.labels = torch.load(cache_file)
        print(f"Dataset initialized with {len(self.images)} samples from {cache_file}")

    # Returns the total number of samples
    def __len__(self):
        return len(self.images)

    # Fetches an image and its label
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Add a channel dimension
        # (28, 28) --> (1, 28, 28)
        image = image.unsqueeze(0)
        
        return image, label