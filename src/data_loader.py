from torch.utils.data import WeightedRandomSampler
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class AnimalPostureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom dataset for loading animal posture images, with an option to include Stable Diffusion data.
        
        Args:
        - root_dir (str): Root directory containing 'train' or 'test' folders with class subfolders.
        - transform (callable, optional): Optional transform to be applied on a sample.
        - use_stable_diffusion (bool): Whether to include Stable Diffusion-generated images.
        - stable_diffusion_dir (str, optional): Path to the Stable Diffusion data directory.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.class_map = {"lie": 0, "run": 1, "sit": 2, "walk_stand": 3}
        
        for class_name in self.class_map:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_map[class_name])  # Assign numeric label
            
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
def get_dataloaders(data_dir, batch_size=32, img_size=224, augment=False):
    """
    Load dataset with optional data augmentation and optional class balancing.
    
    Parameters:
    - balance_classes (bool): Whether to use class-balanced sampling for the training loader.
    """
    
    # Transform definitions
    base_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    augmented_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_transform = augmented_transform if augment else base_transform
    val_transform = base_transform

    # Load datasets
    train_dataset = AnimalPostureDataset(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = AnimalPostureDataset(os.path.join(data_dir, "test"), transform=val_transform)


    # Count instances per class
    class_counts = {}
    for label in train_dataset.labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    # Compute class weights: inverse of counts
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    
    # Assign a sample weight to each image
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    # Create sampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)


    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
