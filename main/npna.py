import toml
import os
import torch
import wandb
from torchvision.transforms import transforms
from datasets import EELGrass
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from models.unet_model import UNet
from torch import nn, optim
from trainer import Trainer
import numpy as np
from PIL import Image
from losses import DiceLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb.login(key='434d12235bff28857fbf238c1278bdacead1838d')
grp_id = wandb.util.generate_id()
os.environ['WANDB_RUN_GROUP'] = 'experiment-' + grp_id

config = toml.load(open('../configs/npna.toml'))

torch.manual_seed(config['seed'])  # PyTorch random seed for CPU
torch.cuda.manual_seed(config['seed'])  # PyTorch random seed for GPU

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
    # transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
    transforms.Grayscale(1),
    # transforms.ToTensor()
])

augment = transforms.Compose([
    transforms.RandomRotation(degrees=90),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.3, 0.3),
        shear=0.5,
        scale=(1 - 0.3, 1 + 0.3),
        # fill='reflect',
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

tensor_to_image = transforms.ToPILImage()

dataset = EELGrass(config['image_dir'],
                   config['mask_dir'],
                   seed=config['seed']
                   )

validation_splits = []
kf = KFold(n_splits=config['num_folds'], shuffle=True, random_state=config['seed'])

for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}/{config['num_folds']}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    validation_splits.append(val_indices)

    # Split data into training and validation sets
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)

    # Create model
    model = UNet(n_channels=3,
                 n_classes=2).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    # criterion = nn.CrossEntropyLoss()
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_losses = []
    valid_losses = []

    # Train model
    trainer = Trainer(model, optimizer, criterion, train_loader, valid_loader, lr=config['learning_rate'],
                      device=device)
    trainer.train_and_evaluate(fold, config)
