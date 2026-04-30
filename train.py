import argparse
import csv
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MLP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--scheduler', type=str, required=True, choices=['no_warmup', 'linear_warmup', 'cosine_warmup'])
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    # Hyperparameters (fixed)
    base_lr = 0.01
    momentum = 0.9
    batch_size = 64
    warmup_epochs = 5
    total_epochs = 30

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Load dataset
    if args.dataset == 'mnist':
        train_data = datasets.MNIST('./data', train=True, download=True,
                                    transform=transforms.ToTensor())
        # Use official train/val split: first 50k train, last 10k val
        train_subset, val_subset = torch.utils.data.random_split(train_data, [50000, 10000],
                                                                 generator=torch.Generator().manual_seed(42))
    else:
        train_data = datasets.FashionMNIST('./data', train=True, download=True,
                                           transform=transforms.ToTensor())
        train_subset, val_subset = torch.utils.data.random_split(train_data, [50000, 10000],
                                                                 generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum)

    # Prepare CSV logging
    csv_filename = f"{args.dataset}_{args.scheduler}_seed{args.seed}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_acc', 'grad_norm_epoch_avg'])

    # Training loop
    for epoch in range(1, total_epochs + 1):
        # Set learning rate according to scheduler
        if args.scheduler == 'no_warmup':
            lr = base_lr
        elif args.scheduler == 'linear_warmup':
            if epoch <= warmup_epochs:
                lr = base_lr * epoch / warmup_epochs
            else:
                lr = base_lr
        elif args.scheduler == 'cosine_warmup':
            if epoch <= warmup_epochs:
                lr = base_lr * 0.5 * (1 - math.cos(math.pi * epoch / warmup_epochs))
            else:
                lr = base_lr
        # Update optimizer's learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training phase
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0
        total_grad_norm = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Compute gradient norm (L2 over all parameters) for this batch
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            total_grad_norm += grad_norm
            num_batches += 1

            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train
        epoch_grad_norm_avg = total_grad_norm / num_batches

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        val_acc = correct_val / total_val

        # Log to CSV
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, train_acc, val_acc, epoch_grad_norm_avg])

        print(f"Epoch {epoch:2d} | LR {lr:.5f} | Train Loss {avg_loss:.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | Grad Norm {epoch_grad_norm_avg:.4f}")

if __name__ == '__main__':
    main()