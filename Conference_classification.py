import torch

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

class DoraemonConferenceDataset(Dataset):
    def __init__(self, root_dir, weights_path, label_map):
        self.features = []
        self.labels = []
        print(f"\nLoading weights from {weights_path}")
        self.weights = torch.load(weights_path)
        
        total_samples = 0
        class_counts = {label: 0 for label in label_map.keys()}
        
        print("\nLoading conference data...")
        for label, subfolder in label_map.items():
            subfolder_path = os.path.join(root_dir, "publishable", subfolder)
            if os.path.isdir(subfolder_path):
                files = [f for f in os.listdir(subfolder_path) if f.endswith(".pt")]
                for file in tqdm(files, desc=f"Loading {subfolder}"):
                    tensor = torch.load(os.path.join(subfolder_path, file))
                    tensor = tensor * self.weights
                    self.features.append(tensor)
                    self.labels.append(label)
                    class_counts[label] += 1
                    total_samples += 1
        
        self.features = torch.stack(self.features)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        print("\nDataset Summary:")
        print(f"Total samples: {total_samples}")
        for label, count in class_counts.items():
            print(f"{label_map[label]}: {count} samples")
        print(f"Feature dimension: {self.features.shape[1]}")
        
        # Normalize features
        print("\nNormalizing features...")
        self.features = (self.features - self.features.mean(dim=0)) / (self.features.std(dim=0) + 1e-6)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DoraemonConferenceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DoraemonConferenceClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, num_classes)
        )
        
        # Print model architecture
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nModel Architecture:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def forward(self, x):
        return self.model(x)

def calculate_multiclass_metrics(y_true, y_pred, num_classes):
    correct = (y_pred == y_true).sum().item()
    total = y_true.size(0)
    accuracy = correct / total
    
    # Per-class metrics
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    for i in range(num_classes):
        mask = (y_true == i)
        class_correct[i] = ((y_pred == y_true) & mask).sum().item()
        class_total[i] = mask.sum().item()
    
    class_accuracies = class_correct / (class_total + 1e-6)
    
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies.tolist(),
        'class_counts': class_total.tolist()
    }

def conference_model(input_dim, num_classes):
    print(f"\nCreating model with input dimension: {input_dim}")
    model = DoraemonConferenceClassifier(input_dim, num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    return model, loss_fn, optimizer, scheduler

def train_multiclass_model(model, loss_fn, optimizer, scheduler, train_loader, val_loader, label_map, epochs=20, device='cpu'):
    model.to(device)
    best_val_loss = float('inf')
    num_classes = len(label_map)
    
    print(f"\nStarting training for {epochs} epochs")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for X_batch, y_batch in train_pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.extend(torch.argmax(outputs, dim=1).cpu())
            train_labels.extend(y_batch.cpu())
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        with torch.no_grad():
            for X_val, y_val in val_pbar:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                loss = loss_fn(val_outputs, y_val)
                val_loss += loss.item()
                
                val_predictions.extend(torch.argmax(val_outputs, dim=1).cpu())
                val_labels.extend(y_val.cpu())
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_predictions = torch.tensor(train_predictions)
        train_labels = torch.tensor(train_labels)
        val_predictions = torch.tensor(val_predictions)
        val_labels = torch.tensor(val_labels)
        
        train_metrics = calculate_multiclass_metrics(train_labels, train_predictions, num_classes)
        val_metrics = calculate_multiclass_metrics(val_labels, val_predictions, num_classes)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"Training:")
        print(f"  Loss: {avg_train_loss:.4f}")
        print(f"  Overall Accuracy: {train_metrics['accuracy']:.4f}")
        print("  Per-class Accuracies:")
        for i, acc in enumerate(train_metrics['class_accuracies']):
            print(f"    {label_map[i]}: {acc:.4f} ({train_metrics['class_counts'][i]} samples)")
        
        print(f"\nValidation:")
        print(f"  Loss: {avg_val_loss:.4f}")
        print(f"  Overall Accuracy: {val_metrics['accuracy']:.4f}")
        print("  Per-class Accuracies:")
        for i, acc in enumerate(val_metrics['class_accuracies']):
            print(f"    {label_map[i]}: {acc:.4f} ({val_metrics['class_counts'][i]} samples)")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"\nNew best model found! Saving checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'metrics': val_metrics,
            }, 'doraemon_conference_classifier.pt')
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

def prepare_multiclass_data(data_dir, weights_path, label_map, train_split=0.8, batch_size=32):
    print(f"\nPreparing data from {data_dir}")
    print(f"Train split: {train_split}")
    print(f"Batch size: {batch_size}")
    
    dataset = DoraemonConferenceDataset(data_dir, weights_path, label_map)
    labels = [label for _, label in dataset]
    
    train_indices, val_indices = train_test_split(
        range(len(labels)),
        test_size=1 - train_split,
        stratify=labels,
        random_state=42
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    return train_loader, val_loader, dataset.features.shape[1]

def main():
    print("\nStarting conference classification training")
    
    data_dir = "Dataset/vectors"
    weights_path = "Dataset/weight2.pt"
    label_map = {0: "CVPR", 1: "TMLR", 2: "KDD", 3: "NEURIPS", 4: "EMNLP"}
    batch_size = 32
    epochs = 10
    
    train_loader, val_loader, input_dim = prepare_multiclass_data(data_dir, weights_path, label_map, batch_size=batch_size)
    
    model, loss_fn, optimizer, scheduler = conference_model(input_dim, len(label_map))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    train_multiclass_model(model, loss_fn, optimizer, scheduler, train_loader, val_loader, 
                          label_map, epochs=epochs, device=device)
    
    print("\nTraining completed. Best model saved as 'doraemon_conference_classifier.pt'")

if __name__ == "__main__":
    main()
