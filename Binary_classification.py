import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
import os
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from datetime import datetime

class DoraemonDataset(Dataset):
    def __init__(self, root_dir, weights_path):
        self.features = []
        self.labels = []
        print(f"\nLoading weights from {weights_path}")
        self.weights = torch.load(weights_path)
        
        # Load non-publishable data
        non_pub_dir = os.path.join(root_dir, "non-publishable")
        print(f"\nLoading non-publishable data from {non_pub_dir}")
        non_pub_files = [f for f in os.listdir(non_pub_dir) if f.endswith(".pt")]
        for file in tqdm(non_pub_files, desc="Loading non-publishable data"):
            tensor = torch.load(os.path.join(non_pub_dir, file))
            tensor = tensor * self.weights
            self.features.append(tensor)
            self.labels.append(0)
        
        # Load publishable data
        pub_dir = os.path.join(root_dir, "publishable")
        print(f"\nLoading publishable data from {pub_dir}")
        pub_count = 0
        for subfolder in os.listdir(pub_dir):
            subfolder_path = os.path.join(pub_dir, subfolder)
            if os.path.isdir(subfolder_path):
                files = [f for f in os.listdir(subfolder_path) if f.endswith(".pt")]
                for file in tqdm(files, desc=f"Loading {subfolder}"):
                    tensor = torch.load(os.path.join(subfolder_path, file))
                    tensor = tensor * self.weights
                    self.features.append(tensor)
                    self.labels.append(1)
                    pub_count += 1
        
        self.features = torch.stack(self.features)
        self.labels = torch.tensor(self.labels, dtype=torch.float)
        
        print("\nDataset Summary:")
        print(f"Total samples: {len(self.labels)}")
        print(f"Non-publishable samples: {len(non_pub_files)}")
        print(f"Publishable samples: {pub_count}")
        print(f"Feature dimension: {self.features.shape[1]}")
        print(f"Class distribution: {torch.bincount(self.labels.long()).tolist()}")
        
        # Normalize features
        print("\nNormalizing features...")
        self.features = (self.features - self.features.mean(dim=0)) / self.features.std(dim=0)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DoraemonBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DoraemonBinaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.7),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.7),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Print model architecture
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nModel Architecture:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def forward(self, x):
        return self.model(x)

def calculate_metrics(y_true, y_pred):
    tp = torch.sum((y_true == 1) & (y_pred == 1)).float()
    tn = torch.sum((y_true == 0) & (y_pred == 0)).float()
    fp = torch.sum((y_true == 0) & (y_pred == 1)).float()
    fn = torch.sum((y_true == 1) & (y_pred == 0)).float()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else torch.tensor(0.0)
    recall = tp / (tp + fn) if tp + fn > 0 else torch.tensor(0.0)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else torch.tensor(0.0)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'tp': tp.item(),
        'tn': tn.item(),
        'fp': fp.item(),
        'fn': fn.item()
    }

def create_model(input_dim):
    print(f"\nCreating model with input dimension: {input_dim}")
    model = DoraemonBinaryClassifier(input_dim)
    loss_fn = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    return model, loss_fn, optimizer, scheduler

def train_model(model, loss_fn, optimizer, scheduler, train_loader, val_loader, epochs=20, device='cpu'):
    model.to(device)
    best_val_loss = float('inf')
    
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
            outputs = model(X_batch).squeeze()
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.extend((outputs >= 0.5).float().cpu())
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
                val_outputs = model(X_val).squeeze()
                loss = loss_fn(val_outputs, y_val)
                val_loss += loss.item()
                
                val_predictions.extend((val_outputs >= 0.5).float().cpu())
                val_labels.extend(y_val.cpu())
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_predictions = torch.tensor(train_predictions)
        train_labels = torch.tensor(train_labels)
        val_predictions = torch.tensor(val_predictions)
        val_labels = torch.tensor(val_labels)
        
        train_metrics = calculate_metrics(train_labels, train_predictions)
        val_metrics = calculate_metrics(val_labels, val_predictions)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"Training:")
        print(f"  Loss: {avg_train_loss:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall: {train_metrics['recall']:.4f}")
        print(f"  F1 Score: {train_metrics['f1']:.4f}")
        print(f"  Confusion Matrix: [TP: {train_metrics['tp']}, TN: {train_metrics['tn']}, FP: {train_metrics['fp']}, FN: {train_metrics['fn']}]")
        
        print(f"Validation:")
        print(f"  Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1 Score: {val_metrics['f1']:.4f}")
        print(f"  Confusion Matrix: [TP: {val_metrics['tp']}, TN: {val_metrics['tn']}, FP: {val_metrics['fp']}, FN: {val_metrics['fn']}]")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"\nNew best model found! Saving checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'metrics': val_metrics,
            }, 'doraemon_binary_classifier.pt')
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

def prepare_data(data_dir, weights_path, train_split=0.8, batch_size=32):
    print(f"\nPreparing data from {data_dir}")
    print(f"Train split: {train_split}")
    print(f"Batch size: {batch_size}")
    
    dataset = DoraemonDataset(data_dir, weights_path)
    labels = dataset.labels
    
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_split), random_state=42)
    train_indices, val_indices = next(stratified_split.split(torch.arange(len(labels)), labels))
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    return train_loader, val_loader, dataset.features.shape[1]

def main():
    print("\nStarting binary classification training")
    
    data_dir = "Dataset/vectors"
    weights_path = "Dataset/weight1.pt"
    batch_size = 32
    epochs = 10
    
    train_loader, val_loader, input_dim = prepare_data(data_dir, weights_path, batch_size=batch_size)
    
    model, loss_fn, optimizer, scheduler = create_model(input_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    train_model(model, loss_fn, optimizer, scheduler, train_loader, val_loader, 
                epochs=epochs, device=device)
    
    print("\nTraining completed. Best model saved as 'doraemon_binary_classifier.pt'")

if __name__ == "__main__":
    main()