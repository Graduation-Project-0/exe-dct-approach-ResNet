import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tqdm import tqdm

class MetricsTracker:    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_scores = []
    
    def update(self, labels, predictions, scores):
        self.y_true.extend(labels.cpu().numpy().tolist())
        self.y_pred.extend(predictions.cpu().numpy().tolist())
        # For 2-class output, use softmax probabilities for the positive class (Malware)
        probs = torch.softmax(scores, dim=1)[:, 1]  # Probability of class 1 (Malware)
        self.y_scores.extend(probs.cpu().numpy().tolist())
    
    def compute_metrics(self) -> Dict[str, float]:
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        y_scores = np.array(self.y_scores)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # ROC AUC
        if len(np.unique(y_true)) > 1:  # Need both classes for ROC
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            metrics['auc'] = auc(fpr, tpr)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
        else:
            metrics['auc'] = 0.0
        
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        return metrics

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    gradient_accumulation_steps: int = 1
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True) 
        labels = labels.to(device, non_blocking=True).long() 
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        running_loss += loss.item() * images.size(0) * gradient_accumulation_steps

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
        
        total += labels.size(0)
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            predictions = torch.argmax(outputs, dim=1)
            
            metrics_tracker.update(labels, predictions, outputs)
    
    avg_loss = running_loss / len(data_loader.dataset)
    metrics = metrics_tracker.compute_metrics()
    
    return avg_loss, metrics

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: torch.device = None,
    save_path: Optional[str] = None,
    patience: int = 10
) -> Dict:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    print(f"Training on device: {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        val_acc = val_metrics['accuracy']
        val_auc = val_metrics.get('auc', 0.0)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            print(f"-> New best validation loss")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model from training")
    
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history
        }, save_path)
        print(f"Model saved to {save_path}")
    
    return history

def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = None
) -> Dict:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    print("\nEvaluating on test set...")
    test_loss, metrics = evaluate(model, test_loader, criterion, device)
    
    print("Test Results:")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1-Score:       {metrics['f1']:.4f}")
    print(f"AUC:            {metrics.get('auc', 0):.4f}")
    print()
    
    cm = metrics['confusion_matrix']
    
    metrics['test_loss'] = test_loss
    return metrics

def plot_training_history(history: Dict, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss 
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy 
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # AUC 
    axes[2].plot(epochs, history['val_auc'], 'g-', label='Val AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('Validation AUC')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_roc_curve(metrics: Dict, save_path: Optional[str] = None):
    if 'fpr' not in metrics or 'tpr' not in metrics:
        print("ROC curve data not available")
        return
    
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2, label=f'ROC curve (AUC = {metrics["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Benign', 'Malware']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=20)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
