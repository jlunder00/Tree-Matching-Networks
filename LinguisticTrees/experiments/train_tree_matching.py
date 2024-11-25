# LinguisticTrees/experiments/train_tree_matching.py
from COMMON.src.utils.config import cfg
from COMMON.src.dataset.data_loader import get_dataloader
from LinguisticTrees.graph_matching.dataset import TreeMatchingDataset
from LinguisticTrees.graph_matching.model import TreeMatchingNet
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import time

def train_model():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir='runs/tree_matching')
    
    # Load data
    train_dataset = TreeMatchingDataset('data/snli_train.json')
    val_dataset = TreeMatchingDataset('data/snli_val.json')
    
    train_loader = get_dataloader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE)
    val_loader = get_dataloader(val_dataset, batch_size=cfg.EVAL.BATCH_SIZE)
    
    # Create model
    model = TreeMatchingNet().to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for i, (graphs, labels) in enumerate(train_loader):
            graphs = graphs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(graphs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f'Epoch {epoch}, Iter {i}, Loss: {loss.item():.4f}')
                writer.add_scalar('training_loss', loss.item(), 
                                epoch * len(train_loader) + i)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        print(f'Epoch {epoch} Validation:')
        for metric, value in val_metrics.items():
            print(f'{metric}: {value:.4f}')
            writer.add_scalar(f'val_{metric}', value, epoch)
            
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), 'checkpoints/best_tree_matching.pt')
            
        print(f'Epoch {epoch} completed in {time.time() - start_time:.2f}s')
        
    writer.close()

if __name__ == '__main__':
    train_model()
