import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from Loader import *
from Models import *

class MNISTTrainer():
    def __init__(self, config):
        self.config = config
        self.device = self.config['device']
        self.validation_freq = self.config.get('validation_freq', 1)

        self.global_step = 0
        self.best_accuracy = 0.0

        # Model Initialization
        self.model = self.config['model'].to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
        self.loss_fn = self.config['loss_funct'].to(self.device)

        # Directory Setup
        self.top_dir = './Exps/'
        self.exp_dir = self.config['exp_dir']

        self.weights_dir = os.path.join(self.top_dir, self.exp_dir)
        os.makedirs(self.weights_dir, exist_ok=True)

        # Prepare and Load Data
        print("Loading data...")
        cache_data(cache_dir="./dataset_cache")
        
        train_dataset = MNISTCachedDataset("./dataset_cache/train.pt")
        val_dataset = MNISTCachedDataset("./dataset_cache/val.pt")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )

    def save_checkpoint(self, ckpt_name='model_latest.pt'):
        ckpt_path = os.path.join(self.weights_dir, ckpt_name)
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_accuracy': self.best_accuracy,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"=> Checkpoint saved to {ckpt_path}")

    def train(self):
        print("\nTraining...")

        for epoch in range(self.config['num_epochs']):
            self.train_epoch(epoch)
            
            # Validation
            if (epoch + 1) % self.validation_freq == 0:
                accuracy = self.validation()
                
                # Save checkpoint
                self.save_checkpoint(ckpt_name='model_latest.pt')
                
                # Update best checkpoint
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.save_checkpoint(ckpt_name='model_best.pt')
                    print(f"[Best Accuracy: {self.best_accuracy:.2f}%]\n")
            
        print("Training Complete!\n")

    def train_epoch(self, epoch: int):
        self.model.train()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}", ncols=100)

        for images, labels in pbar:
            self.global_step += 1
            
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            predictions = self.model(images)
            loss = self.loss_fn(predictions, labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    def validation(self):
        print("\nValidating...")
        self.model.eval()
        
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation", ncols=100):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Calculate loss
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average accuracy and loss
        accuracy = 100 * correct / total
        avg_loss = val_loss / len(self.val_loader)
        
        print(f"[Validation Results - Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%]")
        
        # Switch back to training mode
        self.model.train()
        return accuracy


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    '''
        num_epochs :       Training Epochs
        lr :               Learning Rate
        device :           Device used for training
        batch_size :       Size of samples in a batch
        num_workers :      Number of CPU workers
        validation_freq :  Number of epochs before validation
        model :            Model for training
        loss_funct :       Loss function
        exp_dir :          Name of save directory
    '''
    config = {
        'num_epochs': 10,  
        'lr': 1e-3,  
        'device': device, 
        'batch_size': 64, 
        'num_workers': 8,   
        'validation_freq': 10,  
        'model' : SimpleMNISTCNN(), 
        'loss_funct' : nn.CrossEntropyLoss(),
        'exp_dir' : 'Example_Training'            
    }
    
    trainer = MNISTTrainer(config)
    trainer.train()