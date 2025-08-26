"""
train_failure_detection.py

Fine-tunes OpenVLA with failure detection capability using evaluation rollouts data.
"""

import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

import draccus
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import tqdm
from PIL import Image

# Import our failure detection model
from failure_model import OpenVLAWithFailureDetection


@dataclass
class FailureTrainingConfig:
    # Model
    pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-object"
    
    # Data
    rollouts_dir: str = "./rollouts_libero"  # Directory containing rollout episodes
    run_id: str = "EVAL-libero_object-openvla-2025_08_21-15_40_27"  # Specific run ID to use
    
    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 10
    grad_accumulation_steps: int = 1
    
    # Validation
    val_split: float = 0.2  # Use 20% of data for validation
    
    # Checkpointing
    save_dir: str = "./failure_detection_checkpoints"
    save_freq: int = 500  # Save every N steps
    
    # Device
    device: str = "cuda"
    
    # Logging
    log_freq: int = 100


class FailureDetectionDataset(Dataset):
    """Dataset for loading rollout data with failure labels"""
    
    def __init__(self, rollouts_dir, run_id, processor, split="train", val_split=0.2, seed=42):
        self.rollouts_dir = Path(rollouts_dir) / run_id
        self.processor = processor
        self.split = split
        
        # Load all episodes
        self.episodes = []
        episode_dirs = sorted([d for d in self.rollouts_dir.iterdir() if d.is_dir() and d.name.startswith("ep_")])
        
        # Split into train/val
        np.random.seed(seed)
        num_val = int(len(episode_dirs) * val_split)
        indices = np.random.permutation(len(episode_dirs))
        
        if split == "train":
            selected_dirs = [episode_dirs[i] for i in indices[num_val:]]
        else:
            selected_dirs = [episode_dirs[i] for i in indices[:num_val]]
            
        print(f"Loading {split} episodes: {len(selected_dirs)} episodes")
        
        # Load episode data
        for episode_dir in tqdm.tqdm(selected_dirs, desc=f"Loading {split} data"):
            try:
                episode_data = self._load_episode(episode_dir)
                if episode_data:
                    self.episodes.extend(episode_data)
            except Exception as e:
                print(f"Error loading episode {episode_dir}: {e}")
                
        print(f"Loaded {len(self.episodes)} timesteps for {split}")
        
    def _load_episode(self, episode_dir):
        """Load single episode data"""
        # Load metadata
        meta_path = episode_dir / "meta.json"
        if not meta_path.exists():
            return None
            
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        instruction = meta['instruction']
        
        # Load failure tokens
        failure_path = episode_dir / "failure_token.npy"
        if not failure_path.exists():
            return None
            
        failure_tokens = np.load(failure_path)  # Shape: (T,)
        
        # Load actions
        actions_path = episode_dir / "actions.npy"
        actions = np.load(actions_path)  # Shape: (T, 7)
        
        # Get image paths
        images_dir = episode_dir / "images"
        if not images_dir.exists():
            return None
            
        image_paths = sorted([p for p in images_dir.iterdir() if p.suffix == '.png'])
        
        # Ensure all have same length
        min_len = min(len(failure_tokens), len(actions), len(image_paths))
        
        episode_data = []
        for t in range(min_len):
            episode_data.append({
                'image_path': str(image_paths[t]),
                'instruction': instruction,
                'action': actions[t],
                'failure_label': int(failure_tokens[t])  # 0: success, 1: failure
            })
            
        return episode_data
        
    def __len__(self):
        return len(self.episodes)
        
    def __getitem__(self, idx):
        item = self.episodes[idx]
        
        # Load and process image
        image = Image.open(item['image_path']).convert('RGB')
        
        # Prepare prompt (same format as OpenVLA)
        prompt = f"In: What action should the robot take to {item['instruction'].lower()}?\nOut:"
        
        # Process inputs
        inputs = self.processor(prompt, image)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'failure_label': torch.tensor(item['failure_label'], dtype=torch.long),
            'instruction': item['instruction']
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    from torch.nn.utils.rnn import pad_sequence
    
    # Get max length for padding
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    # Pad sequences to same length
    input_ids = []
    attention_mask = []
    for item in batch:
        seq_len = item['input_ids'].size(0)
        if seq_len < max_len:
            # Pad with pad_token_id (usually 0 or specific pad token)
            pad_length = max_len - seq_len
            padded_input_ids = torch.cat([
                item['input_ids'], 
                torch.zeros(pad_length, dtype=item['input_ids'].dtype)
            ])
            padded_attention_mask = torch.cat([
                item['attention_mask'],
                torch.zeros(pad_length, dtype=item['attention_mask'].dtype)
            ])
        else:
            padded_input_ids = item['input_ids']
            padded_attention_mask = item['attention_mask']
            
        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)
    
    # Stack tensors
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    failure_labels = torch.stack([item['failure_label'] for item in batch])
    instructions = [item['instruction'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'failure_labels': failure_labels,
        'instructions': instructions
    }


def compute_metrics(predictions, labels):
    """Compute accuracy and other metrics"""
    pred_labels = torch.argmax(predictions, dim=-1)
    accuracy = (pred_labels == labels).float().mean().item()
    
    # Compute per-class metrics
    success_mask = (labels == 0)
    failure_mask = (labels == 1)
    
    success_acc = (pred_labels[success_mask] == labels[success_mask]).float().mean().item() if success_mask.sum() > 0 else 0.0
    failure_acc = (pred_labels[failure_mask] == labels[failure_mask]).float().mean().item() if failure_mask.sum() > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'success_accuracy': success_acc,
        'failure_accuracy': failure_acc,
        'num_success': success_mask.sum().item(),
        'num_failure': failure_mask.sum().item()
    }


def train_one_epoch(model, dataloader, optimizer, device, cfg):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm.tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'pixel_values': batch['pixel_values'].to(device, dtype=torch.bfloat16)
        }
        labels = batch['failure_labels'].to(device)
        
        # Forward pass
        outputs = model(inputs, return_hidden_states=True)
        
        if 'failure_logits' not in outputs:
            print("Warning: No failure_logits in outputs, skipping batch")
            continue
            
        failure_logits = outputs['failure_logits']
        
        # Compute loss
        loss = F.cross_entropy(failure_logits, labels)
        
        # Backward pass
        loss = loss / cfg.grad_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Collect predictions and labels for metrics
        with torch.no_grad():
            all_predictions.append(failure_logits.cpu())
            all_labels.append(labels.cpu())
        
        total_loss += loss.item() * cfg.grad_accumulation_steps
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item() * cfg.grad_accumulation_steps:.4f}'
        })
        
        # Log periodically
        if batch_idx % cfg.log_freq == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item() * cfg.grad_accumulation_steps:.4f}")
    
    # Compute final metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_predictions, all_labels)
    
    avg_loss = total_loss / len(dataloader)
    metrics['loss'] = avg_loss
    
    return metrics


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Validating"):
            # Move to device
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'pixel_values': batch['pixel_values'].to(device, dtype=torch.bfloat16)
            }
            labels = batch['failure_labels'].to(device)
            
            # Forward pass
            outputs = model(inputs, return_hidden_states=True)
            
            if 'failure_logits' not in outputs:
                continue
                
            failure_logits = outputs['failure_logits']
            
            # Compute loss
            loss = F.cross_entropy(failure_logits, labels)
            total_loss += loss.item()
            
            # Collect predictions
            all_predictions.append(failure_logits.cpu())
            all_labels.append(labels.cpu())
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_predictions, all_labels)
    
    avg_loss = total_loss / len(dataloader)
    metrics['loss'] = avg_loss
    
    return metrics


@draccus.wrap()
def train_failure_detection(cfg: FailureTrainingConfig):
    """Main training function"""
    print(f"Training failure detection on {cfg.rollouts_dir}/{cfg.run_id}")
    
    # Create save directory
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # Initialize model
    print("Loading OpenVLA model...")
    model = OpenVLAWithFailureDetection(
        pretrained_checkpoint=cfg.pretrained_checkpoint,
        device=cfg.device
    )
    
    # Freeze base model and only train failure head
    model.freeze_base_model()
    print("Base model frozen, training only failure detection head")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = FailureDetectionDataset(
        cfg.rollouts_dir, cfg.run_id, model.processor, 
        split="train", val_split=cfg.val_split
    )
    val_dataset = FailureDetectionDataset(
        cfg.rollouts_dir, cfg.run_id, model.processor, 
        split="val", val_split=cfg.val_split
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Setup optimizer
    optimizer = AdamW(model.get_trainable_parameters(), lr=cfg.learning_rate)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.7)
    
    print(f"Starting training for {cfg.num_epochs} epochs...")
    
    best_val_acc = 0.0
    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.num_epochs}")
        
        # Train
        train_metrics = train_one_epoch(model, train_dataloader, optimizer, cfg.device, cfg)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Train - Success Acc: {train_metrics['success_accuracy']:.4f}, Failure Acc: {train_metrics['failure_accuracy']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_dataloader, cfg.device)
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val - Success Acc: {val_metrics['success_accuracy']:.4f}, Failure Acc: {val_metrics['failure_accuracy']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            print(f"New best validation accuracy: {best_val_acc:.4f}")
            save_path = os.path.join(cfg.save_dir, "best_model")
            model.save_pretrained(save_path)
            print(f"Model saved to {save_path}")
        
        # Save epoch checkpoint
        save_path = os.path.join(cfg.save_dir, f"epoch_{epoch + 1}")
        model.save_pretrained(save_path)
        
        # Step scheduler
        scheduler.step()
        
        # Save training log
        log_data = {
            'epoch': epoch + 1,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        
        with open(os.path.join(cfg.save_dir, 'training_log.json'), 'a') as f:
            f.write(json.dumps(log_data) + '\n')
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved in: {cfg.save_dir}")


if __name__ == "__main__":
    train_failure_detection()