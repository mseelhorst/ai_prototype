#!/usr/bin/env python3
"""
Step 3: Fine-tune CLIP on Domain-Specific Dataset
Demonstrates fine-tuning a pretrained CLIP model on a custom dataset and measuring improvements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
import json
import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
from datetime import datetime

class CustomCLIPDataset(Dataset):
    """Custom dataset for CLIP fine-tuning."""
    
    def __init__(self, image_text_pairs: List[Tuple[str, str]], processor, max_length: int = 77):
        """
        Initialize dataset.
        
        Args:
            image_text_pairs: List of (image_path_or_url, text_description) pairs
            processor: CLIP processor for preprocessing
            max_length: Maximum text length
        """
        self.pairs = image_text_pairs
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        image_source, text = self.pairs[idx]
        
        # Load image
        try:
            if image_source.startswith('http'):
                response = requests.get(image_source)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_source).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_source}: {e}")
            # Create a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Process image and text
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Remove batch dimension
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'text': text
        }


class CLIPFineTuner:
    """Fine-tuning class for CLIP models."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", 
                 device: Optional[str] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load pretrained model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        
        self.model_name = model_name
        print(f"Loaded pretrained CLIP model: {model_name}")
    
    def create_contrastive_loss(self, logits_per_image, logits_per_text, temperature: float = 0.07):
        """Create contrastive loss for CLIP training."""
        batch_size = logits_per_image.shape[0]
        
        # Create labels (diagonal matrix for positive pairs)
        labels = torch.arange(batch_size, device=self.device)
        
        # Compute cross-entropy loss for both directions
        loss_img = F.cross_entropy(logits_per_image / temperature, labels)
        loss_txt = F.cross_entropy(logits_per_text / temperature, labels)
        
        # Average the losses
        loss = (loss_img + loss_txt) / 2
        return loss
    
    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Compute contrastive loss
            loss = self.create_contrastive_loss(
                outputs.logits_per_image,
                outputs.logits_per_text
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        return total_loss / num_batches
    
    def evaluate(self, dataloader, compute_retrieval_metrics: bool = True):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_image_embeds = []
        all_text_embeds = []
        all_texts = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Compute loss
                loss = self.create_contrastive_loss(
                    outputs.logits_per_image,
                    outputs.logits_per_text
                )
                total_loss += loss.item()
                
                if compute_retrieval_metrics:
                    # Store embeddings for retrieval evaluation
                    all_image_embeds.append(outputs.image_embeds.cpu())
                    all_text_embeds.append(outputs.text_embeds.cpu())
                    all_texts.extend(batch['text'])
        
        avg_loss = total_loss / len(dataloader)
        metrics = {'val_loss': avg_loss}
        
        if compute_retrieval_metrics:
            # Compute retrieval metrics
            image_embeds = torch.cat(all_image_embeds, dim=0)
            text_embeds = torch.cat(all_text_embeds, dim=0)
            
            # Normalize embeddings
            image_embeds = F.normalize(image_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(image_embeds, text_embeds.T)
            
            # Image-to-text retrieval accuracy (R@1)
            img_to_text_preds = similarity_matrix.argmax(dim=1)
            img_to_text_acc = (img_to_text_preds == torch.arange(len(image_embeds))).float().mean()
            
            # Text-to-image retrieval accuracy (R@1)
            text_to_img_preds = similarity_matrix.argmax(dim=0)
            text_to_img_acc = (text_to_img_preds == torch.arange(len(text_embeds))).float().mean()
            
            metrics.update({
                'img_to_text_r1': img_to_text_acc.item(),
                'text_to_img_r1': text_to_img_acc.item(),
                'avg_retrieval_r1': (img_to_text_acc + text_to_img_acc).item() / 2
            })
        
        return metrics
    
    def fine_tune(self, train_dataset, val_dataset=None, 
                  epochs: int = 5, batch_size: int = 16, 
                  learning_rate: float = 5e-5, warmup_steps: int = 100,
                  save_dir: str = "checkpoints", use_wandb: bool = False):
        """Fine-tune the CLIP model."""
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4 if torch.cuda.is_available() else 0
            )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project="clip-finetuning",
                config={
                    "model": self.model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "dataset_size": len(train_dataset)
                }
            )
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        train_losses = []
        val_metrics_history = []
        best_val_loss = float('inf')
        
        print(f"Starting fine-tuning for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            train_losses.append(train_loss)
            
            print(f"Training loss: {train_loss:.4f}")
            
            # Validate
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                val_metrics_history.append(val_metrics)
                
                print(f"Validation metrics:")
                for key, value in val_metrics.items():
                    print(f"  {key}: {value:.4f}")
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    self.save_model(os.path.join(save_dir, "best_model"))
                    print("Saved best model!")
            
            # Log to wandb
            if use_wandb:
                log_dict = {"epoch": epoch + 1, "train_loss": train_loss}
                if val_loader:
                    log_dict.update(val_metrics)
                wandb.log(log_dict)
            
            # Save checkpoint
            self.save_model(os.path.join(save_dir, f"epoch_{epoch + 1}"))
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_metrics_history, save_dir)
        
        if use_wandb:
            wandb.finish()
        
        return {
            'train_losses': train_losses,
            'val_metrics': val_metrics_history
        }
    
    def save_model(self, save_path: str):
        """Save the fine-tuned model."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
    
    def load_model(self, model_path: str):
        """Load a fine-tuned model."""
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.to(self.device)
    
    def plot_training_curves(self, train_losses, val_metrics_history, save_dir):
        """Plot and save training curves."""
        if not val_metrics_history:
            return
        
        epochs = range(1, len(train_losses) + 1)
        val_losses = [m['val_loss'] for m in val_metrics_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Retrieval metrics
        if 'img_to_text_r1' in val_metrics_history[0]:
            img_to_text_acc = [m['img_to_text_r1'] for m in val_metrics_history]
            text_to_img_acc = [m['text_to_img_r1'] for m in val_metrics_history]
            avg_acc = [m['avg_retrieval_r1'] for m in val_metrics_history]
            
            ax2.plot(epochs, img_to_text_acc, 'g-', label='Image→Text R@1')
            ax2.plot(epochs, text_to_img_acc, 'b-', label='Text→Image R@1')
            ax2.plot(epochs, avg_acc, 'r-', label='Average R@1')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Retrieval Accuracy (R@1)')
            ax2.legend()
            ax2.grid(True)
        
        # Learning rate (if available)
        ax3.text(0.5, 0.5, 'Learning Rate Schedule\n(Cosine Annealing)', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Learning Rate Schedule')
        
        # Model info
        info_text = f"""
        Model: {self.model_name}
        Device: {self.device}
        Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}
        Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}
        """
        ax4.text(0.1, 0.5, info_text, ha='left', va='center', 
                transform=ax4.transAxes, fontfamily='monospace')
        ax4.set_title('Model Information')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {os.path.join(save_dir, 'training_curves.png')}")


def create_sample_finetuning_dataset():
    """Create a sample dataset for fine-tuning demonstration."""
    
    # Sample image-text pairs (in practice, you'd have a real dataset)
    # For demonstration, we'll use some publicly available images
    sample_pairs = [
        # Animals
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/300px-Cat_November_2010-1a.jpg", 
         "A beautiful orange tabby cat with bright eyes sitting peacefully"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/300px-Collage_of_Nine_Dogs.jpg", 
         "Multiple different dog breeds showing variety in size and appearance"),
        
        # Add more synthetic pairs for demonstration
        ("https://picsum.photos/300/300?random=1", "A scenic landscape with mountains and trees"),
        ("https://picsum.photos/300/300?random=2", "Modern architecture with clean geometric lines"),
        ("https://picsum.photos/300/300?random=3", "Colorful flowers blooming in a garden setting"),
        ("https://picsum.photos/300/300?random=4", "Urban street scene with buildings and traffic"),
        ("https://picsum.photos/300/300?random=5", "Natural water scene with reflections and ripples"),
        ("https://picsum.photos/300/300?random=6", "Interior design with modern furniture and lighting"),
    ]
    
    return sample_pairs


def compare_models(original_model_name: str, finetuned_model_path: str, test_dataset):
    """Compare original and fine-tuned models."""
    print("\n=== Model Comparison ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load original model
    original_model = CLIPModel.from_pretrained(original_model_name)
    original_processor = CLIPProcessor.from_pretrained(original_model_name)
    original_model.to(device)
    
    # Load fine-tuned model
    finetuned_model = CLIPModel.from_pretrained(finetuned_model_path)
    finetuned_processor = CLIPProcessor.from_pretrained(finetuned_model_path)
    finetuned_model.to(device)
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    def evaluate_model(model, processor):
        model.eval()
        all_image_embeds = []
        all_text_embeds = []
        
        with torch.no_grad():
            for batch in test_loader:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                all_image_embeds.append(outputs.image_embeds.cpu())
                all_text_embeds.append(outputs.text_embeds.cpu())
        
        # Compute retrieval metrics
        image_embeds = F.normalize(torch.cat(all_image_embeds, dim=0), dim=-1)
        text_embeds = F.normalize(torch.cat(all_text_embeds, dim=0), dim=-1)
        similarity_matrix = torch.matmul(image_embeds, text_embeds.T)
        
        img_to_text_acc = (similarity_matrix.argmax(dim=1) == torch.arange(len(image_embeds))).float().mean()
        text_to_img_acc = (similarity_matrix.argmax(dim=0) == torch.arange(len(text_embeds))).float().mean()
        
        return {
            'img_to_text_r1': img_to_text_acc.item(),
            'text_to_img_r1': text_to_img_acc.item(),
            'avg_retrieval_r1': (img_to_text_acc + text_to_img_acc).item() / 2
        }
    
    # Evaluate both models
    print("Evaluating original model...")
    original_metrics = evaluate_model(original_model, original_processor)
    
    print("Evaluating fine-tuned model...")
    finetuned_metrics = evaluate_model(finetuned_model, finetuned_processor)
    
    # Print comparison
    print("\nComparison Results:")
    print(f"{'Metric':<20} {'Original':<12} {'Fine-tuned':<12} {'Improvement':<12}")
    print("-" * 56)
    
    for metric in ['img_to_text_r1', 'text_to_img_r1', 'avg_retrieval_r1']:
        original_val = original_metrics[metric]
        finetuned_val = finetuned_metrics[metric]
        improvement = finetuned_val - original_val
        
        print(f"{metric:<20} {original_val:<12.4f} {finetuned_val:<12.4f} {improvement:+.4f}")
    
    return original_metrics, finetuned_metrics


def demo_clip_finetuning():
    """Demonstrate CLIP fine-tuning."""
    print("CLIP Fine-tuning Demo - Step 3")
    print("=" * 50)
    
    # Create sample dataset
    print("Creating sample dataset...")
    sample_pairs = create_sample_finetuning_dataset()
    
    # Initialize processor (we'll use it for dataset creation)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Split dataset (in practice, you'd have separate train/val datasets)
    train_pairs = sample_pairs[:6]  # Use first 6 for training
    val_pairs = sample_pairs[6:]    # Use last 2 for validation
    
    train_dataset = CustomCLIPDataset(train_pairs, processor)
    val_dataset = CustomCLIPDataset(val_pairs, processor)
    
    print(f"Created dataset with {len(train_dataset)} training and {len(val_dataset)} validation samples")
    
    # Initialize fine-tuner
    fine_tuner = CLIPFineTuner()
    
    # Fine-tune the model
    print("\nStarting fine-tuning...")
    training_history = fine_tuner.fine_tune(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=3,  # Small number for demo
        batch_size=4,  # Small batch size for demo
        learning_rate=1e-5,
        save_dir="clip_finetuned",
        use_wandb=False  # Set to True if you want to use W&B
    )
    
    # Compare models
    if os.path.exists("clip_finetuned/best_model"):
        print("\nComparing original vs fine-tuned model...")
        original_metrics, finetuned_metrics = compare_models(
            "openai/clip-vit-base-patch32",
            "clip_finetuned/best_model",
            val_dataset
        )
    
    print("\n" + "=" * 50)
    print("Fine-tuning demo completed!")
    print("Files created:")
    print("- clip_finetuned/: Directory with fine-tuned model checkpoints")
    print("- clip_finetuned/training_curves.png: Training visualization")


if __name__ == "__main__":
    demo_clip_finetuning()