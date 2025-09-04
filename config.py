#!/usr/bin/env python3
"""
Configuration file for AI Techniques with CLIP demonstrations.
Modify these settings to customize the behavior of the demos.
"""

import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class GlobalConfig:
    """Global configuration for all demonstrations."""
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model configuration
    clip_model_name: str = "openai/clip-vit-base-patch32"
    
    # Performance settings
    batch_size: int = 16
    num_workers: int = 4
    
    # Output settings
    save_visualizations: bool = True
    output_dir: str = "outputs"
    
    # Logging
    verbose: bool = True
    log_level: str = "INFO"

@dataclass
class Step1Config(GlobalConfig):
    """Configuration for Step 1: CLIP Demo."""
    
    # Number of sample images to process
    num_sample_images: int = 3
    
    # Classification labels
    classification_labels: list = None
    
    def __post_init__(self):
        if self.classification_labels is None:
            self.classification_labels = [
                "cat", "dog", "car", "airplane", "bird", 
                "fish", "horse", "bicycle", "flower", "building"
            ]

@dataclass
class Step2Config(GlobalConfig):
    """Configuration for Step 2: Search Engine."""
    
    # FAISS index settings
    faiss_index_type: str = "IndexFlatIP"  # Inner product for cosine similarity
    
    # Dataset settings
    num_sample_texts: int = 100
    num_sample_images: int = 20
    
    # Search settings
    default_top_k: int = 5

@dataclass
class Step3Config(GlobalConfig):
    """Configuration for Step 3: Fine-tuning."""
    
    # Training settings
    learning_rate: float = 5e-5
    epochs: int = 3
    batch_size: int = 8  # Smaller for fine-tuning
    
    # Fine-tuning specific settings
    temperature: float = 0.07
    weight_decay: float = 0.01
    
    # Model saving
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"

@dataclass
class Step4Config(GlobalConfig):
    """Configuration for Step 4: Experiment Design."""
    
    # Experiment tracking
    track_experiments: bool = True
    experiments_dir: str = "experiments"
    
    # Statistical testing
    confidence_level: float = 0.95
    n_bootstrap: int = 1000
    
    # Visualization settings
    figure_size: tuple = (12, 8)
    dpi: int = 150

@dataclass
class Step5Config(GlobalConfig):
    """Configuration for Step 5: Scaling."""
    
    # Vector database settings
    vector_db_type: str = "pinecone"  # Options: pinecone, weaviate, milvus
    
    # Performance optimization
    use_half_precision: bool = torch.cuda.is_available()
    use_torch_compile: bool = False  # Set to True for PyTorch 2.0+
    max_workers: int = 4
    
    # Benchmarking
    benchmark_num_texts: int = 100
    benchmark_num_images: int = 50
    benchmark_num_queries: int = 50

# Global configuration instance
CONFIG = GlobalConfig()

# Step-specific configurations
STEP1_CONFIG = Step1Config()
STEP2_CONFIG = Step2Config()
STEP3_CONFIG = Step3Config()
STEP4_CONFIG = Step4Config()
STEP5_CONFIG = Step5Config()

def get_config(step: Optional[int] = None):
    """Get configuration for a specific step or global config."""
    if step == 1:
        return STEP1_CONFIG
    elif step == 2:
        return STEP2_CONFIG
    elif step == 3:
        return STEP3_CONFIG
    elif step == 4:
        return STEP4_CONFIG
    elif step == 5:
        return STEP5_CONFIG
    else:
        return CONFIG

def update_config(**kwargs):
    """Update global configuration with new values."""
    for key, value in kwargs.items():
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)
        else:
            print(f"Warning: Unknown configuration key: {key}")

def print_config(step: Optional[int] = None):
    """Print current configuration."""
    config = get_config(step)
    print(f"Configuration for {'Step ' + str(step) if step else 'Global'}:")
    print("-" * 40)
    
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    print()

if __name__ == "__main__":
    # Print all configurations
    print("AI Techniques with CLIP - Configuration Overview")
    print("=" * 60)
    
    print_config()  # Global
    
    for i in range(1, 6):
        print_config(i)  # Step-specific