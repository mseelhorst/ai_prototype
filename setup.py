#!/usr/bin/env python3
"""
Setup script for AI Techniques with CLIP
Provides easy installation and environment setup.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            print("⚠️  No GPU detected - will run on CPU (slower performance)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - GPU check skipped")
        return False

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        print("✅ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "experiments",
        "checkpoints",
        "outputs",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Created necessary directories")

def run_quick_test():
    """Run a quick test to verify installation."""
    print("🧪 Running quick installation test...")
    
    try:
        # Test imports
        import torch
        import transformers
        import faiss
        import matplotlib
        import numpy as np
        from PIL import Image
        
        print("✅ All core packages imported successfully")
        
        # Test CLIP model loading
        from transformers import CLIPModel, CLIPProcessor
        
        print("📥 Testing CLIP model download (this may take a moment)...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        print("✅ CLIP model loaded successfully")
        
        # Test basic functionality
        test_image = Image.new('RGB', (224, 224), color='red')
        test_text = ["a red image", "a blue image"]
        
        inputs = processor(text=test_text, images=test_image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        
        print("✅ Basic CLIP functionality test passed")
        print("🎉 Installation verification complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        print("   Try running the installation again or check the error messages above.")
        return False

def main():
    """Main setup function."""
    print("🚀 AI Techniques with CLIP - Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_python_version():
        return False
    
    print(f"🖥️  Operating System: {platform.system()} {platform.release()}")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check GPU after installation
    check_gpu()
    
    # Create directories
    create_directories()
    
    # Run verification test
    if not run_quick_test():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python main_demo.py' to start the interactive learning journey")
    print("2. Or run individual steps: 'python step1_clip_demo.py', etc.")
    print("3. Check README.md for detailed documentation")
    print("\nHappy learning! 🚀")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)