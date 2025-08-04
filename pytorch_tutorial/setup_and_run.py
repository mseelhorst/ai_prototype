"""
Quick Setup and Run Script for PyTorch Tutorial

This script helps you get started quickly by checking dependencies
and running the customer sentiment analysis example.
"""

import subprocess
import sys
import os

def check_and_install_requirements():
    """Check if required packages are installed and install if needed"""
    
    print("🔍 Checking PyTorch installation...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} is installed")
    except ImportError:
        print("❌ PyTorch not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
        print("✅ PyTorch installed successfully")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"⚠️ {package} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")

def main():
    """Main setup and run function"""
    
    print("🚀 PyTorch Tutorial Setup")
    print("=" * 40)
    
    # Check current directory
    if not os.path.exists('customer_sentiment_analysis.py'):
        print("❌ Please run this script from the pytorch_tutorial directory")
        print("📁 Navigate to the pytorch_tutorial folder first")
        return
    
    # Check and install requirements
    print("\n📦 Setting up dependencies...")
    check_and_install_requirements()
    
    print("\n🎯 Setup complete! Ready to run the tutorial.")
    print("\nOptions:")
    print("1. Run the complete customer sentiment analysis tutorial")
    print("2. Just check that everything is working")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🏃‍♂️ Running the complete tutorial...")
        print("This will demonstrate PyTorch capabilities with a real business application!")
        print("-" * 60)
        
        # Import and run the main tutorial
        try:
            from customer_sentiment_analysis import main
            main()
        except Exception as e:
            print(f"❌ Error running tutorial: {e}")
            print("💡 Try installing requirements manually: pip install -r requirements.txt")
    
    elif choice == "2":
        print("\n🔧 Testing installation...")
        try:
            import torch
            import pandas as pd
            import numpy as np
            import sklearn
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print("✅ All packages imported successfully!")
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            
            # Test basic tensor operations
            x = torch.randn(3, 3)
            y = torch.matmul(x, x.T)
            print(f"✅ Basic tensor operations working")
            print("🎉 Your PyTorch setup is ready!")
            
        except Exception as e:
            print(f"❌ Setup test failed: {e}")
            print("💡 Try running: pip install -r requirements.txt")
    
    else:
        print("👋 Setup complete. Run 'python customer_sentiment_analysis.py' when ready!")

if __name__ == "__main__":
    main()