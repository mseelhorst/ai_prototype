# 🚀 Quick Start Guide

Get up and running with PyTorch in under 5 minutes!

## Option 1: Automated Setup (Recommended)

```bash
cd pytorch_tutorial
python setup_and_run.py
```

This will automatically check dependencies and run the tutorial.

## Option 2: Manual Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Tutorial
```bash
python customer_sentiment_analysis.py
```

## What You'll Learn

- ✅ **PyTorch Basics**: Tensors, models, training loops
- ✅ **Real Business Application**: Customer sentiment analysis  
- ✅ **Complete ML Pipeline**: Data → Model → Insights
- ✅ **Production Ready**: Save and load trained models

## Expected Output

The tutorial will:
1. Create sample customer review data
2. Train a neural network to classify sentiment
3. Evaluate model performance
4. Generate business insights
5. Demonstrate real-time predictions
6. Save the model for production use

## Files Generated

After running, you'll have:
- `sample_reviews.csv` - Sample customer data
- `confusion_matrix.png` - Model performance visualization
- `sentiment_model.pth` - Trained model for production

## Troubleshooting

**PyTorch Installation Issues?**
```bash
# For CPU-only (easier)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Missing Dependencies?**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Import Errors?**
Make sure you're in the `pytorch_tutorial` directory when running the scripts.

## Next Steps

1. 📖 Read the main `README.md` for comprehensive PyTorch explanation
2. 🔧 Modify the model architecture in `customer_sentiment_analysis.py`
3. 📊 Try your own dataset by replacing the sample data
4. 🚀 Deploy the model using Flask or FastAPI
5. 📈 Scale up with larger datasets and GPU training

## Need Help?

- Check the detailed `README.md`
- Review the commented code in `customer_sentiment_analysis.py`
- Visit [PyTorch Documentation](https://pytorch.org/docs/)
- Ask questions in [PyTorch Forums](https://discuss.pytorch.org/)

Happy learning! 🎉