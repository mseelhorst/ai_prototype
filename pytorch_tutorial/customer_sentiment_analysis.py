"""
Customer Sentiment Analysis - A Practical PyTorch Business Application

This script demonstrates how to use PyTorch to build a sentiment analysis model
that can help businesses understand customer feedback and improve their services.

Business Value:
- Automatically classify customer reviews as positive/negative
- Identify areas for improvement based on negative feedback
- Monitor customer satisfaction trends over time
- Scale customer service by prioritizing negative reviews

Author: PyTorch Tutorial
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SentimentDataset(Dataset):
    """Custom PyTorch Dataset for sentiment analysis"""
    
    def __init__(self, texts, labels, vectorizer=None, max_features=5000):
        self.texts = texts
        self.labels = labels
        
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),  # Use both unigrams and bigrams
                lowercase=True
            )
            # Fit the vectorizer on the texts
            self.features = self.vectorizer.fit_transform(texts).toarray()
        else:
            self.vectorizer = vectorizer
            # Transform using existing vectorizer
            self.features = vectorizer.transform(texts).toarray()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Convert to PyTorch tensors
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])
        return features, label

class SentimentClassifier(nn.Module):
    """Neural Network for Sentiment Classification"""
    
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
        super(SentimentClassifier, self).__init__()
        
        # Define the neural network layers
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, 2)  # 2 classes: positive (1) and negative (0)
        )
    
    def forward(self, x):
        return self.network(x)

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def create_sample_data():
    """Create sample customer review data for demonstration"""
    
    # Sample customer reviews (in a real scenario, you'd load from a database or CSV)
    sample_reviews = [
        # Positive reviews
        "This product is amazing! Great quality and fast shipping.",
        "Excellent customer service. Very helpful and friendly staff.",
        "Love this product! Exactly what I was looking for.",
        "Outstanding quality and great value for money.",
        "Fast delivery and product works perfectly.",
        "Highly recommend this to anyone looking for quality.",
        "Great experience shopping here. Will definitely return.",
        "Product exceeded my expectations. Very satisfied.",
        "Fantastic quality and excellent customer support.",
        "Perfect product, exactly as described.",
        
        # Negative reviews
        "Terrible product quality. Broke after one day.",
        "Worst customer service ever. Very rude staff.",
        "Product arrived damaged and return process was difficult.",
        "Poor quality materials. Not worth the price.",
        "Delivery was extremely slow and tracking was inaccurate.",
        "Product doesn't work as advertised. Very disappointed.",
        "Customer service was unhelpful and dismissive.",
        "Overpriced for the quality you get.",
        "Would not recommend. Waste of money.",
        "Product failed to meet basic expectations."
    ]
    
    # Labels: 1 = positive, 0 = negative
    labels = [1] * 10 + [0] * 10
    
    # Create DataFrame
    df = pd.DataFrame({
        'review': sample_reviews,
        'sentiment': labels
    })
    
    # Save to CSV for future use
    df.to_csv('pytorch_tutorial/sample_reviews.csv', index=False)
    print("Sample data saved to 'sample_reviews.csv'")
    
    return df

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the sentiment analysis model"""
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Lists to store training history
    train_losses = []
    val_accuracies = []
    
    print("Starting training...")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for features, labels in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            labels = labels.squeeze()  # Remove extra dimension
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                labels = labels.squeeze()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        val_accuracy = 100 * val_correct / val_total
        
        train_losses.append(avg_loss)
        val_accuracies.append(val_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    print("-" * 50)
    print("Training completed!")
    
    return train_losses, val_accuracies

def evaluate_model(model, test_loader, dataset):
    """Evaluate the model and provide business insights"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.numpy())
            all_labels.extend(labels.squeeze().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['Negative', 'Positive']))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - Customer Sentiment Analysis')
    plt.ylabel('Actual Sentiment')
    plt.xlabel('Predicted Sentiment')
    plt.tight_layout()
    plt.savefig('pytorch_tutorial/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy, all_predictions, all_labels

def business_insights(predictions, labels, texts):
    """Generate business insights from the sentiment analysis"""
    
    print("\n" + "="*60)
    print("BUSINESS INSIGHTS")
    print("="*60)
    
    # Overall sentiment distribution
    positive_count = sum(predictions)
    negative_count = len(predictions) - positive_count
    total_reviews = len(predictions)
    
    print(f"📊 SENTIMENT DISTRIBUTION:")
    print(f"   Positive Reviews: {positive_count} ({positive_count/total_reviews:.1%})")
    print(f"   Negative Reviews: {negative_count} ({negative_count/total_reviews:.1%})")
    
    # Model performance insights
    correct_predictions = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct_predictions / len(predictions)
    
    print(f"\n🎯 MODEL PERFORMANCE:")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Correctly classified: {correct_predictions}/{total_reviews} reviews")
    
    # Business recommendations
    print(f"\n💼 BUSINESS RECOMMENDATIONS:")
    
    if negative_count > positive_count:
        print("   ⚠️  ATTENTION NEEDED: More negative than positive reviews detected!")
        print("   • Prioritize customer service improvements")
        print("   • Investigate common complaint themes")
        print("   • Implement proactive customer outreach")
    else:
        print("   ✅ POSITIVE TREND: More positive reviews than negative")
        print("   • Continue current successful practices")
        print("   • Leverage positive feedback for marketing")
        print("   • Monitor for any declining trends")
    
    print(f"\n📈 AUTOMATION BENEFITS:")
    print(f"   • Can automatically classify ~{accuracy:.0%} of reviews")
    print(f"   • Reduces manual review time by {accuracy:.0%}")
    print(f"   • Enables real-time customer satisfaction monitoring")
    print(f"   • Helps prioritize customer service resources")

def predict_new_review(model, vectorizer, review_text):
    """Predict sentiment for a new customer review"""
    
    # Preprocess the text
    cleaned_text = preprocess_text(review_text)
    
    # Transform using the fitted vectorizer
    features = vectorizer.transform([cleaned_text]).toarray()
    features_tensor = torch.FloatTensor(features)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(features_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
    
    sentiment = "Positive" if predicted.item() == 1 else "Negative"
    confidence = probabilities[0][predicted.item()].item()
    
    return sentiment, confidence

def main():
    """Main function to run the complete sentiment analysis pipeline"""
    
    print("🚀 Customer Sentiment Analysis with PyTorch")
    print("=" * 60)
    
    # Step 1: Create or load data
    print("\n📊 Step 1: Loading customer review data...")
    df = create_sample_data()
    
    # Preprocess text data
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    
    print(f"Loaded {len(df)} customer reviews")
    print(f"Positive reviews: {sum(df['sentiment'])}")
    print(f"Negative reviews: {len(df) - sum(df['sentiment'])}")
    
    # Step 2: Prepare data for PyTorch
    print("\n🔧 Step 2: Preparing data for machine learning...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'].values, 
        df['sentiment'].values, 
        test_size=0.3, 
        random_state=42,
        stratify=df['sentiment']
    )
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train)
    test_dataset = SentimentDataset(X_test, y_test, vectorizer=train_dataset.vectorizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Feature dimensions: {train_dataset.features.shape[1]}")
    
    # Step 3: Create and train model
    print("\n🧠 Step 3: Creating and training the neural network...")
    
    # Initialize model
    input_size = train_dataset.features.shape[1]
    model = SentimentClassifier(input_size=input_size)
    
    print(f"Model architecture:")
    print(f"  Input features: {input_size}")
    print(f"  Hidden layers: 128 → 64 neurons")
    print(f"  Output classes: 2 (positive/negative)")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    train_losses, val_accuracies = train_model(model, train_loader, test_loader, num_epochs=100)
    
    # Step 4: Evaluate model
    print("\n📈 Step 4: Evaluating model performance...")
    accuracy, predictions, labels = evaluate_model(model, test_loader, test_dataset)
    
    # Step 5: Generate business insights
    business_insights(predictions, labels, X_test)
    
    # Step 6: Demonstrate real-time prediction
    print("\n🔮 Step 6: Real-time sentiment prediction demo...")
    print("-" * 40)
    
    # Test with new reviews
    new_reviews = [
        "This product is fantastic! Amazing quality and service.",
        "Terrible experience. Product broke immediately.",
        "Good value for money, would recommend to others.",
        "Customer service was really unhelpful and slow."
    ]
    
    for review in new_reviews:
        sentiment, confidence = predict_new_review(model, train_dataset.vectorizer, review)
        print(f"Review: '{review[:50]}{'...' if len(review) > 50 else ''}'")
        print(f"Prediction: {sentiment} (confidence: {confidence:.2%})")
        print()
    
    # Step 7: Save model for production use
    print("💾 Step 7: Saving model for production deployment...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vectorizer': train_dataset.vectorizer,
        'model_config': {
            'input_size': input_size,
            'hidden_size': 128
        }
    }, 'pytorch_tutorial/sentiment_model.pth')
    
    print("✅ Model saved successfully!")
    print("\n🎉 Tutorial completed! Your sentiment analysis model is ready for business use.")
    print("\nNext steps:")
    print("• Integrate with your customer feedback system")
    print("• Set up automated alerts for negative sentiment spikes")
    print("• Create dashboards to monitor sentiment trends")
    print("• Scale up with larger datasets for better accuracy")

if __name__ == "__main__":
    main()