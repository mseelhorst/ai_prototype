#!/usr/bin/env python3
"""
Step 1: CLIP Hands-on Demo
Demonstrates zero-shot image classification and image-text retrieval using CLIP.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import open_clip
from transformers import CLIPProcessor, CLIPModel
import os

class CLIPDemo:
    def __init__(self, model_name="ViT-B/32", use_openai=True):
        """Initialize CLIP model and processor."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if use_openai:
            # Use OpenAI CLIP from transformers
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            print("Loaded OpenAI CLIP model")
        else:
            # Use OpenCLIP
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained='openai'
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.to(self.device)
            print(f"Loaded OpenCLIP model: {model_name}")
        
        self.use_openai = use_openai
    
    def load_image_from_url(self, url):
        """Load image from URL."""
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
            return None
    
    def zero_shot_classification(self, image, class_labels):
        """Perform zero-shot image classification."""
        if self.use_openai:
            # Prepare text prompts
            text_prompts = [f"a photo of a {label}" for label in class_labels]
            
            # Process inputs
            inputs = self.processor(
                text=text_prompts, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
        else:
            # OpenCLIP approach
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            text_prompts = [f"a photo of a {label}" for label in class_labels]
            text_input = self.tokenizer(text_prompts).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                
                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Compute similarities
                similarities = (image_features @ text_features.T)
                probs = similarities.softmax(dim=-1)
        
        return probs.cpu().numpy()[0]
    
    def image_text_retrieval(self, images, texts):
        """Perform image-text retrieval."""
        if self.use_openai:
            # Process all images and texts
            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Normalize embeddings
                image_embeds = F.normalize(image_embeds, dim=-1)
                text_embeds = F.normalize(text_embeds, dim=-1)
                
                # Compute similarity matrix
                similarity_matrix = torch.matmul(image_embeds, text_embeds.T)
        else:
            # OpenCLIP approach
            image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            text_inputs = self.tokenizer(texts).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_inputs)
                text_features = self.model.encode_text(text_inputs)
                
                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Compute similarity matrix
                similarity_matrix = torch.matmul(image_features, text_features.T)
        
        return similarity_matrix.cpu().numpy()
    
    def visualize_results(self, image, class_labels, probabilities):
        """Visualize classification results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Input Image')
        
        # Show probabilities
        y_pos = np.arange(len(class_labels))
        bars = ax2.barh(y_pos, probabilities)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(class_labels)
        ax2.set_xlabel('Probability')
        ax2.set_title('Zero-shot Classification Results')
        
        # Color bars based on probability
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            bar.set_color(plt.cm.viridis(prob))
        
        plt.tight_layout()
        return fig
    
    def demo_zero_shot_classification(self):
        """Run zero-shot classification demo."""
        print("\n=== Zero-shot Image Classification Demo ===")
        
        # Sample images (you can replace with local images)
        image_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/1200px-Collage_of_Nine_Dogs.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Solid_blue.svg/1200px-Solid_blue.svg.png"
        ]
        
        # Class labels for classification
        class_labels = ["cat", "dog", "car", "airplane", "bird", "fish", "horse"]
        
        for i, url in enumerate(image_urls):
            print(f"\nProcessing image {i+1}...")
            image = self.load_image_from_url(url)
            if image is None:
                continue
            
            # Perform classification
            probabilities = self.zero_shot_classification(image, class_labels)
            
            # Print results
            print("Classification results:")
            for label, prob in zip(class_labels, probabilities):
                print(f"  {label}: {prob:.3f}")
            
            # Find top prediction
            top_idx = np.argmax(probabilities)
            print(f"Top prediction: {class_labels[top_idx]} ({probabilities[top_idx]:.3f})")
            
            # Visualize (save to file)
            fig = self.visualize_results(image, class_labels, probabilities)
            plt.savefig(f'classification_result_{i+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved as classification_result_{i+1}.png")
    
    def demo_image_text_retrieval(self):
        """Run image-text retrieval demo."""
        print("\n=== Image-Text Retrieval Demo ===")
        
        # Sample images
        image_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/300px-Cat_November_2010-1a.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/300px-Collage_of_Nine_Dogs.jpg",
        ]
        
        # Sample texts
        texts = [
            "a cute cat sitting",
            "multiple dogs playing",
            "a red sports car",
            "beautiful sunset landscape"
        ]
        
        # Load images
        images = []
        for url in image_urls:
            img = self.load_image_from_url(url)
            if img:
                images.append(img)
        
        if not images:
            print("No images loaded successfully")
            return
        
        # Perform retrieval
        similarity_matrix = self.image_text_retrieval(images, texts)
        
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print("Similarity scores (images x texts):")
        print("Rows: Images, Columns: Texts")
        
        # Print similarity matrix
        for i in range(len(images)):
            print(f"\nImage {i+1}:")
            for j, text in enumerate(texts):
                score = similarity_matrix[i, j]
                print(f"  '{text}': {score:.3f}")
        
        # Find best matches
        print("\nBest text matches for each image:")
        for i in range(len(images)):
            best_text_idx = np.argmax(similarity_matrix[i])
            best_score = similarity_matrix[i, best_text_idx]
            print(f"Image {i+1} -> '{texts[best_text_idx]}' (score: {best_score:.3f})")
        
        print("\nBest image matches for each text:")
        for j in range(len(texts)):
            best_image_idx = np.argmax(similarity_matrix[:, j])
            best_score = similarity_matrix[best_image_idx, j]
            print(f"'{texts[j]}' -> Image {best_image_idx+1} (score: {best_score:.3f})")
        
        # Visualize similarity matrix
        plt.figure(figsize=(10, 6))
        plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Similarity Score')
        plt.xlabel('Text Descriptions')
        plt.ylabel('Images')
        plt.title('Image-Text Similarity Matrix')
        
        # Add text labels
        plt.xticks(range(len(texts)), [f"Text {i+1}" for i in range(len(texts))], rotation=45)
        plt.yticks(range(len(images)), [f"Image {i+1}" for i in range(len(images))])
        
        # Add similarity scores as text
        for i in range(len(images)):
            for j in range(len(texts)):
                plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                        ha='center', va='center', color='white')
        
        plt.tight_layout()
        plt.savefig('similarity_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Similarity matrix visualization saved as similarity_matrix.png")


def main():
    """Main demo function."""
    print("CLIP Demonstration - Step 1")
    print("=" * 50)
    
    # Initialize CLIP demo
    demo = CLIPDemo(use_openai=True)  # Set to False to use OpenCLIP
    
    # Run demos
    demo.demo_zero_shot_classification()
    demo.demo_image_text_retrieval()
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the generated visualization files.")
    print("Files created:")
    print("- classification_result_*.png")
    print("- similarity_matrix.png")


if __name__ == "__main__":
    main()