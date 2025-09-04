#!/usr/bin/env python3
"""
Step 2: Toy Search Engine with FAISS
Build a search engine that encodes text and images, stores them in FAISS, and performs retrieval.
"""

import torch
import torch.nn.functional as F
import faiss
import numpy as np
from PIL import Image
import pickle
import os
from typing import List, Dict, Tuple, Union
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
import json
from tqdm import tqdm

class MultimodalSearchEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32", index_file="search_index.faiss"):
        """Initialize the search engine with CLIP model and FAISS index."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize FAISS index
        self.embedding_dim = 512  # CLIP embedding dimension
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        self.index_file = index_file
        
        # Storage for metadata
        self.items = []  # List of dicts with metadata
        self.item_types = []  # 'text' or 'image'
        
        print(f"Initialized search engine with {model_name}")
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text using CLIP."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features.cpu().numpy()
    
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """Encode images using CLIP."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
        
        return image_features.cpu().numpy()
    
    def add_texts(self, texts: List[str], metadata: List[Dict] = None):
        """Add texts to the search index."""
        print(f"Adding {len(texts)} texts to index...")
        
        # Encode texts in batches
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i+batch_size]
            embeddings = self.encode_text(batch_texts)
            all_embeddings.append(embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        for i, text in enumerate(texts):
            item_metadata = metadata[i] if metadata else {}
            item_metadata.update({
                'content': text,
                'type': 'text',
                'id': len(self.items)
            })
            self.items.append(item_metadata)
            self.item_types.append('text')
        
        print(f"Added {len(texts)} texts. Total items: {len(self.items)}")
    
    def add_images(self, images: List[Image.Image], metadata: List[Dict] = None):
        """Add images to the search index."""
        print(f"Adding {len(images)} images to index...")
        
        # Encode images in batches
        batch_size = 16  # Smaller batch size for images
        all_embeddings = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            batch_images = images[i:i+batch_size]
            embeddings = self.encode_images(batch_images)
            all_embeddings.append(embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        for i, image in enumerate(images):
            item_metadata = metadata[i] if metadata else {}
            item_metadata.update({
                'type': 'image',
                'id': len(self.items)
            })
            # Note: We don't store the actual image, just metadata
            # In a real system, you'd store image paths or URLs
            self.items.append(item_metadata)
            self.item_types.append('image')
        
        print(f"Added {len(images)} images. Total items: {len(self.items)}")
    
    def search_with_text(self, query: str, k: int = 5, filter_type: str = None) -> List[Dict]:
        """Search using text query."""
        # Encode query
        query_embedding = self.encode_text([query])
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding, min(k * 2, len(self.items)))
        
        # Filter results if needed
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            item = self.items[idx].copy()
            item['score'] = float(score)
            
            # Apply type filter
            if filter_type is None or item['type'] == filter_type:
                results.append(item)
            
            if len(results) >= k:
                break
        
        return results
    
    def search_with_image(self, image: Image.Image, k: int = 5, filter_type: str = None) -> List[Dict]:
        """Search using image query."""
        # Encode query
        query_embedding = self.encode_images([image])
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding, min(k * 2, len(self.items)))
        
        # Filter results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            item = self.items[idx].copy()
            item['score'] = float(score)
            
            # Apply type filter
            if filter_type is None or item['type'] == filter_type:
                results.append(item)
            
            if len(results) >= k:
                break
        
        return results
    
    def save_index(self, metadata_file: str = "search_metadata.json"):
        """Save FAISS index and metadata."""
        faiss.write_index(self.index, self.index_file)
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump({
                'items': self.items,
                'item_types': self.item_types
            }, f, indent=2)
        
        print(f"Saved index to {self.index_file} and metadata to {metadata_file}")
    
    def load_index(self, metadata_file: str = "search_metadata.json"):
        """Load FAISS index and metadata."""
        if os.path.exists(self.index_file) and os.path.exists(metadata_file):
            self.index = faiss.read_index(self.index_file)
            
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                self.items = data['items']
                self.item_types = data['item_types']
            
            print(f"Loaded index with {len(self.items)} items")
            return True
        return False
    
    def get_stats(self):
        """Get statistics about the search index."""
        text_count = sum(1 for t in self.item_types if t == 'text')
        image_count = sum(1 for t in self.item_types if t == 'image')
        
        return {
            'total_items': len(self.items),
            'text_items': text_count,
            'image_items': image_count,
            'index_size': self.index.ntotal
        }


class SearchEngineEvaluator:
    """Evaluate search engine performance."""
    
    def __init__(self, search_engine: MultimodalSearchEngine):
        self.search_engine = search_engine
    
    def evaluate_text_retrieval(self, queries_and_expected: List[Tuple[str, List[str]]], k: int = 5):
        """Evaluate text retrieval performance."""
        total_queries = len(queries_and_expected)
        total_precision = 0
        total_recall = 0
        total_mrr = 0  # Mean Reciprocal Rank
        
        print(f"Evaluating text retrieval on {total_queries} queries...")
        
        for query, expected_results in tqdm(queries_and_expected):
            results = self.search_engine.search_with_text(query, k=k)
            retrieved_items = [item['content'] for item in results if item['type'] == 'text']
            
            # Calculate metrics
            relevant_retrieved = set(retrieved_items) & set(expected_results)
            
            precision = len(relevant_retrieved) / len(retrieved_items) if retrieved_items else 0
            recall = len(relevant_retrieved) / len(expected_results) if expected_results else 0
            
            # Calculate MRR
            mrr = 0
            for i, item in enumerate(retrieved_items):
                if item in expected_results:
                    mrr = 1 / (i + 1)
                    break
            
            total_precision += precision
            total_recall += recall
            total_mrr += mrr
        
        avg_precision = total_precision / total_queries
        avg_recall = total_recall / total_queries
        avg_mrr = total_mrr / total_queries
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'mrr': avg_mrr
        }
    
    def evaluate_cross_modal_retrieval(self, image_text_pairs: List[Tuple[Image.Image, str]], k: int = 5):
        """Evaluate cross-modal retrieval (image->text, text->image)."""
        print(f"Evaluating cross-modal retrieval on {len(image_text_pairs)} pairs...")
        
        image_to_text_hits = 0
        text_to_image_hits = 0
        
        for image, expected_text in tqdm(image_text_pairs):
            # Image -> Text
            text_results = self.search_engine.search_with_image(image, k=k, filter_type='text')
            retrieved_texts = [item['content'] for item in text_results]
            if expected_text in retrieved_texts:
                image_to_text_hits += 1
            
            # Text -> Image (harder to evaluate without ground truth image matching)
            # For now, we'll skip this or implement with image similarity
        
        image_to_text_accuracy = image_to_text_hits / len(image_text_pairs)
        
        return {
            'image_to_text_accuracy': image_to_text_accuracy,
            'text_to_image_accuracy': None  # Not implemented
        }


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    # Sample texts (product descriptions, articles, etc.)
    sample_texts = [
        "A beautiful red sports car with sleek design and powerful engine",
        "Cute orange tabby cat sitting on a windowsill in the morning sun",
        "Modern smartphone with advanced camera and long battery life",
        "Delicious pizza with pepperoni, mushrooms, and extra cheese",
        "Majestic mountain landscape with snow-capped peaks and clear blue sky",
        "Cozy coffee shop interior with wooden furniture and warm lighting",
        "Professional laptop computer perfect for business and gaming",
        "Fresh green salad with tomatoes, cucumbers, and olive oil dressing",
        "Adorable golden retriever puppy playing in a sunny garden",
        "Vintage bicycle with basket, perfect for city commuting",
        "Luxurious hotel room with ocean view and modern amenities",
        "Homemade chocolate cake with strawberries and whipped cream",
        "High-tech gaming setup with multiple monitors and RGB lighting",
        "Peaceful forest trail with tall trees and dappled sunlight",
        "Stylish winter jacket with waterproof material and warm insulation"
    ]
    
    # Sample image URLs (for demonstration - in practice you'd have local images)
    sample_image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/300px-Cat_November_2010-1a.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/300px-Collage_of_Nine_Dogs.jpg",
    ]
    
    return sample_texts, sample_image_urls


def load_image_from_url(url):
    """Load image from URL."""
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None


def demo_search_engine():
    """Demonstrate the search engine functionality."""
    print("Multimodal Search Engine Demo - Step 2")
    print("=" * 50)
    
    # Initialize search engine
    engine = MultimodalSearchEngine()
    
    # Try to load existing index
    if not engine.load_index():
        print("Creating new search index...")
        
        # Create sample dataset
        texts, image_urls = create_sample_dataset()
        
        # Add texts to index
        text_metadata = [{'source': 'sample_dataset', 'category': 'product'} for _ in texts]
        engine.add_texts(texts, text_metadata)
        
        # Load and add images
        images = []
        image_metadata = []
        for i, url in enumerate(image_urls):
            image = load_image_from_url(url)
            if image:
                images.append(image)
                image_metadata.append({
                    'source': 'sample_dataset',
                    'url': url,
                    'category': 'animal'
                })
        
        if images:
            engine.add_images(images, image_metadata)
        
        # Save index
        engine.save_index()
    
    # Print statistics
    stats = engine.get_stats()
    print(f"\nSearch Index Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Demonstrate text search
    print("\n=== Text Search Demo ===")
    text_queries = [
        "red sports car",
        "cute cat",
        "mountain landscape",
        "chocolate cake"
    ]
    
    for query in text_queries:
        print(f"\nQuery: '{query}'")
        results = engine.search_with_text(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['type']}] Score: {result['score']:.3f}")
            if result['type'] == 'text':
                content = result['content'][:80] + "..." if len(result['content']) > 80 else result['content']
                print(f"     Content: {content}")
            else:
                print(f"     Image metadata: {result.get('category', 'N/A')}")
    
    # Demonstrate image search (if images available)
    if stats['image_items'] > 0:
        print("\n=== Image Search Demo ===")
        # Load a test image
        test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/300px-Cat_November_2010-1a.jpg"
        test_image = load_image_from_url(test_image_url)
        
        if test_image:
            print("Searching with cat image...")
            results = engine.search_with_image(test_image, k=5)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result['type']}] Score: {result['score']:.3f}")
                if result['type'] == 'text':
                    content = result['content'][:80] + "..." if len(result['content']) > 80 else result['content']
                    print(f"     Content: {content}")
    
    # Demonstrate evaluation
    print("\n=== Search Engine Evaluation ===")
    evaluator = SearchEngineEvaluator(engine)
    
    # Create simple evaluation queries
    eval_queries = [
        ("red car", ["A beautiful red sports car with sleek design and powerful engine"]),
        ("cat", ["Cute orange tabby cat sitting on a windowsill in the morning sun"]),
        ("pizza", ["Delicious pizza with pepperoni, mushrooms, and extra cheese"])
    ]
    
    metrics = evaluator.evaluate_text_retrieval(eval_queries, k=5)
    print("Text Retrieval Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print("\n" + "=" * 50)
    print("Search engine demo completed!")
    print(f"Index saved to: {engine.index_file}")
    print("You can now run queries against the built index.")


if __name__ == "__main__":
    demo_search_engine()