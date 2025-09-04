#!/usr/bin/env python3
"""
Step 5: Scaling with Vector Databases and Efficient Inference
Explore vector databases (Pinecone, Weaviate, Milvus) and efficient inference techniques.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import threading
import queue
from contextlib import contextmanager

# Vector database clients (mock implementations for demo)
class MockVectorDB:
    """Base class for mock vector database implementations."""
    
    def __init__(self, name: str, dimension: int = 512):
        self.name = name
        self.dimension = dimension
        self.vectors = {}
        self.metadata = {}
        self.index_built = False
    
    def upsert(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]):
        """Insert or update vectors."""
        for vector_id, vector, meta in vectors:
            self.vectors[vector_id] = np.array(vector)
            self.metadata[vector_id] = meta
    
    def query(self, vector: List[float], top_k: int = 10, filter_dict: Dict = None) -> List[Dict]:
        """Query similar vectors."""
        if not self.vectors:
            return []
        
        query_vec = np.array(vector)
        similarities = {}
        
        for vec_id, stored_vec in self.vectors.items():
            # Apply filters if provided
            if filter_dict:
                meta = self.metadata.get(vec_id, {})
                skip = False
                for key, value in filter_dict.items():
                    if meta.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Compute cosine similarity
            similarity = np.dot(query_vec, stored_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
            )
            similarities[vec_id] = similarity
        
        # Sort by similarity and return top_k
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [
            {
                'id': vec_id,
                'score': float(score),
                'metadata': self.metadata.get(vec_id, {})
            }
            for vec_id, score in sorted_results
        ]
    
    def delete(self, vector_ids: List[str]):
        """Delete vectors by IDs."""
        for vec_id in vector_ids:
            self.vectors.pop(vec_id, None)
            self.metadata.pop(vec_id, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'name': self.name,
            'vector_count': len(self.vectors),
            'dimension': self.dimension,
            'index_built': self.index_built
        }


class MockPinecone(MockVectorDB):
    """Mock Pinecone implementation."""
    
    def __init__(self, dimension: int = 512):
        super().__init__("Pinecone", dimension)
        print(f"Initialized mock Pinecone with dimension {dimension}")
    
    def build_index(self):
        """Build/optimize index for better performance."""
        time.sleep(0.1)  # Simulate index building time
        self.index_built = True
        print("Pinecone index built successfully")


class MockWeaviate(MockVectorDB):
    """Mock Weaviate implementation."""
    
    def __init__(self, dimension: int = 512):
        super().__init__("Weaviate", dimension)
        self.schema_created = False
        print(f"Initialized mock Weaviate with dimension {dimension}")
    
    def create_schema(self, class_name: str = "Document"):
        """Create schema for storing vectors."""
        self.schema_created = True
        self.class_name = class_name
        print(f"Created Weaviate schema for class '{class_name}'")
    
    def query_with_graphql(self, query: str) -> List[Dict]:
        """Mock GraphQL query interface."""
        # Simplified mock - in reality, this would parse GraphQL
        return self.query(np.random.randn(self.dimension), top_k=5)


class MockMilvus(MockVectorDB):
    """Mock Milvus implementation."""
    
    def __init__(self, dimension: int = 512):
        super().__init__("Milvus", dimension)
        self.collection_created = False
        print(f"Initialized mock Milvus with dimension {dimension}")
    
    def create_collection(self, collection_name: str = "clip_embeddings"):
        """Create collection for storing vectors."""
        self.collection_created = True
        self.collection_name = collection_name
        print(f"Created Milvus collection '{collection_name}'")
    
    def create_index(self, index_type: str = "IVF_FLAT"):
        """Create index for efficient similarity search."""
        self.index_type = index_type
        self.index_built = True
        print(f"Created Milvus index with type '{index_type}'")


@dataclass
class InferenceConfig:
    """Configuration for inference optimization."""
    batch_size: int = 32
    use_half_precision: bool = False
    use_torch_compile: bool = False
    use_onnx: bool = False
    max_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EfficientCLIPInference:
    """Optimized CLIP inference with various acceleration techniques."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", 
                 config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Apply optimizations
        self._apply_optimizations()
        
        print(f"Initialized efficient CLIP inference on {self.device}")
        print(f"Optimizations: half_precision={self.config.use_half_precision}, "
              f"torch_compile={self.config.use_torch_compile}")
    
    def _apply_optimizations(self):
        """Apply various optimization techniques."""
        self.model.eval()
        
        # Half precision
        if self.config.use_half_precision and self.device.type == "cuda":
            self.model = self.model.half()
            print("Applied half precision (FP16)")
        
        # Torch compile (PyTorch 2.0+)
        if self.config.use_torch_compile:
            try:
                self.model = torch.compile(self.model)
                print("Applied torch.compile optimization")
            except Exception as e:
                print(f"Could not apply torch.compile: {e}")
        
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
    
    @contextmanager
    def inference_mode(self):
        """Context manager for optimized inference."""
        with torch.no_grad():
            with torch.inference_mode():
                yield
    
    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts in batches for efficiency."""
        all_embeddings = []
        
        with self.inference_mode():
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Move to device and handle precision
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if self.config.use_half_precision and self.device.type == "cuda":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v 
                             for k, v in inputs.items()}
                
                text_features = self.model.get_text_features(**inputs)
                text_features = F.normalize(text_features, dim=-1)
                
                all_embeddings.append(text_features.cpu().float().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_images_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Encode images in batches for efficiency."""
        all_embeddings = []
        
        with self.inference_mode():
            for i in range(0, len(images), self.config.batch_size):
                batch_images = images[i:i + self.config.batch_size]
                
                inputs = self.processor(
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if self.config.use_half_precision and self.device.type == "cuda":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v 
                             for k, v in inputs.items()}
                
                image_features = self.model.get_image_features(**inputs)
                image_features = F.normalize(image_features, dim=-1)
                
                all_embeddings.append(image_features.cpu().float().numpy())
        
        return np.vstack(all_embeddings)
    
    def benchmark_inference(self, num_texts: int = 100, num_images: int = 50) -> Dict[str, float]:
        """Benchmark inference performance."""
        print(f"\nBenchmarking inference performance...")
        
        # Generate dummy data
        dummy_texts = [f"This is sample text number {i}" for i in range(num_texts)]
        dummy_images = [Image.new('RGB', (224, 224), color='red') for _ in range(num_images)]
        
        # Warm up
        _ = self.encode_texts_batch(dummy_texts[:5])
        _ = self.encode_images_batch(dummy_images[:5])
        
        # Benchmark text encoding
        start_time = time.time()
        text_embeddings = self.encode_texts_batch(dummy_texts)
        text_time = time.time() - start_time
        
        # Benchmark image encoding
        start_time = time.time()
        image_embeddings = self.encode_images_batch(dummy_images)
        image_time = time.time() - start_time
        
        results = {
            'text_throughput': num_texts / text_time,
            'image_throughput': num_images / image_time,
            'text_latency_ms': (text_time / num_texts) * 1000,
            'image_latency_ms': (image_time / num_images) * 1000
        }
        
        print(f"Text encoding: {results['text_throughput']:.1f} texts/sec, "
              f"{results['text_latency_ms']:.2f} ms/text")
        print(f"Image encoding: {results['image_throughput']:.1f} images/sec, "
              f"{results['image_latency_ms']:.2f} ms/image")
        
        return results


class AsyncBatchProcessor:
    """Asynchronous batch processor for high-throughput inference."""
    
    def __init__(self, inference_engine: EfficientCLIPInference, max_queue_size: int = 1000):
        self.inference_engine = inference_engine
        self.text_queue = queue.Queue(maxsize=max_queue_size)
        self.image_queue = queue.Queue(maxsize=max_queue_size)
        self.results = {}
        self.processing = False
        self.worker_thread = None
    
    def start_processing(self):
        """Start the async processing worker."""
        if self.processing:
            return
        
        self.processing = True
        self.worker_thread = threading.Thread(target=self._process_batches)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        print("Started async batch processor")
    
    def stop_processing(self):
        """Stop the async processing worker."""
        self.processing = False
        if self.worker_thread:
            self.worker_thread.join()
        print("Stopped async batch processor")
    
    def _process_batches(self):
        """Worker function to process batches asynchronously."""
        while self.processing:
            # Process text batch
            text_batch = []
            text_ids = []
            
            try:
                # Collect batch
                while len(text_batch) < self.inference_engine.config.batch_size:
                    try:
                        item_id, text = self.text_queue.get(timeout=0.1)
                        text_batch.append(text)
                        text_ids.append(item_id)
                    except queue.Empty:
                        break
                
                # Process if we have items
                if text_batch:
                    embeddings = self.inference_engine.encode_texts_batch(text_batch)
                    for item_id, embedding in zip(text_ids, embeddings):
                        self.results[item_id] = embedding
            
            except Exception as e:
                print(f"Error processing text batch: {e}")
            
            time.sleep(0.01)  # Small delay to prevent busy waiting
    
    def submit_text(self, item_id: str, text: str) -> bool:
        """Submit text for async processing."""
        try:
            self.text_queue.put((item_id, text), timeout=1.0)
            return True
        except queue.Full:
            return False
    
    def get_result(self, item_id: str, timeout: float = 10.0) -> Optional[np.ndarray]:
        """Get processing result."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if item_id in self.results:
                return self.results.pop(item_id)
            time.sleep(0.01)
        return None


class ScalableMultimodalSearch:
    """Scalable multimodal search system with vector database integration."""
    
    def __init__(self, vector_db_type: str = "pinecone", inference_config: InferenceConfig = None):
        self.inference_config = inference_config or InferenceConfig()
        self.inference_engine = EfficientCLIPInference(config=self.inference_config)
        
        # Initialize vector database
        self.vector_db = self._init_vector_db(vector_db_type)
        
        # Initialize async processor
        self.async_processor = AsyncBatchProcessor(self.inference_engine)
        
        print(f"Initialized scalable search with {vector_db_type} backend")
    
    def _init_vector_db(self, db_type: str) -> MockVectorDB:
        """Initialize the specified vector database."""
        if db_type.lower() == "pinecone":
            db = MockPinecone()
            db.build_index()
            return db
        elif db_type.lower() == "weaviate":
            db = MockWeaviate()
            db.create_schema()
            return db
        elif db_type.lower() == "milvus":
            db = MockMilvus()
            db.create_collection()
            db.create_index()
            return db
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")
    
    def index_texts(self, texts: List[str], metadata: List[Dict] = None, 
                   batch_size: int = 100) -> Dict[str, Any]:
        """Index texts into the vector database."""
        print(f"Indexing {len(texts)} texts...")
        
        if metadata is None:
            metadata = [{'type': 'text', 'index': i} for i in range(len(texts))]
        
        start_time = time.time()
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Indexing texts"):
            batch_texts = texts[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            
            # Encode batch
            embeddings = self.inference_engine.encode_texts_batch(batch_texts)
            
            # Prepare for vector DB
            vectors_to_upsert = []
            for j, (text, embedding, meta) in enumerate(zip(batch_texts, embeddings, batch_metadata)):
                vector_id = f"text_{i + j}"
                vectors_to_upsert.append((vector_id, embedding.tolist(), meta))
            
            # Upsert to vector DB
            self.vector_db.upsert(vectors_to_upsert)
        
        indexing_time = time.time() - start_time
        
        stats = {
            'indexed_count': len(texts),
            'indexing_time_seconds': indexing_time,
            'throughput_texts_per_second': len(texts) / indexing_time
        }
        
        print(f"Indexed {len(texts)} texts in {indexing_time:.2f}s "
              f"({stats['throughput_texts_per_second']:.1f} texts/sec)")
        
        return stats
    
    def index_images(self, images: List[Image.Image], metadata: List[Dict] = None,
                    batch_size: int = 50) -> Dict[str, Any]:
        """Index images into the vector database."""
        print(f"Indexing {len(images)} images...")
        
        if metadata is None:
            metadata = [{'type': 'image', 'index': i} for i in range(len(images))]
        
        start_time = time.time()
        
        for i in tqdm(range(0, len(images), batch_size), desc="Indexing images"):
            batch_images = images[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            
            embeddings = self.inference_engine.encode_images_batch(batch_images)
            
            vectors_to_upsert = []
            for j, (embedding, meta) in enumerate(zip(embeddings, batch_metadata)):
                vector_id = f"image_{i + j}"
                vectors_to_upsert.append((vector_id, embedding.tolist(), meta))
            
            self.vector_db.upsert(vectors_to_upsert)
        
        indexing_time = time.time() - start_time
        
        stats = {
            'indexed_count': len(images),
            'indexing_time_seconds': indexing_time,
            'throughput_images_per_second': len(images) / indexing_time
        }
        
        print(f"Indexed {len(images)} images in {indexing_time:.2f}s "
              f"({stats['throughput_images_per_second']:.1f} images/sec)")
        
        return stats
    
    def search_with_text(self, query: str, top_k: int = 10, 
                        filter_dict: Dict = None) -> List[Dict]:
        """Search using text query."""
        # Encode query
        query_embedding = self.inference_engine.encode_texts_batch([query])[0]
        
        # Search in vector DB
        results = self.vector_db.query(
            query_embedding.tolist(),
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return results
    
    def search_with_image(self, image: Image.Image, top_k: int = 10,
                         filter_dict: Dict = None) -> List[Dict]:
        """Search using image query."""
        # Encode query
        query_embedding = self.inference_engine.encode_images_batch([image])[0]
        
        # Search in vector DB
        results = self.vector_db.query(
            query_embedding.tolist(),
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return results
    
    def benchmark_search_performance(self, num_queries: int = 100) -> Dict[str, float]:
        """Benchmark search performance."""
        print(f"\nBenchmarking search performance with {num_queries} queries...")
        
        # Generate random queries
        dummy_texts = [f"Search query number {i}" for i in range(num_queries)]
        
        # Warm up
        _ = self.search_with_text(dummy_texts[0])
        
        # Benchmark
        start_time = time.time()
        for text in tqdm(dummy_texts, desc="Running search queries"):
            _ = self.search_with_text(text, top_k=5)
        
        search_time = time.time() - start_time
        
        results = {
            'queries_per_second': num_queries / search_time,
            'average_latency_ms': (search_time / num_queries) * 1000
        }
        
        print(f"Search performance: {results['queries_per_second']:.1f} queries/sec, "
              f"{results['average_latency_ms']:.2f} ms/query")
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        db_stats = self.vector_db.get_stats()
        
        return {
            'vector_database': db_stats,
            'inference_config': {
                'batch_size': self.inference_config.batch_size,
                'device': self.inference_config.device,
                'use_half_precision': self.inference_config.use_half_precision,
                'use_torch_compile': self.inference_config.use_torch_compile
            }
        }


def compare_vector_databases():
    """Compare different vector database backends."""
    print("\n=== Vector Database Comparison ===")
    
    databases = ["pinecone", "weaviate", "milvus"]
    sample_texts = [f"Sample document {i}" for i in range(100)]
    
    results = {}
    
    for db_name in databases:
        print(f"\nTesting {db_name.upper()}...")
        
        # Initialize system
        search_system = ScalableMultimodalSearch(vector_db_type=db_name)
        
        # Index documents
        start_time = time.time()
        indexing_stats = search_system.index_texts(sample_texts)
        indexing_time = time.time() - start_time
        
        # Benchmark search
        search_stats = search_system.benchmark_search_performance(num_queries=20)
        
        results[db_name] = {
            'indexing_throughput': indexing_stats['throughput_texts_per_second'],
            'search_throughput': search_stats['queries_per_second'],
            'search_latency_ms': search_stats['average_latency_ms']
        }
    
    # Display comparison
    print(f"\n{'Database':<10} {'Index (docs/s)':<15} {'Search (q/s)':<15} {'Latency (ms)':<12}")
    print("-" * 52)
    
    for db_name, stats in results.items():
        print(f"{db_name:<10} {stats['indexing_throughput']:<15.1f} "
              f"{stats['search_throughput']:<15.1f} {stats['search_latency_ms']:<12.2f}")
    
    return results


def demonstrate_optimization_techniques():
    """Demonstrate various optimization techniques."""
    print("\n=== Optimization Techniques Comparison ===")
    
    configs = [
        InferenceConfig(batch_size=1, use_half_precision=False, use_torch_compile=False),
        InferenceConfig(batch_size=32, use_half_precision=False, use_torch_compile=False),
        InferenceConfig(batch_size=32, use_half_precision=True, use_torch_compile=False),
        InferenceConfig(batch_size=32, use_half_precision=True, use_torch_compile=True),
    ]
    
    config_names = [
        "Baseline (batch=1)",
        "Batching (batch=32)",
        "Batching + FP16",
        "Batching + FP16 + Compile"
    ]
    
    results = []
    
    for config, name in zip(configs, config_names):
        print(f"\nTesting: {name}")
        
        try:
            engine = EfficientCLIPInference(config=config)
            benchmark_results = engine.benchmark_inference(num_texts=50, num_images=25)
            
            results.append({
                'name': name,
                'text_throughput': benchmark_results['text_throughput'],
                'image_throughput': benchmark_results['image_throughput'],
                'text_latency': benchmark_results['text_latency_ms'],
                'image_latency': benchmark_results['image_latency_ms']
            })
        
        except Exception as e:
            print(f"Error with {name}: {e}")
            results.append({
                'name': name,
                'text_throughput': 0,
                'image_throughput': 0,
                'text_latency': float('inf'),
                'image_latency': float('inf')
            })
    
    # Display results
    print(f"\n{'Configuration':<25} {'Text (docs/s)':<15} {'Image (imgs/s)':<15} {'Text Lat (ms)':<15}")
    print("-" * 70)
    
    baseline_text = results[0]['text_throughput'] if results[0]['text_throughput'] > 0 else 1
    baseline_image = results[0]['image_throughput'] if results[0]['image_throughput'] > 0 else 1
    
    for result in results:
        speedup_text = result['text_throughput'] / baseline_text
        speedup_image = result['image_throughput'] / baseline_image
        
        print(f"{result['name']:<25} {result['text_throughput']:<15.1f} "
              f"{result['image_throughput']:<15.1f} {result['text_latency']:<15.2f}")
        print(f"{'  Speedup:':<25} {speedup_text:<15.2f}x {speedup_image:<15.2f}x")
    
    return results


def create_sample_large_dataset():
    """Create a sample large dataset for scaling demonstration."""
    # In a real scenario, you'd load from a large dataset
    # Here we create synthetic data for demonstration
    
    categories = ["technology", "nature", "food", "travel", "sports"]
    
    texts = []
    metadata = []
    
    for i in range(1000):  # 1000 sample documents
        category = categories[i % len(categories)]
        text = f"This is a {category} document about {category} topics. Document number {i}."
        
        texts.append(text)
        metadata.append({
            'category': category,
            'document_id': i,
            'type': 'text'
        })
    
    # Create some sample images (placeholder)
    images = [Image.new('RGB', (224, 224), color=f'C{i % 10}') for i in range(100)]
    image_metadata = [{'type': 'image', 'image_id': i, 'category': categories[i % len(categories)]} 
                     for i in range(100)]
    
    return texts, metadata, images, image_metadata


def demo_scaling():
    """Demonstrate scaling techniques and vector databases."""
    print("Scaling with Vector Databases and Efficient Inference - Step 5")
    print("=" * 70)
    
    # 1. Compare optimization techniques
    print("1. Optimization Techniques Comparison")
    optimization_results = demonstrate_optimization_techniques()
    
    # 2. Compare vector databases
    print("\n2. Vector Database Comparison")
    db_comparison = compare_vector_databases()
    
    # 3. Demonstrate large-scale indexing and search
    print("\n3. Large-Scale Search System Demo")
    
    # Create sample dataset
    texts, text_metadata, images, image_metadata = create_sample_large_dataset()
    
    # Initialize scalable search system
    config = InferenceConfig(
        batch_size=32,
        use_half_precision=torch.cuda.is_available(),
        use_torch_compile=False  # Set to True if you have PyTorch 2.0+
    )
    
    search_system = ScalableMultimodalSearch(
        vector_db_type="pinecone",
        inference_config=config
    )
    
    # Index the dataset
    print("\nIndexing large dataset...")
    text_indexing_stats = search_system.index_texts(texts[:500], text_metadata[:500])  # Subset for demo
    image_indexing_stats = search_system.index_images(images[:50], image_metadata[:50])
    
    # Demonstrate search
    print("\nPerforming sample searches...")
    
    # Text search
    text_query = "technology and innovation"
    text_results = search_system.search_with_text(text_query, top_k=5)
    
    print(f"Text query: '{text_query}'")
    print("Top results:")
    for i, result in enumerate(text_results[:3], 1):
        print(f"  {i}. Score: {result['score']:.3f}, Category: {result['metadata'].get('category', 'N/A')}")
    
    # Search with filters
    filtered_results = search_system.search_with_text(
        "nature and environment", 
        top_k=5,
        filter_dict={'category': 'nature'}
    )
    
    print(f"\nFiltered search (category='nature'):")
    for i, result in enumerate(filtered_results[:3], 1):
        print(f"  {i}. Score: {result['score']:.3f}, Category: {result['metadata'].get('category', 'N/A')}")
    
    # System statistics
    stats = search_system.get_system_stats()
    print(f"\nSystem Statistics:")
    print(f"  Vector Database: {stats['vector_database']['name']}")
    print(f"  Total Vectors: {stats['vector_database']['vector_count']}")
    print(f"  Inference Device: {stats['inference_config']['device']}")
    print(f"  Batch Size: {stats['inference_config']['batch_size']}")
    
    # 4. Demonstrate async processing
    print("\n4. Asynchronous Processing Demo")
    
    async_processor = AsyncBatchProcessor(search_system.inference_engine)
    async_processor.start_processing()
    
    # Submit some texts for async processing
    async_texts = ["async text 1", "async text 2", "async text 3"]
    for i, text in enumerate(async_texts):
        success = async_processor.submit_text(f"async_{i}", text)
        print(f"Submitted '{text}': {'Success' if success else 'Failed'}")
    
    # Wait for results
    time.sleep(2)  # Give time for processing
    
    for i in range(len(async_texts)):
        result = async_processor.get_result(f"async_{i}", timeout=5.0)
        if result is not None:
            print(f"Got result for async_{i}: shape {result.shape}")
        else:
            print(f"No result for async_{i}")
    
    async_processor.stop_processing()
    
    print("\n" + "=" * 70)
    print("Scaling demo completed!")
    
    print(f"\nKey Performance Insights:")
    print(f"1. Batching provides significant speedup over single-item processing")
    print(f"2. Half-precision (FP16) can improve throughput on modern GPUs")
    print(f"3. Vector databases enable fast similarity search at scale")
    print(f"4. Async processing allows for better resource utilization")
    print(f"5. Proper indexing and filtering are crucial for large-scale systems")
    
    return {
        'optimization_results': optimization_results,
        'db_comparison': db_comparison,
        'indexing_stats': {
            'text': text_indexing_stats,
            'image': image_indexing_stats
        },
        'system_stats': stats
    }


if __name__ == "__main__":
    demo_scaling()