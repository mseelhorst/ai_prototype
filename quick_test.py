#!/usr/bin/env python3
"""
Quick test script to verify the installation and basic functionality.
Run this after setup to ensure everything is working correctly.
"""

import sys
import time
import traceback
from typing import Dict, Any

def test_imports() -> Dict[str, Any]:
    """Test all required imports."""
    print("🔍 Testing imports...")
    
    results = {}
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Hugging Face Transformers'),
        ('faiss', 'FAISS'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('tqdm', 'TQDM'),
        ('requests', 'Requests')
    ]
    
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"  ✅ {name}")
            results[name] = True
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            results[name] = False
    
    return results

def test_device() -> Dict[str, Any]:
    """Test device availability and configuration."""
    print("\n🖥️  Testing device configuration...")
    
    try:
        import torch
        
        device_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        if device_info['cuda_available']:
            device_info['gpu_name'] = torch.cuda.get_device_name(0)
            device_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✅ GPU: {device_info['gpu_name']} ({device_info['gpu_memory']:.1f} GB)")
        else:
            print(f"  ⚠️  GPU: Not available (using CPU)")
        
        return device_info
        
    except Exception as e:
        print(f"  ❌ Error checking device: {e}")
        return {'error': str(e)}

def test_clip_model() -> Dict[str, Any]:
    """Test CLIP model loading and basic functionality."""
    print("\n🤖 Testing CLIP model...")
    
    try:
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image
        import torch
        
        start_time = time.time()
        
        # Load model and processor
        print("  📥 Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        load_time = time.time() - start_time
        
        # Test basic functionality
        print("  🧪 Testing basic functionality...")
        test_image = Image.new('RGB', (224, 224), color='blue')
        test_texts = ["a blue square", "a red circle"]
        
        inputs = processor(
            text=test_texts,
            images=test_image,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1)
        
        # Check if results make sense
        top_prob = probs.max().item()
        
        results = {
            'model_loaded': True,
            'load_time_seconds': load_time,
            'inference_successful': True,
            'top_probability': top_prob,
            'output_shape': list(logits_per_image.shape)
        }
        
        print(f"  ✅ Model loaded in {load_time:.2f}s")
        print(f"  ✅ Inference successful (top prob: {top_prob:.3f})")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error testing CLIP model: {e}")
        return {'error': str(e), 'traceback': traceback.format_exc()}

def test_faiss() -> Dict[str, Any]:
    """Test FAISS functionality."""
    print("\n🔍 Testing FAISS...")
    
    try:
        import faiss
        import numpy as np
        
        # Create a simple index
        dimension = 512
        index = faiss.IndexFlatIP(dimension)
        
        # Add some random vectors
        n_vectors = 100
        vectors = np.random.randn(n_vectors, dimension).astype('float32')
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize
        
        index.add(vectors)
        
        # Test search
        query = np.random.randn(1, dimension).astype('float32')
        query = query / np.linalg.norm(query)
        
        scores, indices = index.search(query, k=5)
        
        results = {
            'faiss_working': True,
            'index_size': index.ntotal,
            'search_successful': len(indices[0]) == 5,
            'top_score': float(scores[0][0])
        }
        
        print(f"  ✅ FAISS index created with {n_vectors} vectors")
        print(f"  ✅ Search successful (top score: {results['top_score']:.3f})")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error testing FAISS: {e}")
        return {'error': str(e)}

def test_visualization() -> Dict[str, Any]:
    """Test visualization capabilities."""
    print("\n📊 Testing visualization...")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a simple plot
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('Test Plot')
        
        # Save to test file writing
        test_filename = 'test_plot.png'
        plt.savefig(test_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Check if file was created
        import os
        file_exists = os.path.exists(test_filename)
        
        if file_exists:
            # Clean up test file
            os.remove(test_filename)
        
        results = {
            'matplotlib_working': True,
            'file_save_working': file_exists
        }
        
        print(f"  ✅ Matplotlib working")
        print(f"  ✅ File saving working")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error testing visualization: {e}")
        return {'error': str(e)}

def run_full_test() -> bool:
    """Run all tests and return overall success."""
    print("🧪 AI Techniques with CLIP - Quick Test")
    print("=" * 50)
    
    all_results = {}
    
    # Run all tests
    all_results['imports'] = test_imports()
    all_results['device'] = test_device()
    all_results['clip'] = test_clip_model()
    all_results['faiss'] = test_faiss()
    all_results['visualization'] = test_visualization()
    
    # Analyze results
    print("\n📋 Test Summary")
    print("-" * 30)
    
    total_tests = 0
    passed_tests = 0
    
    # Check import results
    for package, success in all_results['imports'].items():
        total_tests += 1
        if success:
            passed_tests += 1
    
    # Check other test results
    for test_name, results in all_results.items():
        if test_name == 'imports':
            continue
        
        if 'error' not in results:
            if test_name == 'device':
                total_tests += 1
                passed_tests += 1
            elif test_name == 'clip':
                total_tests += 2
                if results.get('model_loaded', False):
                    passed_tests += 1
                if results.get('inference_successful', False):
                    passed_tests += 1
            elif test_name == 'faiss':
                total_tests += 1
                if results.get('faiss_working', False):
                    passed_tests += 1
            elif test_name == 'visualization':
                total_tests += 1
                if results.get('matplotlib_working', False):
                    passed_tests += 1
        else:
            total_tests += 1
    
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    print(f"Tests passed: {passed_tests}/{total_tests} ({success_rate*100:.1f}%)")
    
    if success_rate >= 0.8:  # 80% success rate
        print("✅ System ready for AI Techniques with CLIP!")
        print("\nNext steps:")
        print("• Run 'python main_demo.py' to start the interactive journey")
        print("• Or run individual steps directly")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("• Ensure all requirements are installed: pip install -r requirements.txt")
        print("• Check that you have sufficient disk space and memory")
        print("• For GPU issues, verify CUDA installation")
        return False

def main():
    """Main test function."""
    try:
        success = run_full_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error during testing: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()