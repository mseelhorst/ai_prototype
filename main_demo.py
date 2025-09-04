#!/usr/bin/env python3
"""
Main Demo Script - Complete AI Techniques with CLIP Learning Journey

This script provides an interactive menu to run all the demonstration scripts,
allowing users to explore different aspects of CLIP and multimodal AI.
"""

import sys
import os
import subprocess
import time
from typing import Dict, List, Callable
import argparse

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class DemoRunner:
    """Main demo runner with interactive menu."""
    
    def __init__(self):
        self.demos = {
            '1': {
                'name': 'Step 1: CLIP Hands-on Demo',
                'description': 'Zero-shot classification and image-text retrieval',
                'script': 'step1_clip_demo.py',
                'duration': '~3-5 minutes',
                'requirements': 'Internet connection for downloading images'
            },
            '2': {
                'name': 'Step 2: Search Engine with FAISS',
                'description': 'Build a toy search engine with vector storage',
                'script': 'step2_search_engine.py',
                'duration': '~5-7 minutes',
                'requirements': 'FAISS library, ~1GB RAM'
            },
            '3': {
                'name': 'Step 3: Fine-tune CLIP',
                'description': 'Fine-tune CLIP on domain-specific dataset',
                'script': 'step3_finetune_clip.py',
                'duration': '~10-15 minutes',
                'requirements': 'GPU recommended, ~4GB VRAM'
            },
            '4': {
                'name': 'Step 4: Experiment Design & Evaluation',
                'description': 'Learn experimental design and statistical analysis',
                'script': 'step4_experiment_design.py',
                'duration': '~3-5 minutes',
                'requirements': 'Scientific Python packages'
            },
            '5': {
                'name': 'Step 5: Scaling & Vector Databases',
                'description': 'Production-scale inference and vector databases',
                'script': 'step5_scaling.py',
                'duration': '~7-10 minutes',
                'requirements': 'GPU recommended for optimal performance'
            }
        }
        
        self.completed_demos = set()
    
    def print_header(self):
        """Print the main header."""
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("=" * 80)
        print("🚀 AI TECHNIQUES WITH CLIP - COMPLETE LEARNING JOURNEY 🚀")
        print("=" * 80)
        print(f"{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Welcome to the comprehensive CLIP learning experience!{Colors.ENDC}")
        print(f"{Colors.OKCYAN}This journey will take you from basic CLIP usage to production-scale deployment.{Colors.ENDC}")
        print()
    
    def print_system_info(self):
        """Print system information."""
        print(f"{Colors.BOLD}System Information:{Colors.ENDC}")
        
        # Check Python version
        python_version = sys.version.split()[0]
        print(f"  Python Version: {python_version}")
        
        # Check GPU availability
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"  GPU: {Colors.OKGREEN}✓{Colors.ENDC} {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                print(f"  GPU: {Colors.WARNING}✗ Not available (CPU-only mode){Colors.ENDC}")
        except ImportError:
            print(f"  GPU: {Colors.WARNING}? PyTorch not installed{Colors.ENDC}")
        
        # Check key dependencies
        dependencies = [
            ('torch', 'PyTorch'),
            ('transformers', 'Hugging Face Transformers'),
            ('faiss', 'FAISS'),
            ('PIL', 'Pillow'),
            ('matplotlib', 'Matplotlib'),
            ('numpy', 'NumPy')
        ]
        
        print(f"  Dependencies:")
        for module, name in dependencies:
            try:
                __import__(module)
                print(f"    {name}: {Colors.OKGREEN}✓{Colors.ENDC}")
            except ImportError:
                print(f"    {name}: {Colors.FAIL}✗{Colors.ENDC}")
        
        print()
    
    def print_menu(self):
        """Print the interactive menu."""
        print(f"{Colors.BOLD}Available Demonstrations:{Colors.ENDC}")
        print()
        
        for key, demo in self.demos.items():
            status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if key in self.completed_demos else " "
            print(f"  [{key}] {status} {Colors.BOLD}{demo['name']}{Colors.ENDC}")
            print(f"      {demo['description']}")
            print(f"      Duration: {demo['duration']}")
            print(f"      Requirements: {demo['requirements']}")
            print()
        
        print(f"  [a] Run All Demonstrations (Sequential)")
        print(f"  [i] Install/Check Dependencies")
        print(f"  [h] Help & Documentation")
        print(f"  [q] Quit")
        print()
    
    def check_dependencies(self):
        """Check and optionally install dependencies."""
        print(f"{Colors.BOLD}Dependency Check & Installation{Colors.ENDC}")
        print("=" * 50)
        
        # Check if requirements.txt exists
        if not os.path.exists('requirements.txt'):
            print(f"{Colors.FAIL}Error: requirements.txt not found!{Colors.ENDC}")
            return False
        
        # Try to install dependencies
        print("Checking dependencies from requirements.txt...")
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"{Colors.OKGREEN}✓ All dependencies installed successfully!{Colors.ENDC}")
                return True
            else:
                print(f"{Colors.FAIL}✗ Error installing dependencies:{Colors.ENDC}")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"{Colors.WARNING}Installation timed out. You may need to install manually.{Colors.ENDC}")
            return False
        except Exception as e:
            print(f"{Colors.FAIL}Error during installation: {e}{Colors.ENDC}")
            return False
    
    def run_demo(self, demo_key: str) -> bool:
        """Run a specific demonstration."""
        if demo_key not in self.demos:
            print(f"{Colors.FAIL}Invalid demo key: {demo_key}{Colors.ENDC}")
            return False
        
        demo = self.demos[demo_key]
        script_path = demo['script']
        
        if not os.path.exists(script_path):
            print(f"{Colors.FAIL}Script not found: {script_path}{Colors.ENDC}")
            return False
        
        print(f"{Colors.BOLD}Running: {demo['name']}{Colors.ENDC}")
        print(f"Description: {demo['description']}")
        print(f"Estimated duration: {demo['duration']}")
        print(f"Requirements: {demo['requirements']}")
        print()
        
        # Confirm before running
        response = input(f"Continue with this demonstration? [Y/n]: ").strip().lower()
        if response and response not in ['y', 'yes']:
            print("Demonstration cancelled.")
            return False
        
        print(f"{Colors.OKCYAN}Starting demonstration...{Colors.ENDC}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Run the demonstration script
            result = subprocess.run([
                sys.executable, script_path
            ], timeout=1800)  # 30 minute timeout
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print("-" * 60)
                print(f"{Colors.OKGREEN}✓ Demonstration completed successfully!{Colors.ENDC}")
                print(f"Duration: {duration:.1f} seconds")
                self.completed_demos.add(demo_key)
                
                # Show completion summary
                self.show_completion_summary(demo_key)
                return True
            else:
                print("-" * 60)
                print(f"{Colors.FAIL}✗ Demonstration failed with exit code {result.returncode}{Colors.ENDC}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"{Colors.WARNING}Demonstration timed out after 30 minutes.{Colors.ENDC}")
            return False
        except KeyboardInterrupt:
            print(f"{Colors.WARNING}Demonstration interrupted by user.{Colors.ENDC}")
            return False
        except Exception as e:
            print(f"{Colors.FAIL}Error running demonstration: {e}{Colors.ENDC}")
            return False
    
    def show_completion_summary(self, demo_key: str):
        """Show what was accomplished in the demo."""
        summaries = {
            '1': [
                "✓ Learned zero-shot image classification with CLIP",
                "✓ Explored image-text retrieval capabilities",
                "✓ Generated similarity visualizations",
                "📁 Files created: classification_result_*.png, similarity_matrix.png"
            ],
            '2': [
                "✓ Built a multimodal search engine",
                "✓ Integrated FAISS for efficient similarity search",
                "✓ Implemented search evaluation metrics",
                "📁 Files created: search_index.faiss, search_metadata.json"
            ],
            '3': [
                "✓ Fine-tuned CLIP on domain-specific data",
                "✓ Implemented contrastive learning loss",
                "✓ Compared pre-trained vs fine-tuned performance",
                "📁 Files created: clip_finetuned/ directory, training_curves.png"
            ],
            '4': [
                "✓ Understood contrastive learning theory",
                "✓ Learned experimental design best practices",
                "✓ Implemented statistical significance testing",
                "📁 Files created: embedding_space_evolution.png, experiments/ directory"
            ],
            '5': [
                "✓ Explored vector database integration",
                "✓ Implemented efficient inference techniques",
                "✓ Benchmarked scaling performance",
                "📁 Performance reports and optimization analysis generated"
            ]
        }
        
        print(f"\n{Colors.BOLD}What you accomplished:{Colors.ENDC}")
        for item in summaries.get(demo_key, []):
            print(f"  {item}")
        print()
    
    def run_all_demos(self):
        """Run all demonstrations in sequence."""
        print(f"{Colors.BOLD}Running All Demonstrations{Colors.ENDC}")
        print("=" * 40)
        print("This will run all 5 demonstrations in sequence.")
        print("Total estimated time: 30-45 minutes")
        print()
        
        response = input("Continue with all demonstrations? [Y/n]: ").strip().lower()
        if response and response not in ['y', 'yes']:
            print("Cancelled.")
            return
        
        total_start_time = time.time()
        successful_demos = 0
        
        for demo_key in sorted(self.demos.keys()):
            print(f"\n{Colors.HEADER}{'='*20} STEP {demo_key} {'='*20}{Colors.ENDC}")
            
            if self.run_demo(demo_key):
                successful_demos += 1
                print(f"{Colors.OKGREEN}Step {demo_key} completed successfully!{Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}Step {demo_key} failed.{Colors.ENDC}")
                
                # Ask if user wants to continue
                response = input("Continue with remaining demonstrations? [Y/n]: ").strip().lower()
                if response and response not in ['y', 'yes']:
                    break
            
            # Brief pause between demos
            if demo_key != '5':  # Don't pause after the last demo
                print(f"\n{Colors.OKCYAN}Preparing next demonstration...{Colors.ENDC}")
                time.sleep(2)
        
        # Final summary
        total_duration = time.time() - total_start_time
        print(f"\n{Colors.HEADER}{'='*20} FINAL SUMMARY {'='*20}{Colors.ENDC}")
        print(f"Completed: {successful_demos}/5 demonstrations")
        print(f"Total time: {total_duration/60:.1f} minutes")
        
        if successful_demos == 5:
            print(f"{Colors.OKGREEN}🎉 Congratulations! You've completed the entire CLIP learning journey!{Colors.ENDC}")
            print(f"{Colors.BOLD}Next steps:{Colors.ENDC}")
            print("  • Explore the generated files and visualizations")
            print("  • Try the techniques on your own datasets")
            print("  • Read the research papers mentioned in the code")
            print("  • Build production applications using the scaling techniques")
        else:
            print(f"{Colors.WARNING}Some demonstrations encountered issues.{Colors.ENDC}")
            print("Check the error messages above and ensure all dependencies are installed.")
    
    def show_help(self):
        """Show help and documentation."""
        print(f"{Colors.BOLD}Help & Documentation{Colors.ENDC}")
        print("=" * 30)
        print()
        print(f"{Colors.BOLD}About this Learning Journey:{Colors.ENDC}")
        print("This comprehensive tutorial covers advanced AI techniques using CLIP:")
        print("• Zero-shot learning and multimodal understanding")
        print("• Vector search and retrieval systems")
        print("• Fine-tuning and transfer learning")
        print("• Experimental design and evaluation")
        print("• Production scaling and optimization")
        print()
        
        print(f"{Colors.BOLD}Prerequisites:{Colors.ENDC}")
        print("• Python 3.8+ (Python 3.9+ recommended)")
        print("• 8GB+ RAM (16GB+ recommended)")
        print("• GPU with 4GB+ VRAM (optional but recommended)")
        print("• Internet connection for downloading models and images")
        print()
        
        print(f"{Colors.BOLD}Troubleshooting:{Colors.ENDC}")
        print("• If demos fail, try running 'pip install -r requirements.txt'")
        print("• For GPU issues, ensure CUDA and PyTorch are properly installed")
        print("• For memory issues, reduce batch sizes in the configuration")
        print("• Check the README.md file for detailed documentation")
        print()
        
        print(f"{Colors.BOLD}Learning Path:{Colors.ENDC}")
        print("1. Start with Step 1 to understand CLIP basics")
        print("2. Progress through Steps 2-5 in order")
        print("3. Each step builds on previous knowledge")
        print("4. Take time to explore generated files and visualizations")
        print("5. Experiment with the code and try your own modifications")
        print()
    
    def run(self):
        """Run the main interactive demo."""
        self.print_header()
        self.print_system_info()
        
        while True:
            self.print_menu()
            
            try:
                choice = input(f"{Colors.BOLD}Enter your choice: {Colors.ENDC}").strip().lower()
                print()
                
                if choice == 'q':
                    print(f"{Colors.OKCYAN}Thank you for exploring AI techniques with CLIP!{Colors.ENDC}")
                    print(f"{Colors.OKBLUE}Happy learning! 🚀{Colors.ENDC}")
                    break
                
                elif choice == 'a':
                    self.run_all_demos()
                
                elif choice == 'i':
                    self.check_dependencies()
                
                elif choice == 'h':
                    self.show_help()
                
                elif choice in self.demos:
                    self.run_demo(choice)
                
                else:
                    print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.ENDC}")
                
                # Pause before showing menu again
                input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")
                print("\n" + "="*80 + "\n")
                
            except KeyboardInterrupt:
                print(f"\n\n{Colors.OKCYAN}Goodbye! 👋{Colors.ENDC}")
                break
            except EOFError:
                print(f"\n\n{Colors.OKCYAN}Goodbye! 👋{Colors.ENDC}")
                break


def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(
        description="AI Techniques with CLIP - Interactive Learning Journey"
    )
    parser.add_argument(
        '--step', '-s',
        choices=['1', '2', '3', '4', '5'],
        help='Run a specific step directly'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all demonstrations sequentially'
    )
    parser.add_argument(
        '--check-deps', '-c',
        action='store_true',
        help='Check and install dependencies'
    )
    
    args = parser.parse_args()
    
    demo_runner = DemoRunner()
    
    # Handle command line arguments
    if args.check_deps:
        demo_runner.print_header()
        demo_runner.check_dependencies()
        return
    
    if args.step:
        demo_runner.print_header()
        demo_runner.run_demo(args.step)
        return
    
    if args.all:
        demo_runner.print_header()
        demo_runner.run_all_demos()
        return
    
    # Run interactive mode
    demo_runner.run()


if __name__ == "__main__":
    main()