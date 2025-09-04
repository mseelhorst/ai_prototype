#!/usr/bin/env python3
"""
Step 4: Experiment Design and Evaluation Framework
Learn the science behind contrastive learning and practice rigorous experiment design.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    experiment_name: str
    model_name: str
    dataset_name: str
    batch_size: int
    learning_rate: float
    epochs: int
    temperature: float = 0.07
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    seed: int = 42
    notes: str = ""

@dataclass
class ExperimentResults:
    """Results from an experiment."""
    config: ExperimentConfig
    metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    timestamp: str
    duration_seconds: float
    final_model_path: Optional[str] = None

class ContrastiveLearningTheory:
    """Demonstrate contrastive learning concepts and theory."""
    
    @staticmethod
    def demonstrate_contrastive_loss():
        """Demonstrate how contrastive loss works."""
        print("=== Contrastive Learning Theory Demo ===")
        print("\n1. InfoNCE Loss (used in CLIP):")
        print("   L = -log(exp(sim(x_i, y_i)/τ) / Σ_j exp(sim(x_i, y_j)/τ))")
        print("   Where:")
        print("   - x_i, y_i are positive pairs (image, matching text)")
        print("   - x_i, y_j (j≠i) are negative pairs")
        print("   - τ is temperature parameter")
        print("   - sim() is cosine similarity")
        
        # Simulate embeddings
        batch_size = 4
        embed_dim = 512
        temperature = 0.07
        
        # Create synthetic embeddings
        torch.manual_seed(42)
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        
        # Make some pairs more similar (positive pairs)
        for i in range(batch_size):
            text_embeds[i] = 0.7 * text_embeds[i] + 0.3 * image_embeds[i]
            text_embeds[i] = F.normalize(text_embeds[i], dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeds, text_embeds.T)
        
        print(f"\n2. Example with batch_size={batch_size}:")
        print("Similarity matrix (image-text pairs):")
        print(similarity_matrix.detach().numpy().round(3))
        
        # Compute InfoNCE loss
        labels = torch.arange(batch_size)
        logits_per_image = similarity_matrix / temperature
        logits_per_text = similarity_matrix.T / temperature
        
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        total_loss = (loss_i2t + loss_t2i) / 2
        
        print(f"\nLoss components:")
        print(f"  Image→Text loss: {loss_i2t:.4f}")
        print(f"  Text→Image loss: {loss_t2i:.4f}")
        print(f"  Total loss: {total_loss:.4f}")
        
        # Show effect of temperature
        temperatures = [0.01, 0.07, 0.2, 0.5]
        print(f"\n3. Effect of temperature parameter:")
        print(f"{'Temperature':<12} {'Loss':<8} {'Max Prob':<10}")
        print("-" * 30)
        
        for temp in temperatures:
            logits = similarity_matrix / temp
            loss = F.cross_entropy(logits, labels)
            max_prob = F.softmax(logits, dim=-1).max().item()
            print(f"{temp:<12} {loss:<8.4f} {max_prob:<10.4f}")
        
        return {
            'similarity_matrix': similarity_matrix,
            'total_loss': total_loss,
            'temperature_analysis': temperatures
        }
    
    @staticmethod
    def visualize_embedding_space():
        """Visualize how contrastive learning organizes embedding space."""
        print("\n=== Embedding Space Visualization ===")
        
        # Create synthetic data representing different stages of training
        np.random.seed(42)
        n_samples = 100
        n_classes = 5
        
        # Stage 1: Random embeddings (before training)
        random_embeds = np.random.randn(n_samples, 2)
        random_labels = np.repeat(range(n_classes), n_samples // n_classes)
        
        # Stage 2: Partially trained (some clustering)
        partial_embeds = np.zeros((n_samples, 2))
        for i in range(n_classes):
            mask = random_labels == i
            center = np.random.randn(2) * 2
            partial_embeds[mask] = center + np.random.randn(np.sum(mask), 2) * 0.8
        
        # Stage 3: Well-trained (clear clusters)
        trained_embeds = np.zeros((n_samples, 2))
        for i in range(n_classes):
            mask = random_labels == i
            center = np.random.randn(2) * 3
            trained_embeds[mask] = center + np.random.randn(np.sum(mask), 2) * 0.3
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        stages = [
            (random_embeds, "Before Training\n(Random Embeddings)"),
            (partial_embeds, "During Training\n(Partial Clustering)"),
            (trained_embeds, "After Training\n(Clear Clusters)")
        ]
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for ax, (embeds, title) in zip(axes, stages):
            for i in range(n_classes):
                mask = random_labels == i
                ax.scatter(embeds[mask, 0], embeds[mask, 1], 
                          c=[colors[i]], label=f'Class {i}', alpha=0.7)
            
            ax.set_title(title)
            ax.set_xlabel('Embedding Dimension 1')
            ax.set_ylabel('Embedding Dimension 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('embedding_space_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Embedding space evolution saved as 'embedding_space_evolution.png'")
        
        # Compute and display metrics for each stage
        from sklearn.metrics import silhouette_score
        
        print("\nQuantitative Analysis:")
        print(f"{'Stage':<20} {'Silhouette Score':<15} {'Intra-class Var':<15}")
        print("-" * 50)
        
        for embeds, title in stages:
            sil_score = silhouette_score(embeds, random_labels)
            
            # Compute intra-class variance
            intra_var = 0
            for i in range(n_classes):
                mask = random_labels == i
                class_embeds = embeds[mask]
                if len(class_embeds) > 1:
                    intra_var += np.var(class_embeds)
            intra_var /= n_classes
            
            stage_name = title.split('\n')[0]
            print(f"{stage_name:<20} {sil_score:<15.4f} {intra_var:<15.4f}")


class ExperimentTracker:
    """Track and compare multiple experiments."""
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
        self.results_file = os.path.join(experiments_dir, "experiment_results.json")
        self.load_results()
    
    def load_results(self):
        """Load existing experiment results."""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                data = json.load(f)
                self.results = [ExperimentResults(**result) for result in data]
        else:
            self.results = []
    
    def save_results(self):
        """Save experiment results to file."""
        data = [asdict(result) for result in self.results]
        with open(self.results_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_experiment(self, result: ExperimentResults):
        """Add a new experiment result."""
        self.results.append(result)
        self.save_results()
    
    def compare_experiments(self, metric: str = "avg_retrieval_r1"):
        """Compare experiments based on a specific metric."""
        if not self.results:
            print("No experiments to compare.")
            return
        
        df_data = []
        for result in self.results:
            row = {
                'experiment_name': result.config.experiment_name,
                'model_name': result.config.model_name,
                'learning_rate': result.config.learning_rate,
                'batch_size': result.config.batch_size,
                'temperature': result.config.temperature,
                'epochs': result.config.epochs,
                metric: result.metrics.get(metric, 0),
                'duration': result.duration_seconds
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        print(f"\n=== Experiment Comparison (sorted by {metric}) ===")
        df_sorted = df.sort_values(metric, ascending=False)
        print(df_sorted.to_string(index=False, float_format='%.4f'))
        
        return df_sorted
    
    def plot_hyperparameter_analysis(self):
        """Plot hyperparameter analysis."""
        if len(self.results) < 2:
            print("Need at least 2 experiments for analysis.")
            return
        
        df_data = []
        for result in self.results:
            row = {
                'learning_rate': result.config.learning_rate,
                'batch_size': result.config.batch_size,
                'temperature': result.config.temperature,
                'avg_retrieval_r1': result.metrics.get('avg_retrieval_r1', 0),
                'val_loss': result.metrics.get('val_loss', float('inf'))
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Learning rate vs performance
        axes[0, 0].scatter(df['learning_rate'], df['avg_retrieval_r1'])
        axes[0, 0].set_xlabel('Learning Rate')
        axes[0, 0].set_ylabel('Avg Retrieval R@1')
        axes[0, 0].set_title('Learning Rate vs Performance')
        axes[0, 0].set_xscale('log')
        
        # Batch size vs performance
        axes[0, 1].scatter(df['batch_size'], df['avg_retrieval_r1'])
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Avg Retrieval R@1')
        axes[0, 1].set_title('Batch Size vs Performance')
        
        # Temperature vs performance
        axes[1, 0].scatter(df['temperature'], df['avg_retrieval_r1'])
        axes[1, 0].set_xlabel('Temperature')
        axes[1, 0].set_ylabel('Avg Retrieval R@1')
        axes[1, 0].set_title('Temperature vs Performance')
        
        # Loss vs performance correlation
        valid_losses = df[df['val_loss'] != float('inf')]
        if len(valid_losses) > 1:
            axes[1, 1].scatter(valid_losses['val_loss'], valid_losses['avg_retrieval_r1'])
            axes[1, 1].set_xlabel('Validation Loss')
            axes[1, 1].set_ylabel('Avg Retrieval R@1')
            axes[1, 1].set_title('Loss vs Performance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiments_dir, 'hyperparameter_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Hyperparameter analysis saved to {self.experiments_dir}/hyperparameter_analysis.png")


class StatisticalSignificanceTests:
    """Perform statistical significance tests for model comparisons."""
    
    @staticmethod
    def bootstrap_confidence_interval(scores: List[float], confidence: float = 0.95, 
                                    n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        np.random.seed(42)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_scores.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
        upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
        
        return lower, upper
    
    @staticmethod
    def paired_t_test(scores_a: List[float], scores_b: List[float]) -> Dict[str, float]:
        """Perform paired t-test between two sets of scores."""
        from scipy import stats
        
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have the same length")
        
        differences = np.array(scores_a) - np.array(scores_b)
        t_stat, p_value = stats.ttest_1samp(differences, 0)
        
        return {
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def effect_size_cohens_d(scores_a: List[float], scores_b: List[float]) -> float:
        """Compute Cohen's d effect size."""
        mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
        std_a, std_b = np.std(scores_a, ddof=1), np.std(scores_b, ddof=1)
        
        # Pooled standard deviation
        n_a, n_b = len(scores_a), len(scores_b)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        
        cohens_d = (mean_a - mean_b) / pooled_std
        return cohens_d


class ExperimentDesignBestPractices:
    """Demonstrate best practices for experiment design."""
    
    @staticmethod
    def demonstrate_train_val_test_split():
        """Show proper data splitting strategies."""
        print("\n=== Data Splitting Best Practices ===")
        
        total_samples = 1000
        
        # Standard split
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size
        
        print(f"Dataset size: {total_samples}")
        print(f"Standard split (70/15/15):")
        print(f"  Training: {train_size} samples ({train_size/total_samples*100:.1f}%)")
        print(f"  Validation: {val_size} samples ({val_size/total_samples*100:.1f}%)")
        print(f"  Test: {test_size} samples ({test_size/total_samples*100:.1f}%)")
        
        print(f"\nKey principles:")
        print(f"1. Test set should NEVER be used during development")
        print(f"2. Validation set is for hyperparameter tuning and model selection")
        print(f"3. Final evaluation is done only once on the test set")
        print(f"4. For small datasets, consider k-fold cross-validation")
        
        # Show k-fold strategy
        print(f"\nK-fold Cross-Validation (k=5):")
        k = 5
        fold_size = total_samples // k
        for i in range(k):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size
            print(f"  Fold {i+1}: Train on samples [0:{val_start}, {val_end}:{total_samples}], "
                  f"Validate on [{val_start}:{val_end}]")
    
    @staticmethod
    def demonstrate_baseline_comparison():
        """Show how to establish proper baselines."""
        print("\n=== Baseline Establishment ===")
        
        print("Essential baselines for CLIP-like models:")
        print("1. Random baseline: Random embeddings")
        print("2. Simple baseline: Bag-of-words + basic CNN")
        print("3. Strong baseline: Pretrained CLIP without fine-tuning")
        print("4. Oracle baseline: Upper bound (if available)")
        
        # Simulate baseline results
        baselines = {
            'Random': 0.05,
            'Bag-of-words + CNN': 0.25,
            'Pretrained CLIP': 0.65,
            'Fine-tuned CLIP': 0.72,
            'Oracle (Human)': 0.95
        }
        
        print(f"\nExample Retrieval R@1 Results:")
        print(f"{'Method':<20} {'R@1':<10} {'Improvement':<12}")
        print("-" * 42)
        
        baseline_score = baselines['Pretrained CLIP']
        for method, score in baselines.items():
            improvement = score - baseline_score if method != 'Pretrained CLIP' else 0
            print(f"{method:<20} {score:<10.3f} {improvement:+.3f}")
    
    @staticmethod
    def demonstrate_ablation_studies():
        """Show how to design ablation studies."""
        print("\n=== Ablation Study Design ===")
        
        print("Systematic component removal to understand contribution:")
        
        # Simulate ablation results
        ablations = {
            'Full model': {'r1': 0.72, 'components': ['vision_encoder', 'text_encoder', 'contrastive_loss', 'data_augmentation']},
            '- Data augmentation': {'r1': 0.68, 'components': ['vision_encoder', 'text_encoder', 'contrastive_loss']},
            '- Contrastive loss (MSE instead)': {'r1': 0.45, 'components': ['vision_encoder', 'text_encoder', 'mse_loss', 'data_augmentation']},
            '- Text encoder (image only)': {'r1': 0.35, 'components': ['vision_encoder', 'contrastive_loss', 'data_augmentation']},
            '- Vision encoder (text only)': {'r1': 0.30, 'components': ['text_encoder', 'contrastive_loss', 'data_augmentation']},
        }
        
        print(f"{'Configuration':<30} {'R@1':<10} {'Drop':<10}")
        print("-" * 50)
        
        full_score = ablations['Full model']['r1']
        for config, data in ablations.items():
            score = data['r1']
            drop = full_score - score
            print(f"{config:<30} {score:<10.3f} {drop:<10.3f}")
        
        print(f"\nKey insights:")
        print(f"- Contrastive loss is most important component (-0.27)")
        print(f"- Vision and text encoders are both crucial")
        print(f"- Data augmentation provides modest improvement (+0.04)")


def create_mock_experiment_results():
    """Create mock experiment results for demonstration."""
    experiments = []
    
    # Different learning rates
    for lr in [1e-5, 5e-5, 1e-4, 5e-4]:
        config = ExperimentConfig(
            experiment_name=f"lr_{lr}",
            model_name="openai/clip-vit-base-patch32",
            dataset_name="sample_dataset",
            batch_size=16,
            learning_rate=lr,
            epochs=5,
            temperature=0.07
        )
        
        # Simulate realistic results (higher LR = worse performance after some point)
        if lr <= 5e-5:
            base_performance = 0.65 + np.random.normal(0, 0.02)
        else:
            base_performance = 0.55 + np.random.normal(0, 0.03)
        
        metrics = {
            'avg_retrieval_r1': max(0, min(1, base_performance)),
            'img_to_text_r1': max(0, min(1, base_performance + np.random.normal(0, 0.01))),
            'text_to_img_r1': max(0, min(1, base_performance + np.random.normal(0, 0.01))),
            'val_loss': max(0.1, 2.0 - base_performance + np.random.normal(0, 0.1))
        }
        
        result = ExperimentResults(
            config=config,
            metrics=metrics,
            training_history={'train_loss': [2.5, 2.0, 1.8, 1.6, 1.5], 'val_loss': [2.3, 1.9, 1.7, 1.6, 1.5]},
            timestamp=datetime.now().isoformat(),
            duration_seconds=3600 + np.random.randint(-600, 600)
        )
        experiments.append(result)
    
    # Different batch sizes
    for bs in [8, 16, 32]:
        config = ExperimentConfig(
            experiment_name=f"bs_{bs}",
            model_name="openai/clip-vit-base-patch32",
            dataset_name="sample_dataset",
            batch_size=bs,
            learning_rate=5e-5,
            epochs=5,
            temperature=0.07
        )
        
        # Larger batch sizes generally perform better (up to a point)
        base_performance = 0.60 + (bs - 8) * 0.02 + np.random.normal(0, 0.02)
        
        metrics = {
            'avg_retrieval_r1': max(0, min(1, base_performance)),
            'img_to_text_r1': max(0, min(1, base_performance + np.random.normal(0, 0.01))),
            'text_to_img_r1': max(0, min(1, base_performance + np.random.normal(0, 0.01))),
            'val_loss': max(0.1, 2.0 - base_performance + np.random.normal(0, 0.1))
        }
        
        result = ExperimentResults(
            config=config,
            metrics=metrics,
            training_history={'train_loss': [2.5, 2.0, 1.8, 1.6, 1.5], 'val_loss': [2.3, 1.9, 1.7, 1.6, 1.5]},
            timestamp=datetime.now().isoformat(),
            duration_seconds=3600 + np.random.randint(-600, 600)
        )
        experiments.append(result)
    
    return experiments


def demo_experiment_design():
    """Demonstrate experiment design and evaluation framework."""
    print("Experiment Design and Evaluation Framework - Step 4")
    print("=" * 60)
    
    # 1. Demonstrate contrastive learning theory
    theory = ContrastiveLearningTheory()
    theory.demonstrate_contrastive_loss()
    theory.visualize_embedding_space()
    
    # 2. Show best practices
    practices = ExperimentDesignBestPractices()
    practices.demonstrate_train_val_test_split()
    practices.demonstrate_baseline_comparison()
    practices.demonstrate_ablation_studies()
    
    # 3. Demonstrate experiment tracking
    print("\n" + "=" * 60)
    print("EXPERIMENT TRACKING DEMO")
    
    tracker = ExperimentTracker()
    
    # Add mock experiments
    mock_experiments = create_mock_experiment_results()
    for exp in mock_experiments:
        tracker.add_experiment(exp)
    
    # Compare experiments
    comparison_df = tracker.compare_experiments()
    
    # Plot analysis
    tracker.plot_hyperparameter_analysis()
    
    # 4. Statistical significance testing
    print("\n=== Statistical Significance Testing ===")
    
    # Simulate two model performances
    np.random.seed(42)
    model_a_scores = [0.65 + np.random.normal(0, 0.05) for _ in range(10)]
    model_b_scores = [0.70 + np.random.normal(0, 0.05) for _ in range(10)]
    
    # Bootstrap confidence intervals
    stats_tests = StatisticalSignificanceTests()
    ci_a = stats_tests.bootstrap_confidence_interval(model_a_scores)
    ci_b = stats_tests.bootstrap_confidence_interval(model_b_scores)
    
    print(f"Model A: {np.mean(model_a_scores):.3f} ± {np.std(model_a_scores):.3f}")
    print(f"  95% CI: [{ci_a[0]:.3f}, {ci_a[1]:.3f}]")
    print(f"Model B: {np.mean(model_b_scores):.3f} ± {np.std(model_b_scores):.3f}")
    print(f"  95% CI: [{ci_b[0]:.3f}, {ci_b[1]:.3f}]")
    
    # Paired t-test
    t_test_results = stats_tests.paired_t_test(model_b_scores, model_a_scores)
    effect_size = stats_tests.effect_size_cohens_d(model_b_scores, model_a_scores)
    
    print(f"\nPaired t-test results:")
    print(f"  Mean difference: {t_test_results['mean_difference']:.4f}")
    print(f"  t-statistic: {t_test_results['t_statistic']:.4f}")
    print(f"  p-value: {t_test_results['p_value']:.4f}")
    print(f"  Significant: {t_test_results['significant']}")
    print(f"  Effect size (Cohen's d): {effect_size:.4f}")
    
    # Interpret effect size
    if abs(effect_size) < 0.2:
        effect_interpretation = "negligible"
    elif abs(effect_size) < 0.5:
        effect_interpretation = "small"
    elif abs(effect_size) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    print(f"  Effect size interpretation: {effect_interpretation}")
    
    print("\n" + "=" * 60)
    print("Experiment design demo completed!")
    print("Files created:")
    print("- embedding_space_evolution.png: Visualization of embedding space evolution")
    print("- experiments/experiment_results.json: Experiment tracking database")
    print("- experiments/hyperparameter_analysis.png: Hyperparameter analysis plots")
    
    print(f"\nKey takeaways:")
    print(f"1. Always establish proper baselines and use statistical significance testing")
    print(f"2. Design systematic ablation studies to understand component contributions")
    print(f"3. Use proper train/validation/test splits and avoid data leakage")
    print(f"4. Track experiments systematically and compare hyperparameters")
    print(f"5. Report confidence intervals and effect sizes, not just point estimates")


if __name__ == "__main__":
    demo_experiment_design()