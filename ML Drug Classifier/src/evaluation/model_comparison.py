"""
Model comparison utilities for drug classification.

This module provides tools to compare multiple machine learning models systematically.

WHY COMPARE MODELS?
Different algorithms have different strengths and weaknesses:
- Some work better with small datasets
- Some handle non-linear patterns better
- Some are more interpretable
- Some are faster to train/predict

COMPARISON STRATEGIES:
1. Performance metrics (accuracy, precision, recall, F1)
2. Training time and prediction speed
3. Model complexity and interpretability
4. Robustness across different data splits
5. Statistical significance of differences
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.helpers import setup_plotting_style, ensure_dir, save_json
from src.evaluation.metrics import ModelEvaluator

class ModelComparison:
    """
    Comprehensive model comparison toolkit.
    
    This class provides tools to systematically compare multiple ML models.
    """
    
    def __init__(self):
        """Initialize the model comparison toolkit."""
        setup_plotting_style()
        self.models = {}
        self.results = {}
        self.comparison_data = None
        self.evaluator = ModelEvaluator()
    
    def add_model(self, model, model_name=None):
        """
        Add a model to the comparison.
        
        Args:
            model: Trained model object
            model_name (str): Name for the model (optional)
        """
        if model_name is None:
            model_name = getattr(model, 'model_name', f'Model_{len(self.models)}')
        
        self.models[model_name] = model
        print(f"✅ Added {model_name} to comparison")
    
    def compare_models(self, X_test, y_test, class_names=None, save_results=True):
        """
        Compare all added models on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            class_names (list): Names of the classes
            save_results (bool): Whether to save results
            
        Returns:
            dict: Comprehensive comparison results
        """
        if len(self.models) < 2:
            print("⚠️  Need at least 2 models for comparison")
            return None
        
        print(f"\n{'='*80}")
        print(f"🏆 COMPREHENSIVE MODEL COMPARISON")
        print(f"{'='*80}")
        print(f"Comparing {len(self.models)} models on {len(X_test)} test samples")
        
        # Evaluate each model
        model_results = []
        
        for model_name, model in self.models.items():
            print(f"\n🔄 Evaluating {model_name}...")
            
            # Time the prediction
            start_time = time.time()
            results = self.evaluator.evaluate_model(model, X_test, y_test, model_name, class_names)
            prediction_time = time.time() - start_time
            
            results['prediction_time'] = prediction_time
            results['test_size'] = len(X_test)
            
            model_results.append(results)
            self.results[model_name] = results
        
        # Create comparison summary
        comparison_summary = self._create_comparison_summary(model_results)
        
        # Perform statistical significance tests
        significance_results = self._perform_significance_tests(model_results)
        
        # Create comprehensive comparison results
        comparison_results = {
            'individual_results': model_results,
            'comparison_summary': comparison_summary,
            'significance_tests': significance_results,
            'best_models': self._identify_best_models(model_results),
            'ranking': self._rank_models(model_results)
        }
        
        # Save results if requested
        if save_results:
            self._save_comparison_results(comparison_results)
        
        # Print summary
        self._print_comparison_summary(comparison_results)
        
        return comparison_results
    
    def cross_validate_models(self, X, y, cv_folds=5, scoring='accuracy'):
        """
        Perform cross-validation comparison of all models.
        
        Args:
            X: Features
            y: Target
            cv_folds (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            dict: Cross-validation comparison results
        """
        if len(self.models) < 2:
            print("⚠️  Need at least 2 models for comparison")
            return None
        
        print(f"\n🔄 CROSS-VALIDATION COMPARISON")
        print(f"{'='*50}")
        print(f"Performing {cv_folds}-fold cross-validation with {scoring} scoring")
        
        cv_results = {}
        
        # Set up cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for model_name, model in self.models.items():
            print(f"\n   🔄 Cross-validating {model_name}...")
            
            # Time the cross-validation
            start_time = time.time()
            
            # Get the underlying sklearn model
            sklearn_model = getattr(model, 'model', model)
            
            # Perform cross-validation
            cv_scores = cross_val_score(sklearn_model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
            cv_time = time.time() - start_time
            
            cv_results[model_name] = {
                'cv_scores': cv_scores,
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'min_score': np.min(cv_scores),
                'max_score': np.max(cv_scores),
                'cv_time': cv_time,
                'scoring_metric': scoring
            }
            
            print(f"     Mean {scoring}: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
            print(f"     Range: [{np.min(cv_scores):.4f}, {np.max(cv_scores):.4f}]")
            print(f"     CV Time: {cv_time:.2f} seconds")
        
        # Statistical comparison
        self._compare_cv_results(cv_results)
        
        return cv_results
    
    def _create_comparison_summary(self, model_results):
        """Create a summary comparison table."""
        summary_data = []
        
        for results in model_results:
            row = {
                'Model': results['model_name'],
                'Accuracy': results['accuracy'],
                'Precision': results['precision_macro'],
                'Recall': results['recall_macro'],
                'F1-Score': results['f1_macro'],
                'Prediction Time (s)': results['prediction_time']
            }
            
            # Add ROC AUC if available
            if 'roc_auc' in results:
                row['ROC AUC'] = results['roc_auc']
            elif 'roc_auc_macro' in results:
                row['ROC AUC'] = results['roc_auc_macro']
            
            # Add log loss if available
            if 'log_loss' in results and results['log_loss'] is not None:
                row['Log Loss'] = results['log_loss']
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def _perform_significance_tests(self, model_results):
        """Perform pairwise statistical significance tests."""
        significance_results = {}
        
        print(f"\n📊 STATISTICAL SIGNIFICANCE TESTS")
        print("="*60)
        
        # Compare each pair of models
        for i, results1 in enumerate(model_results):
            for j, results2 in enumerate(model_results[i+1:], i+1):
                model1_name = results1['model_name']
                model2_name = results2['model_name']
                
                comparison_key = f"{model1_name}_vs_{model2_name}"
                
                # Perform statistical test
                sig_result = self.evaluator.calculate_statistical_significance(
                    results1, results2, alpha=0.05
                )
                
                significance_results[comparison_key] = sig_result
        
        return significance_results
    
    def _identify_best_models(self, model_results):
        """Identify best performing models for each metric."""
        best_models = {}
        
        # Metrics to find best for (higher is better)
        metrics_higher_better = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Metrics to find best for (lower is better) 
        metrics_lower_better = ['prediction_time']
        
        if any('log_loss' in results and results['log_loss'] is not None for results in model_results):
            metrics_lower_better.append('log_loss')
        
        if any('roc_auc' in results or 'roc_auc_macro' in results for results in model_results):
            metrics_higher_better.append('roc_auc')
        
        # Find best for each metric
        for metric in metrics_higher_better:
            best_score = -1
            best_model = None
            
            for results in model_results:
                score = results.get(metric, results.get(f'{metric}_macro', -1))
                if score > best_score:
                    best_score = score
                    best_model = results['model_name']
            
            if best_model:
                best_models[metric] = {'model': best_model, 'score': best_score}
        
        for metric in metrics_lower_better:
            best_score = float('inf')
            best_model = None
            
            for results in model_results:
                score = results.get(metric, float('inf'))
                if score < best_score and score is not None:
                    best_score = score
                    best_model = results['model_name']
            
            if best_model:
                best_models[metric] = {'model': best_model, 'score': best_score}
        
        return best_models
    
    def _rank_models(self, model_results):
        """Rank models by overall performance."""
        # Create ranking based on multiple metrics
        ranking_scores = {}
        
        for results in model_results:
            model_name = results['model_name']
            
            # Weighted score: accuracy (40%) + f1_macro (40%) + speed (20%)
            accuracy_score = results['accuracy']
            f1_score = results['f1_macro'] 
            
            # Normalize speed (lower is better, so invert)
            max_time = max(r['prediction_time'] for r in model_results)
            speed_score = 1 - (results['prediction_time'] / max_time)
            
            # Calculate weighted score
            overall_score = (0.4 * accuracy_score + 
                           0.4 * f1_score + 
                           0.2 * speed_score)
            
            ranking_scores[model_name] = {
                'overall_score': overall_score,
                'accuracy': accuracy_score,
                'f1_score': f1_score,
                'speed_score': speed_score
            }
        
        # Sort by overall score
        ranked_models = sorted(ranking_scores.items(), 
                             key=lambda x: x[1]['overall_score'], 
                             reverse=True)
        
        return ranked_models
    
    def _compare_cv_results(self, cv_results):
        """Compare cross-validation results statistically."""
        from scipy.stats import ttest_rel
        
        print(f"\n📊 CROSS-VALIDATION STATISTICAL COMPARISON")
        print("="*60)
        
        model_names = list(cv_results.keys())
        
        # Pairwise t-tests
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                scores1 = cv_results[model1]['cv_scores']
                scores2 = cv_results[model2]['cv_scores']
                
                # Paired t-test
                try:
                    t_stat, p_value = ttest_rel(scores1, scores2)
                    
                    print(f"\n{model1} vs {model2}:")
                    print(f"   Mean difference: {np.mean(scores1) - np.mean(scores2):.4f}")
                    print(f"   t-statistic: {t_stat:.4f}")
                    print(f"   p-value: {p_value:.4f}")
                    
                    if p_value < 0.05:
                        better_model = model1 if np.mean(scores1) > np.mean(scores2) else model2
                        print(f"   ✅ {better_model} is significantly better (p < 0.05)")
                    else:
                        print(f"   ❌ No significant difference (p >= 0.05)")
                        
                except Exception as e:
                    print(f"   ⚠️  Could not perform t-test: {str(e)}")
    
    def _save_comparison_results(self, comparison_results):
        """Save comparison results to file."""
        save_dir = "results/metrics/"
        ensure_dir(save_dir)
        
        # Save summary table
        summary_path = os.path.join(save_dir, "model_comparison_summary.csv")
        comparison_results['comparison_summary'].to_csv(summary_path, index=False)
        
        # Save detailed results
        results_path = os.path.join(save_dir, "detailed_comparison_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in comparison_results.items():
            if key == 'comparison_summary':
                serializable_results[key] = value.to_dict()
            elif key == 'individual_results':
                serializable_results[key] = []
                for result in value:
                    serialized_result = {}
                    for k, v in result.items():
                        if isinstance(v, np.ndarray):
                            serialized_result[k] = v.tolist()
                        else:
                            serialized_result[k] = v
                    serializable_results[key].append(serialized_result)
            else:
                serializable_results[key] = value
        
        save_json(serializable_results, results_path)
        
        print(f"\n💾 Comparison results saved:")
        print(f"   Summary: {summary_path}")
        print(f"   Detailed: {results_path}")
    
    def _print_comparison_summary(self, comparison_results):
        """Print a formatted comparison summary."""
        print(f"\n{'='*80}")
        print(f"🏆 MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Display comparison table
        summary_df = comparison_results['comparison_summary']
        print("\n📊 PERFORMANCE COMPARISON:")
        print(summary_df.round(4).to_string(index=False))
        
        # Display best models
        print(f"\n🥇 BEST PERFORMERS BY METRIC:")
        best_models = comparison_results['best_models']
        for metric, info in best_models.items():
            print(f"   {metric.replace('_', ' ').title():<20}: {info['model']} ({info['score']:.4f})")
        
        # Display overall ranking
        print(f"\n🏆 OVERALL RANKING:")
        ranking = comparison_results['ranking']
        for i, (model_name, scores) in enumerate(ranking, 1):
            print(f"   {i}. {model_name:<25} (Score: {scores['overall_score']:.4f})")
        
        # Display significance test summary
        sig_tests = comparison_results['significance_tests']
        significant_differences = sum(1 for test in sig_tests.values() if test['is_significant'])
        total_comparisons = len(sig_tests)
        
        print(f"\n📊 STATISTICAL SIGNIFICANCE:")
        print(f"   Significant differences: {significant_differences}/{total_comparisons} comparisons")
        
        if significant_differences > 0:
            print(f"   Significant comparisons:")
            for comparison_name, test_result in sig_tests.items():
                if test_result['is_significant']:
                    model1, model2 = comparison_name.split('_vs_')
                    better_model = model1 if test_result['model1_accuracy'] > test_result['model2_accuracy'] else model2
                    print(f"     {better_model} > {model1 if better_model == model2 else model2} (p = {test_result['mcnemar_p_value']:.4f})")
    
    def plot_model_comparison(self, comparison_results, save_path=None):
        """
        Plot comprehensive model comparison visualization.
        
        Args:
            comparison_results (dict): Results from compare_models()
            save_path (str): Path to save the plot
        """
        summary_df = comparison_results['comparison_summary']
        
        # Create subplot layout
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Performance metrics bar plot
        ax1 = plt.subplot(2, 3, 1)
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        available_metrics = [m for m in metrics_to_plot if m in summary_df.columns]
        
        x = np.arange(len(summary_df))
        width = 0.8 / len(available_metrics)
        
        for i, metric in enumerate(available_metrics):
            offset = (i - len(available_metrics)/2 + 0.5) * width
            ax1.bar(x + offset, summary_df[metric], width, label=metric, alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Speed comparison
        ax2 = plt.subplot(2, 3, 2)
        bars = ax2.bar(summary_df['Model'], summary_df['Prediction Time (s)'], 
                       color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Prediction Time (seconds)')
        ax2.set_title('Prediction Speed Comparison', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s', ha='center', va='bottom')
        
        # 3. Overall ranking radar chart
        ax3 = plt.subplot(2, 3, 3)
        ranking = comparison_results['ranking']
        models = [item[0] for item in ranking]
        scores = [item[1]['overall_score'] for item in ranking]
        
        bars = ax3.barh(models, scores, color='lightgreen', alpha=0.7)
        ax3.set_xlabel('Overall Score')
        ax3.set_title('Overall Model Ranking', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax3.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center')
        
        # 4. ROC AUC comparison (if available)
        if 'ROC AUC' in summary_df.columns:
            ax4 = plt.subplot(2, 3, 4)
            bars = ax4.bar(summary_df['Model'], summary_df['ROC AUC'], 
                          color='lightblue', alpha=0.7)
            ax4.set_xlabel('Models')
            ax4.set_ylabel('ROC AUC')
            ax4.set_title('ROC AUC Comparison', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # 5. Performance vs Speed scatter plot
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(summary_df['Prediction Time (s)'], summary_df['F1-Score'], 
                   s=100, alpha=0.7, color='purple')
        
        # Add model name labels
        for i, model in enumerate(summary_df['Model']):
            ax5.annotate(model, 
                        (summary_df['Prediction Time (s)'].iloc[i], 
                         summary_df['F1-Score'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax5.set_xlabel('Prediction Time (seconds)')
        ax5.set_ylabel('F1-Score')
        ax5.set_title('Performance vs Speed Trade-off', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistical significance heatmap
        ax6 = plt.subplot(2, 3, 6)
        sig_tests = comparison_results['significance_tests']
        
        # Create significance matrix
        model_names = summary_df['Model'].tolist()
        n_models = len(model_names)
        sig_matrix = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    key1 = f"{model1}_vs_{model2}"
                    key2 = f"{model2}_vs_{model1}"
                    
                    if key1 in sig_tests:
                        sig_matrix[i, j] = 1 if sig_tests[key1]['is_significant'] else 0
                    elif key2 in sig_tests:
                        sig_matrix[i, j] = 1 if sig_tests[key2]['is_significant'] else 0
        
        sns.heatmap(sig_matrix, annot=True, cmap='RdYlGn', 
                   xticklabels=model_names, yticklabels=model_names,
                   cbar_kws={'label': 'Statistically Significant'}, ax=ax6)
        ax6.set_title('Statistical Significance Matrix', fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Model comparison plot saved to: {save_path}")
        
        plt.show()
    
    def plot_cross_validation_comparison(self, cv_results, save_path=None):
        """
        Plot cross-validation results comparison.
        
        Args:
            cv_results (dict): Results from cross_validate_models()
            save_path (str): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Box plot of CV scores
        model_names = list(cv_results.keys())
        cv_scores_list = [cv_results[model]['cv_scores'] for model in model_names]
        
        box_plot = ax1.boxplot(cv_scores_list, labels=model_names, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Cross-Validation Score')
        ax1.set_title('Cross-Validation Score Distribution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Mean scores with error bars
        means = [cv_results[model]['mean_score'] for model in model_names]
        stds = [cv_results[model]['std_score'] for model in model_names]
        
        bars = ax2.bar(model_names, means, yerr=stds, capsize=5, 
                      color=colors, alpha=0.7, error_kw={'linewidth': 2})
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Mean CV Score')
        ax2.set_title('Mean Cross-Validation Scores', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Cross-validation comparison plot saved to: {save_path}")
        
        plt.show()
    
    def generate_model_recommendation(self, comparison_results):
        """
        Generate intelligent model recommendation based on comparison results.
        
        Args:
            comparison_results (dict): Results from compare_models()
            
        Returns:
            dict: Model recommendation with reasoning
        """
        ranking = comparison_results['ranking']
        best_models = comparison_results['best_models']
        summary_df = comparison_results['comparison_summary']
        
        # Get top performer
        top_model = ranking[0][0]
        top_score = ranking[0][1]['overall_score']
        
        # Analyze trade-offs
        fastest_model = best_models.get('prediction_time', {}).get('model', 'Unknown')
        most_accurate = best_models.get('accuracy', {}).get('model', 'Unknown')
        best_f1 = best_models.get('f1_macro', {}).get('model', 'Unknown')
        
        # Create recommendation
        recommendation = {
            'primary_recommendation': top_model,
            'overall_score': top_score,
            'reasoning': [],
            'use_cases': {},
            'considerations': []
        }
        
        # Add reasoning
        if top_model == most_accurate and top_model == best_f1:
            recommendation['reasoning'].append(f"{top_model} excels in both accuracy and F1-score")
        
        if top_model == fastest_model:
            recommendation['reasoning'].append(f"{top_model} provides the fastest predictions")
        
        # Different use case recommendations
        recommendation['use_cases'] = {
            'production_deployment': {
                'model': fastest_model,
                'reason': 'Fastest prediction time for real-time applications'
            },
            'highest_accuracy': {
                'model': most_accurate,
                'reason': 'Best overall accuracy for critical decisions'
            },
            'balanced_performance': {
                'model': best_f1,
                'reason': 'Best F1-score for balanced precision and recall'
            },
            'overall_best': {
                'model': top_model,
                'reason': 'Best combination of accuracy, F1-score, and speed'
            }
        }
        
        # Add considerations
        if len(set([top_model, fastest_model, most_accurate, best_f1])) > 1:
            recommendation['considerations'].append(
                "Different models excel in different aspects - consider your priorities"
            )
        
        # Check if differences are significant
        sig_tests = comparison_results['significance_tests']
        significant_differences = sum(1 for test in sig_tests.values() if test['is_significant'])
        
        if significant_differences == 0:
            recommendation['considerations'].append(
                "No statistically significant differences found - any model may be suitable"
            )
        
        # Performance threshold analysis
        high_performers = summary_df[summary_df['Accuracy'] > 0.9]['Model'].tolist()
        if high_performers:
            recommendation['considerations'].append(
                f"Models with >90% accuracy: {', '.join(high_performers)}"
            )
        
        return recommendation
    
    def print_recommendation(self, recommendation):
        """Print formatted model recommendation."""
        print(f"\n{'='*80}")
        print(f"🤖 INTELLIGENT MODEL RECOMMENDATION")
        print(f"{'='*80}")
        
        print(f"\n🏆 PRIMARY RECOMMENDATION: {recommendation['primary_recommendation']}")
        print(f"   Overall Score: {recommendation['overall_score']:.4f}")
        
        if recommendation['reasoning']:
            print(f"\n💡 REASONING:")
            for reason in recommendation['reasoning']:
                print(f"   • {reason}")
        
        print(f"\n🎯 USE CASE SPECIFIC RECOMMENDATIONS:")
        for use_case, info in recommendation['use_cases'].items():
            print(f"   {use_case.replace('_', ' ').title()}:")
            print(f"     Model: {info['model']}")
            print(f"     Reason: {info['reason']}")
        
        if recommendation['considerations']:
            print(f"\n⚠️  IMPORTANT CONSIDERATIONS:")
            for consideration in recommendation['considerations']:
                print(f"   • {consideration}")
        
        print(f"\n📋 SUMMARY:")
        print(f"   For most applications, use: {recommendation['primary_recommendation']}")
        print(f"   For speed-critical applications, consider: {recommendation['use_cases']['production_deployment']['model']}")
        print(f"   For maximum accuracy, consider: {recommendation['use_cases']['highest_accuracy']['model']}")
    
    def export_comparison_report(self, comparison_results, save_path="results/reports/model_comparison_report.md"):
        """
        Export a comprehensive comparison report in Markdown format.
        
        Args:
            comparison_results (dict): Results from compare_models()
            save_path (str): Path to save the report
        """
        ensure_dir(os.path.dirname(save_path))
        
        # Generate recommendation
        recommendation = self.generate_model_recommendation(comparison_results)
        
        # Create markdown report
        report_lines = [
            "# Model Comparison Report",
            "",
            "## Executive Summary",
            "",
            f"**Recommended Model:** {recommendation['primary_recommendation']}",
            f"**Overall Score:** {recommendation['overall_score']:.4f}",
            "",
            "## Performance Comparison",
            "",
        ]
        
        # Add comparison table
        summary_df = comparison_results['comparison_summary']
        report_lines.append(summary_df.round(4).to_markdown(index=False))
        report_lines.append("")
        
        # Add ranking
        report_lines.extend([
            "## Overall Ranking",
            "",
        ])
        
        ranking = comparison_results['ranking']
        for i, (model_name, scores) in enumerate(ranking, 1):
            report_lines.append(f"{i}. **{model_name}** (Score: {scores['overall_score']:.4f})")
        
        report_lines.append("")
        
        # Add best performers
        report_lines.extend([
            "## Best Performers by Metric",
            "",
        ])
        
        best_models = comparison_results['best_models']
        for metric, info in best_models.items():
            metric_name = metric.replace('_', ' ').title()
            report_lines.append(f"- **{metric_name}:** {info['model']} ({info['score']:.4f})")
        
        report_lines.append("")
        
        # Add recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            f"### Primary Recommendation: {recommendation['primary_recommendation']}",
            "",
        ])
        
        for reason in recommendation['reasoning']:
            report_lines.append(f"- {reason}")
        
        report_lines.extend([
            "",
            "### Use Case Specific Recommendations",
            "",
        ])
        
        for use_case, info in recommendation['use_cases'].items():
            use_case_title = use_case.replace('_', ' ').title()
            report_lines.extend([
                f"**{use_case_title}:**",
                f"- Model: {info['model']}",
                f"- Reason: {info['reason']}",
                "",
            ])
        
        # Add considerations
        if recommendation['considerations']:
            report_lines.extend([
                "### Important Considerations",
                "",
            ])
            for consideration in recommendation['considerations']:
                report_lines.append(f"- {consideration}")
            report_lines.append("")
        
        # Statistical significance section
        report_lines.extend([
            "## Statistical Significance",
            "",
        ])
        
        sig_tests = comparison_results['significance_tests']
        significant_differences = [test for test in sig_tests.values() if test['is_significant']]
        
        if significant_differences:
            report_lines.append("### Significant Differences Found:")
            for test in significant_differences:
                better_model = test['model1_name'] if test['model1_accuracy'] > test['model2_accuracy'] else test['model2_name']
                worse_model = test['model2_name'] if better_model == test['model1_name'] else test['model1_name']
                report_lines.append(f"- {better_model} significantly outperforms {worse_model} (p = {test['mcnemar_p_value']:.4f})")
        else:
            report_lines.append("No statistically significant differences found between models.")
        
        # Write report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Comprehensive comparison report saved to: {save_path}")
        
        return save_path