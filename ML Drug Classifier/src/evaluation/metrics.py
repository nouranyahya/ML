"""
Model evaluation metrics for drug classification.

This module provides comprehensive evaluation tools to measure how well our models perform.

WHAT ARE EVALUATION METRICS?
Think of metrics like grading a test - they tell us how well our model is doing.
Different metrics focus on different aspects of performance.

KEY METRICS EXPLAINED:

1. ACCURACY: What percentage of predictions were correct?
   - Example: 85% accuracy means 85 out of 100 predictions were right

2. PRECISION: Of all positive predictions, how many were actually correct?
   - Example: If model predicts 100 patients need DrugA, but only 80 actually need it, precision = 80%

3. RECALL (SENSITIVITY): Of all actual positive cases, how many did we catch?
   - Example: If 100 patients need DrugA, but we only identified 70, recall = 70%

4. F1-SCORE: Harmonic mean of precision and recall (balances both)
   - Good when you want both high precision AND high recall

5. CONFUSION MATRIX: Shows where the model gets confused
   - Rows = actual class, Columns = predicted class
   - Diagonal = correct predictions, off-diagonal = mistakes

6. ROC CURVE: Shows trade-off between true positive rate and false positive rate
   - Area Under Curve (AUC): Higher is better (1.0 = perfect, 0.5 = random)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, log_loss, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.helpers import setup_plotting_style, ensure_dir

class ModelEvaluator:
    """
    Comprehensive model evaluation toolkit.
    
    This class provides all the tools needed to thoroughly evaluate model performance.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        setup_plotting_style()
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name=None, class_names=None):
        """
        Perform comprehensive evaluation of a single model.
        
        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test target (true labels)
            model_name (str): Name of the model for reporting
            class_names (list): Names of the classes
            
        Returns:
            dict: Comprehensive evaluation results
        """
        if model_name is None:
            model_name = getattr(model, 'model_name', 'Unknown Model')
        
        print(f"\n{'='*60}")
        print(f"📊 EVALUATING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except:
                print("⚠️  Could not get prediction probabilities")
        
        # Calculate basic metrics
        results = self._calculate_basic_metrics(y_test, y_pred, y_proba, model_name)
        
        # Calculate per-class metrics
        results.update(self._calculate_per_class_metrics(y_test, y_pred, class_names))
        
        # Calculate advanced metrics
        if y_proba is not None:
            results.update(self._calculate_advanced_metrics(y_test, y_proba, class_names))
        
        # Store results
        self.results[model_name] = results
        
        # Print summary
        self._print_evaluation_summary(results)
        
        return results
    
    def _calculate_basic_metrics(self, y_true, y_pred, y_proba, model_name):
        """Calculate basic classification metrics."""
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        # Calculate log loss if probabilities available
        if y_proba is not None:
            try:
                results['log_loss'] = log_loss(y_true, y_proba)
            except:
                results['log_loss'] = None
        
        return results
    
    def _calculate_per_class_metrics(self, y_true, y_pred, class_names):
        """Calculate per-class precision, recall, and F1-score."""
        unique_classes = np.unique(y_true)
        
        if class_names is None:
            class_names = [f"Class_{cls}" for cls in unique_classes]
        
        per_class_metrics = {}
        
        for i, class_label in enumerate(unique_classes):
            class_name = class_names[i] if i < len(class_names) else f"Class_{class_label}"
            
            # Calculate binary metrics for this class vs all others
            y_true_binary = (y_true == class_label).astype(int)
            y_pred_binary = (y_pred == class_label).astype(int)
            
            per_class_metrics[class_name] = {
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'support': np.sum(y_true_binary)
            }
        
        return {'per_class_metrics': per_class_metrics}
    
    def _calculate_advanced_metrics(self, y_true, y_proba, class_names):
        """Calculate advanced metrics using probabilities."""
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        
        results = {}
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            pr_auc = auc(recall, precision)
            
            results.update({
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'fpr': fpr,
                'tpr': tpr,
                'precision_curve': precision,
                'recall_curve': recall
            })
            
        else:
            # Multi-class classification
            try:
                # Calculate macro-averaged ROC AUC
                roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                results['roc_auc_macro'] = roc_auc
                
                # Calculate per-class ROC curves
                y_true_binarized = label_binarize(y_true, classes=unique_classes)
                
                per_class_roc = {}
                for i, class_label in enumerate(unique_classes):
                    class_name = class_names[i] if class_names and i < len(class_names) else f"Class_{class_label}"
                    
                    fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_proba[:, i])
                    roc_auc_class = auc(fpr, tpr)
                    
                    per_class_roc[class_name] = {
                        'fpr': fpr,
                        'tpr': tpr,
                        'auc': roc_auc_class
                    }
                
                results['per_class_roc'] = per_class_roc
                
            except Exception as e:
                print(f"⚠️  Could not calculate ROC AUC: {str(e)}")
        
        return results
    
    def _print_evaluation_summary(self, results):
        """Print a formatted summary of evaluation results."""
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   Precision (Macro): {results['precision_macro']:.4f}")
        print(f"   Recall (Macro): {results['recall_macro']:.4f}")
        print(f"   F1-Score (Macro): {results['f1_macro']:.4f}")
        
        if 'log_loss' in results and results['log_loss'] is not None:
            print(f"   Log Loss: {results['log_loss']:.4f}")
        
        if 'roc_auc' in results:
            print(f"   ROC AUC: {results['roc_auc']:.4f}")
        elif 'roc_auc_macro' in results:
            print(f"   ROC AUC (Macro): {results['roc_auc_macro']:.4f}")
        
        print(f"\n📋 PER-CLASS PERFORMANCE:")
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"   {class_name}:")
            print(f"     Precision: {metrics['precision']:.3f}")
            print(f"     Recall: {metrics['recall']:.3f}")
            print(f"     F1-Score: {metrics['f1_score']:.3f}")
            print(f"     Support: {metrics['support']} samples")
    
    def plot_confusion_matrix(self, results, class_names=None, normalize=False, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            results (dict): Evaluation results containing confusion matrix
            class_names (list): Names of the classes
            normalize (bool): Whether to normalize the matrix
            save_path (str): Path to save the plot
        """
        cm = results['confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        plt.title(f"{title}\n{results['model_name']}", fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, results, save_path=None):
        """
        Plot ROC curves.
        
        Args:
            results (dict): Evaluation results containing ROC data
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        if 'fpr' in results and 'tpr' in results:
            # Binary classification
            plt.plot(results['fpr'], results['tpr'], 
                    label=f"ROC Curve (AUC = {results['roc_auc']:.3f})",
                    linewidth=2)
        
        elif 'per_class_roc' in results:
            # Multi-class classification
            for class_name, roc_data in results['per_class_roc'].items():
                plt.plot(roc_data['fpr'], roc_data['tpr'],
                        label=f"{class_name} (AUC = {roc_data['auc']:.3f})",
                        linewidth=2)
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f"ROC Curves\n{results['model_name']}", fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 ROC curves saved to: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, results, save_path=None):
        """
        Plot precision-recall curve (for binary classification).
        
        Args:
            results (dict): Evaluation results containing PR curve data
            save_path (str): Path to save the plot
        """
        if 'precision_curve' not in results or 'recall_curve' not in results:
            print("⚠️  Precision-recall curve data not available (binary classification only)")
            return
        
        plt.figure(figsize=(8, 6))
        
        plt.plot(results['recall_curve'], results['precision_curve'],
                linewidth=2, label=f"PR Curve (AUC = {results['pr_auc']:.3f})")
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f"Precision-Recall Curve\n{results['model_name']}", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Precision-recall curve saved to: {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, results, save_path=None):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results (dict): Evaluation results
            save_path (str): Path to save the report
            
        Returns:
            str: Formatted evaluation report
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(f"COMPREHENSIVE EVALUATION REPORT")
        report_lines.append(f"Model: {results['model_name']}")
        report_lines.append("="*80)
        
        # Overall performance section
        report_lines.append("\n📊 OVERALL PERFORMANCE METRICS")
        report_lines.append("-"*50)
        report_lines.append(f"Accuracy:           {results['accuracy']:.4f}")
        report_lines.append(f"Precision (Macro):  {results['precision_macro']:.4f}")
        report_lines.append(f"Recall (Macro):     {results['recall_macro']:.4f}")
        report_lines.append(f"F1-Score (Macro):   {results['f1_macro']:.4f}")
        report_lines.append(f"Precision (Weighted): {results['precision_weighted']:.4f}")
        report_lines.append(f"Recall (Weighted):  {results['recall_weighted']:.4f}")
        report_lines.append(f"F1-Score (Weighted): {results['f1_weighted']:.4f}")
        
        if 'log_loss' in results and results['log_loss'] is not None:
            report_lines.append(f"Log Loss:           {results['log_loss']:.4f}")
        
        if 'roc_auc' in results:
            report_lines.append(f"ROC AUC:            {results['roc_auc']:.4f}")
        elif 'roc_auc_macro' in results:
            report_lines.append(f"ROC AUC (Macro):    {results['roc_auc_macro']:.4f}")
        
        # Per-class performance section
        report_lines.append("\n🎯 PER-CLASS PERFORMANCE")
        report_lines.append("-"*50)
        report_lines.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        report_lines.append("-"*55)
        
        for class_name, metrics in results['per_class_metrics'].items():
            report_lines.append(
                f"{class_name:<15} {metrics['precision']:<10.3f} "
                f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f} "
                f"{metrics['support']:<10d}"
            )
        
        # Confusion matrix section
        report_lines.append("\n🔄 CONFUSION MATRIX")
        report_lines.append("-"*50)
        cm = results['confusion_matrix']
        
        # Create confusion matrix display
        class_names = list(results['per_class_metrics'].keys())
        
        # Header
        header = "Actual\\Predicted"
        for class_name in class_names:
            header += f"{class_name:>12}"
        report_lines.append(header)
        
        # Matrix rows
        for i, class_name in enumerate(class_names):
            row = f"{class_name:<15}"
            for j in range(len(class_names)):
                row += f"{cm[i, j]:>12d}"
            report_lines.append(row)
        
        # Model interpretation section
        report_lines.append("\n🔍 MODEL INTERPRETATION")
        report_lines.append("-"*50)
        
        # Overall performance interpretation
        accuracy = results['accuracy']
        if accuracy >= 0.9:
            performance_level = "Excellent"
        elif accuracy >= 0.8:
            performance_level = "Good"
        elif accuracy >= 0.7:
            performance_level = "Moderate"
        else:
            performance_level = "Needs Improvement"
        
        report_lines.append(f"Overall Performance: {performance_level} ({accuracy:.1%} accuracy)")
        
        # Find best and worst performing classes
        class_f1_scores = {name: metrics['f1_score'] for name, metrics in results['per_class_metrics'].items()}
        best_class = max(class_f1_scores, key=class_f1_scores.get)
        worst_class = min(class_f1_scores, key=class_f1_scores.get)
        
        report_lines.append(f"Best performing class: {best_class} (F1: {class_f1_scores[best_class]:.3f})")
        report_lines.append(f"Worst performing class: {worst_class} (F1: {class_f1_scores[worst_class]:.3f})")
        
        # Balance analysis
        macro_f1 = results['f1_macro']
        weighted_f1 = results['f1_weighted']
        balance_diff = abs(macro_f1 - weighted_f1)
        
        if balance_diff < 0.05:
            balance_status = "Well-balanced performance across classes"
        else:
            balance_status = "Imbalanced performance - some classes perform much better than others"
        
        report_lines.append(f"Class balance: {balance_status}")
        
        # Recommendations section
        report_lines.append("\n💡 RECOMMENDATIONS")
        report_lines.append("-"*50)
        
        if accuracy < 0.7:
            report_lines.append("• Consider feature engineering or more complex models")
            report_lines.append("• Check data quality and preprocessing steps")
        
        if balance_diff > 0.1:
            report_lines.append("• Address class imbalance with techniques like SMOTE or class weights")
            report_lines.append("• Collect more data for underperforming classes")
        
        if 'log_loss' in results and results['log_loss'] and results['log_loss'] > 1.0:
            report_lines.append("• High log loss suggests poor probability calibration")
            report_lines.append("• Consider probability calibration techniques")
        
        # Join all lines
        report = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"📄 Comprehensive report saved to: {save_path}")
        
        print(report)
        return report
    
    def calculate_statistical_significance(self, results1, results2, alpha=0.05):
        """
        Calculate statistical significance between two model results.
        
        Args:
            results1 (dict): Results from first model
            results2 (dict): Results from second model
            alpha (float): Significance level (default: 0.05)
            
        Returns:
            dict: Statistical significance test results
        """
        try:
            from scipy.stats import chi2_contingency, mcnemar
        except ImportError:
            from scipy.stats import chi2_contingency
            try:
                from scipy.stats.contingency import mcnemar
            except ImportError:
                # Simple fallback implementation
                from scipy.stats import chi2
                def mcnemar(table, exact=True, correction=True):
                    b, c = table[0, 1], table[1, 0]
                    if correction:
                        chi2_stat = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
                    else:
                        chi2_stat = (b - c)**2 / (b + c) if (b + c) > 0 else 0
                    p_value = 1 - chi2.cdf(chi2_stat, 1) if chi2_stat > 0 else 1.0
                    return type('McNemarResult', (), {'statistic': abs(b - c), 'pvalue': p_value})()
        
        print(f"\n📊 Statistical Significance Test")
        print(f"   Comparing: {results1['model_name']} vs {results2['model_name']}")
        print(f"   Significance level: {alpha}")
        
        # Get predictions
        y_true = results1['y_true']
        y_pred1 = results1['y_pred'] 
        y_pred2 = results2['y_pred']
        
        # Calculate accuracy difference
        acc1 = results1['accuracy']
        acc2 = results2['accuracy']
        acc_diff = abs(acc1 - acc2)
        
        # McNemar's test for paired predictions
        # Create contingency table: correct/incorrect for each model
        model1_correct = (y_true == y_pred1)
        model2_correct = (y_true == y_pred2)
        
        # McNemar table: [both_correct, model1_correct_model2_wrong, 
        #                 model1_wrong_model2_correct, both_wrong]
        both_correct = np.sum(model1_correct & model2_correct)
        model1_only = np.sum(model1_correct & ~model2_correct)
        model2_only = np.sum(~model1_correct & model2_correct)
        both_wrong = np.sum(~model1_correct & ~model2_correct)
        
        # McNemar's test
        mcnemar_table = np.array([[both_correct, model1_only],
                                  [model2_only, both_wrong]])
        
        try:
            mcnemar_stat, mcnemar_p = mcnemar(mcnemar_table, exact=False, correction=True)
            is_significant = mcnemar_p < alpha
        except:
            mcnemar_stat, mcnemar_p = None, None
            is_significant = False
        
        results = {
            'model1_name': results1['model_name'],
            'model2_name': results2['model_name'],
            'model1_accuracy': acc1,
            'model2_accuracy': acc2,
            'accuracy_difference': acc_diff,
            'mcnemar_statistic': mcnemar_stat,
            'mcnemar_p_value': mcnemar_p,
            'is_significant': is_significant,
            'alpha': alpha,
            'contingency_table': mcnemar_table
        }
        
        print(f"   {results1['model_name']} accuracy: {acc1:.4f}")
        print(f"   {results2['model_name']} accuracy: {acc2:.4f}")
        print(f"   Accuracy difference: {acc_diff:.4f}")
        
        if mcnemar_p is not None:
            print(f"   McNemar's test p-value: {mcnemar_p:.4f}")
            if is_significant:
                print(f"   ✅ Difference is statistically significant (p < {alpha})")
            else:
                print(f"   ❌ Difference is NOT statistically significant (p >= {alpha})")
        else:
            print(f"   ⚠️  Could not perform McNemar's test")
        
        return results
    
    def compare_multiple_models(self, model_results_list):
        """
        Compare multiple model results side by side.
        
        Args:
            model_results_list (list): List of evaluation results from different models
            
        Returns:
            pandas.DataFrame: Comparison table
        """
        if len(model_results_list) < 2:
            print("⚠️  Need at least 2 models for comparison")
            return None
        
        print(f"\n📊 MULTI-MODEL COMPARISON")
        print("="*80)
        
        # Create comparison dataframe
        comparison_data = []
        
        for results in model_results_list:
            row = {
                'Model': results['model_name'],
                'Accuracy': results['accuracy'],
                'Precision (Macro)': results['precision_macro'],
                'Recall (Macro)': results['recall_macro'],
                'F1-Score (Macro)': results['f1_macro'],
                'Precision (Weighted)': results['precision_weighted'],
                'Recall (Weighted)': results['recall_weighted'],
                'F1-Score (Weighted)': results['f1_weighted']
            }
            
            # Add ROC AUC if available
            if 'roc_auc' in results:
                row['ROC AUC'] = results['roc_auc']
            elif 'roc_auc_macro' in results:
                row['ROC AUC'] = results['roc_auc_macro']
            
            # Add log loss if available
            if 'log_loss' in results and results['log_loss'] is not None:
                row['Log Loss'] = results['log_loss']
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Display the comparison table
        print("🏆 PERFORMANCE COMPARISON TABLE")
        print("-"*80)
        
        # Format and display
        pd.set_option('display.precision', 4)
        pd.set_option('display.width', None)
        pd.set_option('display.max_columns', None)
        print(df.to_string(index=False))
        
        # Find best model for each metric
        print("\n🥇 BEST PERFORMERS BY METRIC")
        print("-"*40)
        
        metrics_to_maximize = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 
                              'F1-Score (Macro)', 'ROC AUC']
        metrics_to_minimize = ['Log Loss']
        
        for metric in metrics_to_maximize:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_model = df.loc[best_idx, 'Model']
                best_value = df.loc[best_idx, metric]
                print(f"{metric:<20}: {best_model} ({best_value:.4f})")
        
        for metric in metrics_to_minimize:
            if metric in df.columns:
                best_idx = df[metric].idxmin()
                best_model = df.loc[best_idx, 'Model']
                best_value = df.loc[best_idx, metric]
                print(f"{metric:<20}: {best_model} ({best_value:.4f})")
        
        # Overall ranking (based on F1-Score Macro)
        print(f"\n🏆 OVERALL RANKING (by F1-Score Macro)")
        print("-"*40)
        df_sorted = df.sort_values('F1-Score (Macro)', ascending=False)
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            print(f"{i}. {row['Model']:<20} (F1: {row['F1-Score (Macro)']:.4f})")
        
        return df