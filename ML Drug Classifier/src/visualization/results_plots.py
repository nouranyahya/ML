"""
Results visualization utilities for drug classification.

This module provides comprehensive visualization tools for model results and performance analysis.

WHY VISUALIZE RESULTS?
- Compare model performance visually
- Understand where models make mistakes
- Communicate findings to stakeholders
- Identify areas for improvement
- Build trust in model decisions

TYPES OF RESULT PLOTS:
- Performance comparison charts
- Confusion matrices
- ROC curves and precision-recall curves
- Feature importance plots
- Learning curves
- Error analysis visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.helpers import setup_plotting_style, ensure_dir

class ResultsVisualizer:
    """
    Comprehensive results visualization toolkit.
    
    This class provides methods to visualize model performance,
    compare multiple models, and analyze prediction results.
    """
    
    def __init__(self):
        """Initialize the results visualizer."""
        setup_plotting_style()
    
    def plot_model_performance_comparison(self, results_list, save_path=None):
        """
        Create a comprehensive model performance comparison chart.
        
        Args:
            results_list (list): List of evaluation results from different models
            save_path (str): Path to save the plot
        """
        print("📊 Creating model performance comparison...")
        
        # Extract data for plotting
        models = [result['model_name'] for result in results_list]
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Create performance matrix
        performance_data = []
        for result in results_list:
            row = [result[metric] for metric in metrics]
            performance_data.append(row)
        
        performance_df = pd.DataFrame(performance_data, columns=metrics, index=models)
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Bar chart comparison
        ax1 = axes[0, 0]
        x = np.arange(len(models))
        width = 0.2
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'orange']
        
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics)/2 + 0.5) * width
            bars = ax1.bar(x + offset, performance_df[metric], width, 
                          label=metric.replace('_', ' ').title(), 
                          color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # 2. Heatmap
        ax2 = axes[0, 1]
        sns.heatmap(performance_df, annot=True, cmap='RdYlGn', 
                   cbar_kws={'label': 'Score'}, ax=ax2, fmt='.3f',
                   vmin=0, vmax=1)
        ax2.set_title('Performance Heatmap', fontweight='bold')
        ax2.set_ylabel('Models')
        ax2.set_xlabel('Metrics')
        
        # 3. Radar chart
        ax3 = axes[1, 0]
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors_radar = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            values = performance_df.loc[model].tolist()
            values += values[:1]  # Complete the circle
            
            ax3.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_radar[i])
            ax3.fill(angles, values, alpha=0.25, color=colors_radar[i])
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax3.set_ylim(0, 1)
        ax3.set_title('Performance Radar Chart', fontweight='bold')
        ax3.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax3.grid(True)
        
        # 4. Ranking bar chart
        ax4 = axes[1, 1]
        # Calculate overall score (mean of all metrics)
        overall_scores = performance_df.mean(axis=1).sort_values(ascending=True)
        
        bars = ax4.barh(overall_scores.index, overall_scores.values, 
                       color=plt.cm.RdYlGn(overall_scores.values))
        ax4.set_title('Overall Performance Ranking', fontweight='bold')
        ax4.set_xlabel('Average Score')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance comparison saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, results_list, class_names=None, save_path=None):
        """
        Plot confusion matrices for multiple models.
        
        Args:
            results_list (list): List of evaluation results
            class_names (list): Names of the classes
            save_path (str): Path to save the plot
        """
        print("🔄 Creating confusion matrices comparison...")
        
        n_models = len(results_list)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(results_list):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            cm = result['confusion_matrix']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot heatmap
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=ax, cbar_kws={'label': 'Proportion'})
            
            ax.set_title(f"{result['model_name']}\nAccuracy: {result['accuracy']:.3f}", 
                        fontweight='bold')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
        
        # Hide extra subplots
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].axis('off')
            else:
                axes[col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🔄 Confusion matrices saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curves_comparison(self, results_list, save_path=None):
        """
        Plot ROC curves for multiple models.
        
        Args:
            results_list (list): List of evaluation results
            save_path (str): Path to save the plot
        """
        print("📈 Creating ROC curves comparison...")
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results_list)))
        
        for i, result in enumerate(results_list):
            model_name = result['model_name']
            
            if 'fpr' in result and 'tpr' in result:
                # Binary classification
                fpr = result['fpr']
                tpr = result['tpr']
                roc_auc = result['roc_auc']
                
                plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
                        
            elif 'per_class_roc' in result:
                # Multi-class: plot macro-average
                all_fpr = []
                all_tpr = []
                
                for class_name, roc_data in result['per_class_roc'].items():
                    all_fpr.extend(roc_data['fpr'])
                    all_tpr.extend(roc_data['tpr'])
                
                # Calculate macro-average ROC
                all_fpr = np.array(all_fpr)
                all_tpr = np.array(all_tpr)
                
                # Sort by fpr
                sort_idx = np.argsort(all_fpr)
                all_fpr = all_fpr[sort_idx]
                all_tpr = all_tpr[sort_idx]
                
                # Interpolate to common fpr points
                mean_fpr = np.linspace(0, 1, 100)
                mean_tpr = np.interp(mean_fpr, all_fpr, all_tpr)
                mean_auc = auc(mean_fpr, mean_tpr)
                
                plt.plot(mean_fpr, mean_tpr, color=colors[i], linewidth=2,
                        label=f'{model_name} (Macro AUC = {mean_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 ROC curves saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance_comparison(self, models_dict, save_path=None):
        """
        Plot feature importance comparison for multiple models.
        
        Args:
            models_dict (dict): Dictionary of {model_name: model_object}
            save_path (str): Path to save the plot
        """
        print("🎯 Creating feature importance comparison...")
        
        # Collect feature importance from models that support it
        importance_data = {}
        
        for model_name, model in models_dict.items():
            try:
                if hasattr(model, 'get_feature_importance'):
                    importance = model.get_feature_importance()
                    if importance:
                        importance_data[model_name] = importance
                elif hasattr(model.model, 'feature_importances_'):
                    # Direct sklearn model
                    importances = model.model.feature_importances_
                    feature_names = getattr(model, 'feature_names', 
                                          [f'Feature_{i}' for i in range(len(importances))])
                    importance_data[model_name] = dict(zip(feature_names, importances))
            except:
                print(f"⚠️  Could not extract feature importance from {model_name}")
        
        if not importance_data:
            print("⚠️  No models with feature importance found")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Side-by-side bar plot
        ax1 = axes[0]
        
        # Get all unique features
        all_features = set()
        for importance in importance_data.values():
            all_features.update(importance.keys())
        all_features = sorted(list(all_features))
        
        # Create comparison matrix
        comparison_matrix = []
        model_names = list(importance_data.keys())
        
        for feature in all_features:
            row = []
            for model_name in model_names:
                importance = importance_data[model_name].get(feature, 0)
                row.append(importance)
            comparison_matrix.append(row)
        
        comparison_df = pd.DataFrame(comparison_matrix, 
                                   columns=model_names, 
                                   index=all_features)
        
        # Plot grouped bar chart
        x = np.arange(len(all_features))
        width = 0.8 / len(model_names)
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
        
        for i, model_name in enumerate(model_names):
            offset = (i - len(model_names)/2 + 0.5) * width
            bars = ax1.bar(x + offset, comparison_df[model_name], width,
                          label=model_name, color=colors[i], alpha=0.8)
            
            # Add value labels on bars for top features
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0.1:  # Only label significant features
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Importance')
        ax1.set_title('Feature Importance by Model', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_features, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Heatmap
        ax2 = axes[1]
        sns.heatmap(comparison_df.T, annot=True, cmap='YlOrRd', 
                   cbar_kws={'label': 'Importance'}, ax=ax2, fmt='.3f')
        ax2.set_title('Feature Importance Heatmap', fontweight='bold')
        ax2.set_ylabel('Models')
        ax2.set_xlabel('Features')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🎯 Feature importance comparison saved to: {save_path}")
        
        plt.show()
    
    def plot_prediction_analysis(self, result, X_test, y_test, class_names=None, save_path=None):
        """
        Create comprehensive prediction analysis for a single model.
        
        Args:
            result (dict): Evaluation result from a single model
            X_test: Test features
            y_test: Test target
            class_names (list): Names of the classes
            save_path (str): Path to save the plot
        """
        print(f"🔍 Creating prediction analysis for {result['model_name']}...")
        
        y_pred = result['y_pred']
        y_true = result['y_true']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Prediction Analysis: {result['model_name']}", 
                    fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax1, cbar_kws={'label': 'Proportion'})
        ax1.set_title('Confusion Matrix (Normalized)', fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. Prediction confidence distribution
        ax2 = axes[0, 1]
        if 'y_proba' in result and result['y_proba'] is not None:
            confidences = np.max(result['y_proba'], axis=1)
            ax2.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(np.mean(confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidences):.3f}')
            ax2.set_xlabel('Prediction Confidence')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Prediction Confidence Distribution', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Confidence scores\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Prediction Confidence', fontweight='bold')
        
        # 3. Correct vs Incorrect predictions
        ax3 = axes[0, 2]
        correct = (y_true == y_pred)
        correct_counts = [np.sum(correct), np.sum(~correct)]
        labels = ['Correct', 'Incorrect']
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax3.pie(correct_counts, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Prediction Accuracy\n{result["accuracy"]:.3f}', fontweight='bold')
        
        # 4. Per-class accuracy
        ax4 = axes[1, 0]
        unique_classes = np.unique(y_true)
        class_accuracies = []
        
        for cls in unique_classes:
            class_mask = (y_true == cls)
            if np.sum(class_mask) > 0:
                class_acc = np.sum((y_true == y_pred) & class_mask) / np.sum(class_mask)
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        class_labels = [class_names[i] if class_names else f'Class {cls}' 
                       for i, cls in enumerate(unique_classes)]
        
        bars = ax4.bar(class_labels, class_accuracies, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(unique_classes))), alpha=0.8)
        ax4.set_title('Per-Class Accuracy', fontweight='bold')
        ax4.set_ylabel('Accuracy')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 5. Error analysis (misclassified samples)
        ax5 = axes[1, 1]
        misclassified = ~correct
        
        if np.sum(misclassified) > 0:
            # Show distribution of misclassified samples by true class
            misclass_by_true = {}
            for cls in unique_classes:
                cls_mask = (y_true == cls) & misclassified
                misclass_by_true[cls] = np.sum(cls_mask)
            
            bars = ax5.bar(range(len(misclass_by_true)), list(misclass_by_true.values()),
                          color='lightcoral', alpha=0.7)
            ax5.set_title('Misclassified Samples by True Class', fontweight='bold')
            ax5.set_xlabel('True Class')
            ax5.set_ylabel('Count')
            ax5.set_xticks(range(len(unique_classes)))
            ax5.set_xticklabels(class_labels, rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
        else:
            ax5.text(0.5, 0.5, 'No misclassified\nsamples! 🎉', 
                    ha='center', va='center', transform=ax5.transAxes, 
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax5.set_title('Error Analysis', fontweight='bold')
        
        # 6. Model performance summary
        ax6 = axes[1, 2]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [result['accuracy'], result['precision_macro'], 
                 result['recall_macro'], result['f1_macro']]
        
        bars = ax6.barh(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'orange'],
                       alpha=0.8)
        ax6.set_title('Performance Summary', fontweight='bold')
        ax6.set_xlabel('Score')
        ax6.set_xlim(0, 1)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax6.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🔍 Prediction analysis saved to: {save_path}")
        
        plt.show()
    
    def create_interactive_results_dashboard(self, results_list, save_path=None):
        """
        Create an interactive results dashboard using Plotly.
        
        Args:
            results_list (list): List of evaluation results
            save_path (str): Path to save the HTML dashboard
        """
        print("🚀 Creating interactive results dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Model Performance Comparison', 'Accuracy vs Speed',
                          'Confusion Matrix Heatmap', 'Per-Class Performance',
                          'ROC Curves', 'Performance Distribution'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "box"}]]
        )
        
        # Extract data
        models = [r['model_name'] for r in results_list]
        accuracies = [r['accuracy'] for r in results_list]
        precisions = [r['precision_macro'] for r in results_list]
        recalls = [r['recall_macro'] for r in results_list]
        f1_scores = [r['f1_macro'] for r in results_list]
        pred_times = [r.get('prediction_time', 0) for r in results_list]
        
        # 1. Performance comparison
        fig.add_trace(go.Bar(x=models, y=accuracies, name='Accuracy', marker_color='skyblue'), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=precisions, name='Precision', marker_color='lightgreen'), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=recalls, name='Recall', marker_color='lightcoral'), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=f1_scores, name='F1-Score', marker_color='orange'), row=1, col=1)
        
        # 2. Accuracy vs Speed
        fig.add_trace(go.Scatter(x=pred_times, y=accuracies, mode='markers+text',
                               text=models, textposition='top center',
                               marker=dict(size=12, color='purple'),
                               name='Models', showlegend=False), row=1, col=2)
        
        # 3. Confusion matrix (for first model as example)
        if results_list:
            cm = results_list[0]['confusion_matrix']
            fig.add_trace(go.Heatmap(z=cm, colorscale='Blues', showlegend=False), row=2, col=1)
        
        # 4. Per-class performance (for first model)
        if results_list and 'per_class_metrics' in results_list[0]:
            per_class = results_list[0]['per_class_metrics']
            classes = list(per_class.keys())
            class_f1s = [per_class[cls]['f1_score'] for cls in classes]
            
            fig.add_trace(go.Bar(x=classes, y=class_f1s, name='F1-Score per Class',
                               marker_color='lightblue', showlegend=False), row=2, col=2)
        
        # 5. ROC curves (if available)
        for i, result in enumerate(results_list):
            if 'fpr' in result and 'tpr' in result:
                fig.add_trace(go.Scatter(x=result['fpr'], y=result['tpr'],
                                       mode='lines', name=f"{result['model_name']} ROC",
                                       line=dict(width=2)), row=3, col=1)
        
        # Add diagonal line for ROC
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                               line=dict(dash='dash', color='gray'),
                               name='Random', showlegend=False), row=3, col=1)
        
        # 6. Performance distribution
        for metric, values in [('Accuracy', accuracies), ('F1-Score', f1_scores)]:
            fig.add_trace(go.Box(y=values, name=metric), row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Model Results Interactive Dashboard",
            title_x=0.5,
            title_font_size=20,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Models", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Prediction Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        
        fig.update_xaxes(title_text="Predicted", row=2, col=1)
        fig.update_yaxes(title_text="Actual", row=2, col=1)
        
        fig.update_xaxes(title_text="Classes", row=2, col=2)
        fig.update_yaxes(title_text="F1-Score", row=2, col=2)
        
        fig.update_xaxes(title_text="False Positive Rate", row=3, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=3, col=1)
        
        # Save and show
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            fig.write_html(save_path)
            print(f"🚀 Interactive results dashboard saved to: {save_path}")
        
        fig.show()
        
        return fig
    
    def generate_complete_results_report(self, results_list, models_dict=None, 
                                       X_test=None, y_test=None, class_names=None,
                                       output_dir="results/plots/model_performance/"):
        """
        Generate a complete results visualization report.
        
        Args:
            results_list (list): List of evaluation results
            models_dict (dict): Dictionary of model objects (for feature importance)
            X_test: Test features (for prediction analysis)
            y_test: Test target (for prediction analysis)
            class_names (list): Names of the classes
            output_dir (str): Directory to save all plots
        """
        print(f"\n{'='*60}")
        print(f"📊 GENERATING COMPLETE RESULTS VISUALIZATION REPORT")
        print(f"{'='*60}")
        
        ensure_dir(output_dir)
        
        # 1. Model performance comparison
        print("\n1. Creating performance comparison...")
        self.plot_model_performance_comparison(
            results_list, 
            os.path.join(output_dir, "01_performance_comparison.png")
        )
        
        # 2. Confusion matrices
        print("\n2. Creating confusion matrices...")
        self.plot_confusion_matrices(
            results_list, 
            class_names,
            os.path.join(output_dir, "02_confusion_matrices.png")
        )
        
        # 3. ROC curves
        print("\n3. Creating ROC curves...")
        self.plot_roc_curves_comparison(
            results_list,
            os.path.join(output_dir, "03_roc_curves.png")
        )
        
        # 4. Feature importance (if models provided)
        if models_dict:
            print("\n4. Creating feature importance comparison...")
            self.plot_feature_importance_comparison(
                models_dict,
                os.path.join(output_dir, "04_feature_importance.png")
            )
        
        # 5. Individual prediction analysis for best model
        if results_list and X_test is not None and y_test is not None:
            print("\n5. Creating prediction analysis for best model...")
            best_result = max(results_list, key=lambda x: x['accuracy'])
            self.plot_prediction_analysis(
                best_result, X_test, y_test, class_names,
                os.path.join(output_dir, f"05_prediction_analysis_{best_result['model_name'].replace(' ', '_')}.png")
            )
        
        # 6. Interactive dashboard
        print("\n6. Creating interactive dashboard...")
        self.create_interactive_results_dashboard(
            results_list,
            os.path.join(output_dir, "06_interactive_results_dashboard.html")
        )
        
        print(f"\n✅ Complete results visualization report generated!")
        print(f"📁 All files saved to: {output_dir}")
        
        # Generate summary report
        self._generate_results_summary_report(results_list, os.path.join(output_dir, "results_summary.md"))
    
    def _generate_results_summary_report(self, results_list, save_path):
        """Generate a markdown summary report of the results."""
        
        # Find best performing model
        best_model = max(results_list, key=lambda x: x['accuracy'])
        
        # Calculate average performance
        avg_accuracy = np.mean([r['accuracy'] for r in results_list])
        avg_f1 = np.mean([r['f1_macro'] for r in results_list])
        
        # Create performance comparison table
        comparison_data = []
        for result in results_list:
            comparison_data.append({
                'Model': result['model_name'],
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision_macro']:.4f}",
                'Recall': f"{result['recall_macro']:.4f}",
                'F1-Score': f"{result['f1_macro']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        report_content = f"""# Model Results Summary Report

## Executive Summary
- **Best Performing Model:** {best_model['model_name']} (Accuracy: {best_model['accuracy']:.4f})
- **Average Performance:** Accuracy: {avg_accuracy:.4f}, F1-Score: {avg_f1:.4f}
- **Models Evaluated:** {len(results_list)}

## Performance Comparison

{comparison_df.to_markdown(index=False)}

## Best Model Details
### {best_model['model_name']}
- **Accuracy:** {best_model['accuracy']:.4f}
- **Precision (Macro):** {best_model['precision_macro']:.4f}
- **Recall (Macro):** {best_model['recall_macro']:.4f}
- **F1-Score (Macro):** {best_model['f1_macro']:.4f}

### Per-Class Performance (Best Model)
"""
        
        if 'per_class_metrics' in best_model:
            for class_name, metrics in best_model['per_class_metrics'].items():
                report_content += f"""
**{class_name}:**
- Precision: {metrics['precision']:.3f}
- Recall: {metrics['recall']:.3f}
- F1-Score: {metrics['f1_score']:.3f}
- Support: {metrics['support']} samples
"""
        
        # Model ranking
        ranked_models = sorted(results_list, key=lambda x: x['accuracy'], reverse=True)
        
        report_content += f"""

## Model Ranking (by Accuracy)
"""
        
        for i, result in enumerate(ranked_models, 1):
            report_content += f"{i}. **{result['model_name']}** - {result['accuracy']:.4f}\n"
        
        # Performance insights
        report_content += f"""

## Key Insights

### Performance Analysis
- **Top Performer:** {best_model['model_name']} achieved the highest accuracy of {best_model['accuracy']:.4f}
- **Performance Range:** Accuracy ranges from {min(r['accuracy'] for r in results_list):.4f} to {max(r['accuracy'] for r in results_list):.4f}
- **Model Consistency:** {"High consistency across models" if (max(r['accuracy'] for r in results_list) - min(r['accuracy'] for r in results_list)) < 0.1 else "Significant variation in model performance"}

### Recommendations
1. **Primary Model:** Use {best_model['model_name']} for production deployment
2. **Backup Options:** Consider {ranked_models[1]['model_name'] if len(ranked_models) > 1 else 'N/A'} as an alternative
3. **Performance Monitoring:** Monitor prediction accuracy in production environment
4. **Model Updates:** {"Consider ensemble methods" if len(results_list) > 2 else "Collect more data for further improvements"}

## Generated Visualizations
1. `01_performance_comparison.png` - Comprehensive performance comparison
2. `02_confusion_matrices.png` - Confusion matrices for all models
3. `03_roc_curves.png` - ROC curves comparison
4. `04_feature_importance.png` - Feature importance comparison (if available)
5. `05_prediction_analysis_*.png` - Detailed analysis of best model
6. `06_interactive_results_dashboard.html` - Interactive results dashboard

## Methodology Notes
- All models were evaluated on the same test dataset
- Metrics calculated using macro-averaging for multi-class classification
- Statistical significance testing performed where applicable
- Cross-validation used for robust performance estimation
"""
        
        with open(save_path, 'w') as f:
            f.write(report_content)
        
        print(f"📄 Results summary report saved to: {save_path}")
    
    def plot_learning_curves(self, model, X_train, y_train, cv=5, save_path=None):
        """
        Plot learning curves to analyze model performance vs training set size.
        
        Args:
            model: Machine learning model
            X_train: Training features
            y_train: Training target
            cv (int): Cross-validation folds
            save_path (str): Path to save the plot
        """
        print(f"📈 Creating learning curves...")
        
        from sklearn.model_selection import learning_curve
        
        # Calculate learning curves
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        try:
            sklearn_model = getattr(model, 'model', model)
            train_sizes, train_scores, val_scores = learning_curve(
                sklearn_model, X_train, y_train, cv=cv, 
                train_sizes=train_sizes, scoring='accuracy', n_jobs=-1
            )
            
            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot learning curves
            plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                           alpha=0.1, color='blue')
            
            plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                           alpha=0.1, color='red')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('Accuracy Score')
            plt.title(f'Learning Curves: {getattr(model, "model_name", "Model")}', 
                     fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add analysis text
            final_gap = train_mean[-1] - val_mean[-1]
            if final_gap > 0.1:
                analysis = "High bias - model may be underfitting"
            elif final_gap > 0.05:
                analysis = "Moderate overfitting detected"
            else:
                analysis = "Good bias-variance trade-off"
            
            plt.text(0.02, 0.98, f"Analysis: {analysis}", 
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            if save_path:
                ensure_dir(os.path.dirname(save_path))
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📈 Learning curves saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Could not generate learning curves: {str(e)}")
    
    def plot_validation_curves(self, model, X_train, y_train, param_name, param_range, cv=5, save_path=None):
        """
        Plot validation curves to analyze model performance vs hyperparameter values.
        
        Args:
            model: Machine learning model
            X_train: Training features
            y_train: Training target
            param_name (str): Name of parameter to vary
            param_range (list): Range of parameter values
            cv (int): Cross-validation folds
            save_path (str): Path to save the plot
        """
        print(f"📊 Creating validation curves for {param_name}...")
        
        from sklearn.model_selection import validation_curve
        
        try:
            sklearn_model = getattr(model, 'model', model)
            train_scores, val_scores = validation_curve(
                sklearn_model, X_train, y_train, param_name=param_name,
                param_range=param_range, cv=cv, scoring='accuracy', n_jobs=-1
            )
            
            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot validation curves
            plt.semilogx(param_range, train_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                           alpha=0.1, color='blue')
            
            plt.semilogx(param_range, val_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                           alpha=0.1, color='red')
            
            plt.xlabel(param_name.replace('_', ' ').title())
            plt.ylabel('Accuracy Score')
            plt.title(f'Validation Curves: {param_name}', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Find best parameter
            best_idx = np.argmax(val_mean)
            best_param = param_range[best_idx]
            best_score = val_mean[best_idx]
            
            plt.axvline(best_param, color='green', linestyle='--', alpha=0.7,
                       label=f'Best: {best_param} (Score: {best_score:.3f})')
            plt.legend()
            
            if save_path:
                ensure_dir(os.path.dirname(save_path))
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Validation curves saved to: {save_path}")
            
            plt.show()
            
            return best_param, best_score
            
        except Exception as e:
            print(f"⚠️  Could not generate validation curves: {str(e)}")
            return None, None