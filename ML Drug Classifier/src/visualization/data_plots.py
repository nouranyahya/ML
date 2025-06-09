"""
Data visualization utilities for drug classification.

This module provides comprehensive data exploration and visualization tools.

WHY VISUALIZE DATA?
"A picture is worth a thousand words" - visualizations help us:
1. Understand data patterns and distributions
2. Identify outliers and anomalies
3. Discover relationships between features
4. Communicate insights effectively
5. Make data-driven decisions

TYPES OF PLOTS WE'LL CREATE:
- Distribution plots: Show how data is spread
- Correlation plots: Show relationships between features
- Box plots: Show data spread and outliers
- Bar plots: Show categorical data counts
- Scatter plots: Show relationships between numerical features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.helpers import setup_plotting_style, ensure_dir

class DataVisualizer:
    """
    Comprehensive data visualization toolkit for drug classification dataset.
    
    This class provides methods to create insightful visualizations
    that help understand the data before building models.
    """
    
    def __init__(self):
        """Initialize the data visualizer."""
        setup_plotting_style()
    
    def create_overview_dashboard(self, df, save_path=None):
        """
        Create a comprehensive overview dashboard of the dataset.
        
        Args:
            df (pandas.DataFrame): The dataset to visualize
            save_path (str): Path to save the dashboard
        """
        print("📊 Creating comprehensive data overview dashboard...")
        
        # Create subplot layout
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Dataset shape and basic info
        ax1 = plt.subplot(3, 4, 1)
        info_text = f"""
Dataset Overview:
• Shape: {df.shape[0]} rows × {df.shape[1]} columns
• Features: {df.shape[1] - 1}
• Target: Drug
• Missing values: {df.isnull().sum().sum()}
        """
        ax1.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Dataset Summary', fontweight='bold')
        
        # 2. Target distribution
        ax2 = plt.subplot(3, 4, 2)
        drug_counts = df['Drug'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(drug_counts)))
        bars = ax2.bar(drug_counts.index, drug_counts.values, color=colors, alpha=0.8)
        ax2.set_title('Drug Distribution', fontweight='bold')
        ax2.set_xlabel('Drug Type')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 3. Age distribution
        ax3 = plt.subplot(3, 4, 3)
        ax3.hist(df['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Age Distribution', fontweight='bold')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        age_stats = f"Mean: {df['Age'].mean():.1f}\nStd: {df['Age'].std():.1f}"
        ax3.text(0.02, 0.98, age_stats, transform=ax3.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 4. Na_to_K distribution
        ax4 = plt.subplot(3, 4, 4)
        ax4.hist(df['Na_to_K'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_title('Na_to_K Ratio Distribution', fontweight='bold')
        ax4.set_xlabel('Na_to_K Ratio')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics
        nak_stats = f"Mean: {df['Na_to_K'].mean():.2f}\nStd: {df['Na_to_K'].std():.2f}"
        ax4.text(0.02, 0.98, nak_stats, transform=ax4.transAxes,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 5. Sex distribution
        ax5 = plt.subplot(3, 4, 5)
        sex_counts = df['Sex'].value_counts()
        ax5.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', 
               colors=['lightcoral', 'lightskyblue'], startangle=90)
        ax5.set_title('Sex Distribution', fontweight='bold')
        
        # 6. Blood Pressure distribution
        ax6 = plt.subplot(3, 4, 6)
        bp_counts = df['BP'].value_counts()
        bars = ax6.bar(bp_counts.index, bp_counts.values, 
                      color=['red', 'orange', 'green'], alpha=0.7)
        ax6.set_title('Blood Pressure Distribution', fontweight='bold')
        ax6.set_xlabel('Blood Pressure')
        ax6.set_ylabel('Count')
        
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 7. Cholesterol distribution
        ax7 = plt.subplot(3, 4, 7)
        chol_counts = df['Cholesterol'].value_counts()
        ax7.pie(chol_counts.values, labels=chol_counts.index, autopct='%1.1f%%',
               colors=['orange', 'lightgreen'], startangle=90)
        ax7.set_title('Cholesterol Distribution', fontweight='bold')
        
        # 8. Age vs Drug box plot
        ax8 = plt.subplot(3, 4, 8)
        df.boxplot(column='Age', by='Drug', ax=ax8)
        ax8.set_title('Age Distribution by Drug', fontweight='bold')
        ax8.set_xlabel('Drug Type')
        ax8.set_ylabel('Age')
        plt.suptitle('')  # Remove automatic title
        
        # 9. Na_to_K vs Drug box plot
        ax9 = plt.subplot(3, 4, 9)
        df.boxplot(column='Na_to_K', by='Drug', ax=ax9)
        ax9.set_title('Na_to_K Ratio by Drug', fontweight='bold')
        ax9.set_xlabel('Drug Type')
        ax9.set_ylabel('Na_to_K Ratio')
        plt.suptitle('')  # Remove automatic title
        
        # 10. Drug by Sex and BP
        ax10 = plt.subplot(3, 4, 10)
        cross_tab = pd.crosstab(df['Sex'], df['BP'])
        cross_tab.plot(kind='bar', ax=ax10, alpha=0.8)
        ax10.set_title('Sex vs Blood Pressure', fontweight='bold')
        ax10.set_xlabel('Sex')
        ax10.set_ylabel('Count')
        ax10.legend(title='Blood Pressure')
        ax10.tick_params(axis='x', rotation=0)
        
        # 11. Missing values heatmap
        ax11 = plt.subplot(3, 4, 11)
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            ax11.bar(missing_data.index, missing_data.values, color='red', alpha=0.7)
            ax11.set_title('Missing Values by Column', fontweight='bold')
            ax11.set_ylabel('Missing Count')
            ax11.tick_params(axis='x', rotation=45)
        else:
            ax11.text(0.5, 0.5, 'No Missing Values\n✅', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax11.transAxes, fontsize=16, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            ax11.set_title('Missing Values Status', fontweight='bold')
            ax11.axis('off')
        
        # 12. Correlation matrix (for numerical features)
        ax12 = plt.subplot(3, 4, 12)
        numerical_cols = ['Age', 'Na_to_K']
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax12, cbar_kws={'shrink': 0.8})
            ax12.set_title('Feature Correlation', fontweight='bold')
        else:
            ax12.text(0.5, 0.5, 'Need more numerical\nfeatures for correlation', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax12.transAxes, fontsize=12)
            ax12.set_title('Feature Correlation', fontweight='bold')
            ax12.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Overview dashboard saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_distributions(self, df, save_path=None):
        """
        Plot distributions of all features.
        
        Args:
            df (pandas.DataFrame): The dataset
            save_path (str): Path to save the plot
        """
        print("📊 Creating feature distribution plots...")
        
        # Separate numerical and categorical features
        numerical_features = ['Age', 'Na_to_K']
        categorical_features = ['Sex', 'BP', 'Cholesterol']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        
        # Plot numerical features
        for i, feature in enumerate(numerical_features):
            ax = axes[0, i]
            
            # Histogram with KDE
            ax.hist(df[feature], bins=20, alpha=0.7, density=True, 
                   color='skyblue', edgecolor='black', label='Histogram')
            
            # Add KDE curve
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(df[feature])
            x_range = np.linspace(df[feature].min(), df[feature].max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            ax.set_title(f'{feature} Distribution', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"Mean: {df[feature].mean():.2f}\nStd: {df[feature].std():.2f}\nSkew: {df[feature].skew():.2f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add empty subplot for symmetry
        axes[0, 2].axis('off')
        
        # Plot categorical features
        for i, feature in enumerate(categorical_features):
            ax = axes[1, i]
            
            value_counts = df[feature].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
            
            bars = ax.bar(value_counts.index, value_counts.values, 
                         color=colors, alpha=0.8, edgecolor='black')
            
            ax.set_title(f'{feature} Distribution', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{int(height)}', ha='center', va='bottom')
            
            # Add percentage labels
            total = value_counts.sum()
            for bar, count in zip(bars, value_counts.values):
                percentage = (count / total) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'{percentage:.1f}%', ha='center', va='center', 
                       fontweight='bold', color='white')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature distributions saved to: {save_path}")
        
        plt.show()
    
    def plot_target_analysis(self, df, save_path=None):
        """
        Comprehensive analysis of the target variable (Drug) and its relationships.
        
        Args:
            df (pandas.DataFrame): The dataset
            save_path (str): Path to save the plot
        """
        print("🎯 Creating target variable analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Drug (Target) Variable Analysis', fontsize=16, fontweight='bold')
        
        # 1. Drug distribution
        ax1 = axes[0, 0]
        drug_counts = df['Drug'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(drug_counts)))
        bars = ax1.bar(drug_counts.index, drug_counts.values, color=colors, alpha=0.8)
        ax1.set_title('Drug Distribution', fontweight='bold')
        ax1.set_xlabel('Drug Type')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add counts and percentages
        total = drug_counts.sum()
        for bar, count in zip(bars, drug_counts.values):
            height = bar.get_height()
            percentage = (count / total) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        # 2. Drug distribution pie chart
        ax2 = axes[0, 1]
        ax2.pie(drug_counts.values, labels=drug_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax2.set_title('Drug Distribution (Pie Chart)', fontweight='bold')
        
        # 3. Age vs Drug
        ax3 = axes[0, 2]
        for i, drug in enumerate(drug_counts.index):
            drug_ages = df[df['Drug'] == drug]['Age']
            ax3.hist(drug_ages, bins=15, alpha=0.6, label=drug, color=colors[i])
        ax3.set_title('Age Distribution by Drug', fontweight='bold')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Na_to_K vs Drug
        ax4 = axes[1, 0]
        df.boxplot(column='Na_to_K', by='Drug', ax=ax4)
        ax4.set_title('Na_to_K Ratio by Drug', fontweight='bold')
        ax4.set_xlabel('Drug Type')
        ax4.set_ylabel('Na_to_K Ratio')
        plt.suptitle('')  # Remove automatic title
        
        # 5. Drug by Sex
        ax5 = axes[1, 1]
        cross_tab_sex = pd.crosstab(df['Drug'], df['Sex'])
        cross_tab_sex.plot(kind='bar', ax=ax5, alpha=0.8, color=['lightcoral', 'lightskyblue'])
        ax5.set_title('Drug Distribution by Sex', fontweight='bold')
        ax5.set_xlabel('Drug Type')
        ax5.set_ylabel('Count')
        ax5.legend(title='Sex')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Drug by Blood Pressure
        ax6 = axes[1, 2]
        cross_tab_bp = pd.crosstab(df['Drug'], df['BP'])
        cross_tab_bp.plot(kind='bar', ax=ax6, alpha=0.8, 
                         color=['red', 'orange', 'green'])
        ax6.set_title('Drug Distribution by Blood Pressure', fontweight='bold')
        ax6.set_xlabel('Drug Type')
        ax6.set_ylabel('Count')
        ax6.legend(title='Blood Pressure')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🎯 Target analysis saved to: {save_path}")
        
        plt.show()
    
    def plot_correlation_analysis(self, df, save_path=None):
        """
        Create correlation analysis plots.
        
        Args:
            df (pandas.DataFrame): The dataset
            save_path (str): Path to save the plot
        """
        print("🔗 Creating correlation analysis...")
        
        # Create encoded version for correlation analysis
        df_encoded = df.copy()
        
        # Encode categorical variables
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        categorical_cols = ['Sex', 'BP', 'Cholesterol', 'Drug']
        for col in categorical_cols:
            df_encoded[col] = le.fit_transform(df_encoded[col])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Full correlation matrix
        ax1 = axes[0, 0]
        corr_matrix = df_encoded.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax1, fmt='.2f')
        ax1.set_title('Feature Correlation Matrix', fontweight='bold')
        
        # 2. Correlation with target (Drug)
        ax2 = axes[0, 1]
        target_corr = corr_matrix['Drug'].drop('Drug').abs().sort_values(ascending=True)
        bars = ax2.barh(target_corr.index, target_corr.values, 
                       color=plt.cm.RdYlBu(target_corr.values))
        ax2.set_title('Feature Correlation with Drug', fontweight='bold')
        ax2.set_xlabel('Absolute Correlation')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        # 3. Age vs Na_to_K scatter plot colored by Drug
        ax3 = axes[1, 0]
        drugs = df['Drug'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(drugs)))
        
        for i, drug in enumerate(drugs):
            drug_data = df[df['Drug'] == drug]
            ax3.scatter(drug_data['Age'], drug_data['Na_to_K'], 
                       alpha=0.7, label=drug, color=colors[i], s=50)
        
        ax3.set_title('Age vs Na_to_K by Drug', fontweight='bold')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Na_to_K Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature importance heatmap (correlation-based)
        ax4 = axes[1, 1]
        feature_importance = target_corr.abs()
        sns.heatmap(feature_importance.values.reshape(-1, 1), 
                   annot=True, cmap='Reds', yticklabels=feature_importance.index,
                   xticklabels=['Importance'], ax=ax4, fmt='.3f')
        ax4.set_title('Feature Importance (Correlation-based)', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🔗 Correlation analysis saved to: {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, df, save_path=None):
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            df (pandas.DataFrame): The dataset
            save_path (str): Path to save the HTML dashboard
        """
        print("🚀 Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Drug Distribution', 'Age Distribution', 'Na_to_K Distribution',
                          'Age vs Na_to_K by Drug', 'Drug by Sex', 'Drug by Blood Pressure',
                          'Age Box Plot by Drug', 'Na_to_K Box Plot by Drug', 'Feature Correlation'),
            specs=[[{"type": "bar"}, {"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "box"}, {"type": "heatmap"}]]
        )
        
        # 1. Drug distribution
        drug_counts = df['Drug'].value_counts()
        fig.add_trace(
            go.Bar(x=drug_counts.index, y=drug_counts.values, 
                  name="Drug Count", showlegend=False,
                  marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. Age distribution
        fig.add_trace(
            go.Histogram(x=df['Age'], name="Age", showlegend=False,
                        marker_color='lightgreen'),
            row=1, col=2
        )
        
        # 3. Na_to_K distribution
        fig.add_trace(
            go.Histogram(x=df['Na_to_K'], name="Na_to_K", showlegend=False,
                        marker_color='lightcoral'),
            row=1, col=3
        )
        
        # 4. Age vs Na_to_K scatter plot
        colors_map = {'DrugA': 'red', 'DrugB': 'blue', 'drugC': 'green', 
                     'drugX': 'orange', 'DrugY': 'purple'}
        
        for drug in df['Drug'].unique():
            drug_data = df[df['Drug'] == drug]
            fig.add_trace(
                go.Scatter(x=drug_data['Age'], y=drug_data['Na_to_K'],
                          mode='markers', name=drug,
                          marker=dict(color=colors_map.get(drug, 'gray')),
                          showlegend=True),
                row=2, col=1
            )
        
        # 5. Drug by Sex
        cross_tab_sex = pd.crosstab(df['Drug'], df['Sex'])
        for sex in cross_tab_sex.columns:
            fig.add_trace(
                go.Bar(x=cross_tab_sex.index, y=cross_tab_sex[sex],
                      name=f'Sex: {sex}'),
                row=2, col=2
            )
        
        # 6. Drug by Blood Pressure
        cross_tab_bp = pd.crosstab(df['Drug'], df['BP'])
        for bp in cross_tab_bp.columns:
            fig.add_trace(
                go.Bar(x=cross_tab_bp.index, y=cross_tab_bp[bp],
                      name=f'BP: {bp}'),
                row=2, col=3
            )
        
        # 7. Age box plot by Drug
        for drug in df['Drug'].unique():
            fig.add_trace(
                go.Box(y=df[df['Drug'] == drug]['Age'], name=drug,
                      showlegend=False),
                row=3, col=1
            )
        
        # 8. Na_to_K box plot by Drug
        for drug in df['Drug'].unique():
            fig.add_trace(
                go.Box(y=df[df['Drug'] == drug]['Na_to_K'], name=drug,
                      showlegend=False),
                row=3, col=2
            )
        
        # 9. Correlation heatmap
        df_encoded = df.copy()
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        categorical_cols = ['Sex', 'BP', 'Cholesterol', 'Drug']
        for col in categorical_cols:
            df_encoded[col] = le.fit_transform(df_encoded[col])
        
        corr_matrix = df_encoded.corr()
        
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values,
                      x=corr_matrix.columns,
                      y=corr_matrix.columns,
                      colorscale='RdBu',
                      showlegend=False),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Drug Classification Dataset - Interactive Dashboard",
            title_x=0.5,
            title_font_size=20
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Drug Type", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        fig.update_xaxes(title_text="Age", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Na_to_K Ratio", row=1, col=3)
        fig.update_yaxes(title_text="Frequency", row=1, col=3)
        
        fig.update_xaxes(title_text="Age", row=2, col=1)
        fig.update_yaxes(title_text="Na_to_K Ratio", row=2, col=1)
        
        # Save and show
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            fig.write_html(save_path)
            print(f"🚀 Interactive dashboard saved to: {save_path}")
        
        fig.show()
        
        return fig
    
    def plot_data_quality_report(self, df, save_path=None):
        """
        Create a comprehensive data quality report.
        
        Args:
            df (pandas.DataFrame): The dataset
            save_path (str): Path to save the plot
        """
        print("🔍 Creating data quality report...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Quality Assessment Report', fontsize=16, fontweight='bold')
        
        # 1. Missing values analysis
        ax1 = axes[0, 0]
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            bars = ax1.bar(missing_counts.index, missing_counts.values, 
                          color='red', alpha=0.7)
            ax1.set_title('Missing Values by Column', fontweight='bold')
            ax1.set_ylabel('Missing Count')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{int(height)}', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'No Missing Values Found\n✅', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            ax1.set_title('Missing Values Status', fontweight='bold')
            ax1.axis('off')
        
        # 2. Data types overview
        ax2 = axes[0, 1]
        dtypes_count = df.dtypes.value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(dtypes_count)))
        ax2.pie(dtypes_count.values, labels=dtypes_count.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax2.set_title('Data Types Distribution', fontweight='bold')
        
        # 3. Unique values per column
        ax3 = axes[0, 2]
        unique_counts = df.nunique()
        bars = ax3.bar(unique_counts.index, unique_counts.values, 
                      color='lightblue', alpha=0.8)
        ax3.set_title('Unique Values per Column', fontweight='bold')
        ax3.set_ylabel('Unique Count')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 4. Outlier detection for numerical features
        ax4 = axes[1, 0]
        numerical_cols = ['Age', 'Na_to_K']
        outlier_counts = []
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_counts.append(len(outliers))
        
        bars = ax4.bar(numerical_cols, outlier_counts, color='orange', alpha=0.7)
        ax4.set_title('Outliers in Numerical Features', fontweight='bold')
        ax4.set_ylabel('Outlier Count')
        
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 5. Class balance analysis
        ax5 = axes[1, 1]
        class_counts = df['Drug'].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        bars = ax5.bar(class_counts.index, class_counts.values, 
                      color=plt.cm.RdYlBu(class_counts.values / max_count))
        ax5.set_title(f'Class Balance\n(Ratio: {imbalance_ratio:.2f})', fontweight='bold')
        ax5.set_ylabel('Count')
        ax5.tick_params(axis='x', rotation=45)
        
        # Add balance assessment
        if imbalance_ratio < 2:
            balance_status = "Well Balanced ✅"
            color = "lightgreen"
        elif imbalance_ratio < 5:
            balance_status = "Moderately Imbalanced ⚠️"
            color = "orange"
        else:
            balance_status = "Highly Imbalanced ❌"
            color = "red"
        
        ax5.text(0.5, 0.95, balance_status, transform=ax5.transAxes,
                horizontalalignment='center', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        # 6. Data summary statistics
        ax6 = axes[1, 2]
        summary_text = f"""
Dataset Quality Summary:
• Total Samples: {len(df)}
• Total Features: {len(df.columns) - 1}
• Missing Values: {df.isnull().sum().sum()}
• Duplicate Rows: {df.duplicated().sum()}
• Numerical Features: {len(numerical_cols)}
• Categorical Features: {len(df.select_dtypes(include='object').columns) - 1}
• Class Imbalance Ratio: {imbalance_ratio:.2f}
• Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB
        """
        
        ax6.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                transform=ax6.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax6.set_title('Data Summary', fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🔍 Data quality report saved to: {save_path}")
        
        plt.show()
    
    def generate_complete_data_report(self, df, output_dir="results/plots/data_exploration/"):
        """
        Generate a complete data exploration report with all visualizations.
        
        Args:
            df (pandas.DataFrame): The dataset
            output_dir (str): Directory to save all plots
        """
        print(f"\n{'='*60}")
        print(f"📊 GENERATING COMPLETE DATA EXPLORATION REPORT")
        print(f"{'='*60}")
        
        ensure_dir(output_dir)
        
        # Create all visualizations
        print("\n1. Creating overview dashboard...")
        self.create_overview_dashboard(df, os.path.join(output_dir, "01_overview_dashboard.png"))
        
        print("\n2. Creating feature distributions...")
        self.plot_feature_distributions(df, os.path.join(output_dir, "02_feature_distributions.png"))
        
        print("\n3. Creating target analysis...")
        self.plot_target_analysis(df, os.path.join(output_dir, "03_target_analysis.png"))
        
        print("\n4. Creating correlation analysis...")
        self.plot_correlation_analysis(df, os.path.join(output_dir, "04_correlation_analysis.png"))
        
        print("\n5. Creating data quality report...")
        self.plot_data_quality_report(df, os.path.join(output_dir, "05_data_quality_report.png"))
        
        print("\n6. Creating interactive dashboard...")
        self.create_interactive_dashboard(df, os.path.join(output_dir, "06_interactive_dashboard.html"))
        
        print(f"\n✅ Complete data exploration report generated!")
        print(f"📁 All files saved to: {output_dir}")
        
        # Generate summary report
        self._generate_data_summary_report(df, os.path.join(output_dir, "data_exploration_summary.md"))
    
    def _generate_data_summary_report(self, df, save_path):
        """Generate a markdown summary report of the data exploration."""
        
        # Calculate key statistics
        drug_counts = df['Drug'].value_counts()
        imbalance_ratio = drug_counts.max() / drug_counts.min()
        
        numerical_cols = ['Age', 'Na_to_K']
        outlier_info = {}
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_info[col] = len(outliers)
        
        report_content = f"""# Data Exploration Summary Report

## Dataset Overview
- **Total Samples:** {len(df)}
- **Total Features:** {len(df.columns) - 1}
- **Target Variable:** Drug (5 classes)
- **Missing Values:** {df.isnull().sum().sum()}
- **Duplicate Rows:** {df.duplicated().sum()}

## Feature Summary
### Numerical Features
- **Age:** Range [{df['Age'].min()}, {df['Age'].max()}], Mean: {df['Age'].mean():.1f}, Std: {df['Age'].std():.1f}
- **Na_to_K:** Range [{df['Na_to_K'].min():.2f}, {df['Na_to_K'].max():.2f}], Mean: {df['Na_to_K'].mean():.2f}, Std: {df['Na_to_K'].std():.2f}

### Categorical Features
- **Sex:** {dict(df['Sex'].value_counts())}
- **Blood Pressure:** {dict(df['BP'].value_counts())}
- **Cholesterol:** {dict(df['Cholesterol'].value_counts())}

## Target Variable Analysis
### Drug Distribution
{drug_counts.to_string()}

**Class Balance Assessment:**
- Imbalance Ratio: {imbalance_ratio:.2f}
- Status: {"Well Balanced" if imbalance_ratio < 2 else "Moderately Imbalanced" if imbalance_ratio < 5 else "Highly Imbalanced"}

## Data Quality Assessment
### Outliers Detected
{chr(10).join([f"- {col}: {count} outliers" for col, count in outlier_info.items()])}

### Data Quality Score
- Missing Values: {"✅ None" if df.isnull().sum().sum() == 0 else f"❌ {df.isnull().sum().sum()} found"}
- Duplicates: {"✅ None" if df.duplicated().sum() == 0 else f"❌ {df.duplicated().sum()} found"}
- Outliers: {"✅ Minimal" if sum(outlier_info.values()) < len(df) * 0.05 else f"⚠️ {sum(outlier_info.values())} found"}
- Class Balance: {"✅ Good" if imbalance_ratio < 2 else "⚠️ Imbalanced" if imbalance_ratio < 5 else "❌ Highly Imbalanced"}

## Key Insights
1. **Dataset Size:** The dataset contains {len(df)} samples, which is sufficient for machine learning but not very large.
2. **Feature Quality:** {"All features are complete with no missing values." if df.isnull().sum().sum() == 0 else "Some missing values detected that need attention."}
3. **Target Distribution:** {"Classes are well balanced." if imbalance_ratio < 2 else f"Classes are imbalanced with ratio {imbalance_ratio:.2f}."}
4. **Feature Relationships:** Age and Na_to_K ratio appear to be important predictors based on their distributions across drug classes.

## Recommendations for Modeling
1. **Data Preprocessing:** {"Standard scaling recommended for numerical features." if len(numerical_cols) > 0 else "Focus on categorical encoding."}
2. **Class Imbalance:** {"No special handling needed." if imbalance_ratio < 2 else "Consider techniques like SMOTE or class weights."}
3. **Feature Engineering:** Consider creating interaction features between categorical variables.
4. **Model Selection:** The dataset size and feature types suggest tree-based models and neural networks may work well.

## Generated Visualizations
1. `01_overview_dashboard.png` - Comprehensive dataset overview
2. `02_feature_distributions.png` - Individual feature distributions
3. `03_target_analysis.png` - Target variable analysis
4. `04_correlation_analysis.png` - Feature correlation analysis
5. `05_data_quality_report.png` - Data quality assessment
6. `06_interactive_dashboard.html` - Interactive exploration dashboard
"""
        
        with open(save_path, 'w') as f:
            f.write(report_content)
        
        print(f"📄 Data exploration summary saved to: {save_path}")