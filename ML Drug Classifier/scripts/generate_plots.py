"""
Main script to generate all visualizations for the drug classification project.

This script creates comprehensive visualizations for data exploration
and model results analysis.
"""
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from pathlib import Path

from src.data.load_data import load_drug_data
from src.data.preprocess import preprocess_drug_data
from src.visualization.data_plots import DataVisualizer
from src.visualization.results_plots import ResultsVisualizer
from src.utils.config import config
from src.utils.helpers import ensure_dir

def load_evaluation_results():
    """Load evaluation results from disk."""
    results_dir = Path('results/evaluation')
    
    if not results_dir.exists():
        print("❌ No evaluation results found! Please run evaluate_models.py first.")
        return []
    
    evaluation_results = []
    result_files = list(results_dir.glob('*_evaluation.json'))
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                evaluation_results.append(result)
            print(f"✅ Loaded results for {result['model_name']}")
        except Exception as e:
            print(f"⚠️  Could not load {result_file.name}: {str(e)}")
    
    return evaluation_results

def main():
    """Main visualization generation pipeline."""
    print("📊 DRUG CLASSIFICATION - VISUALIZATION GENERATION")
    print("="*60)
    
    # 1. Load data
    print("\n📥 STEP 1: LOADING DATA")
    print("-"*40)
    
    raw_data_path = config.get('data.raw_data', 'data/raw/drug200.csv')
    df = load_drug_data(raw_data_path)
    preprocessed_data = preprocess_drug_data(df)
    
    X_test = preprocessed_data['X_test']
    y_test = preprocessed_data['y_test']
    target_classes = preprocessed_data['target_classes']
    
    print(f"✅ Data loaded successfully")
    
    # 2. Generate data exploration visualizations
    print("\n🔍 STEP 2: DATA EXPLORATION VISUALIZATIONS")
    print("-"*40)
    
    data_visualizer = DataVisualizer()
    
    print("Creating comprehensive data exploration report...")
    data_visualizer.generate_complete_data_report(df)
    
    print("✅ Data exploration visualizations completed!")
    
    # 3. Load evaluation results
    print("\n📊 STEP 3: LOADING EVALUATION RESULTS")
    print("-"*40)
    
    evaluation_results = load_evaluation_results()
    
    if not evaluation_results:
        print("⚠️  No evaluation results found. Skipping results visualizations.")
        print("Please run evaluate_models.py first.")
        return
    
    print(f"✅ Loaded results for {len(evaluation_results)} models")
    
    # 4. Generate results visualizations
    print("\n🏆 STEP 4: RESULTS VISUALIZATIONS")
    print("-"*40)
    
    results_visualizer = ResultsVisualizer()
    
    print("Creating comprehensive results visualization report...")
    results_visualizer.generate_complete_results_report(
        evaluation_results,
        None,  # models_dict not available here
        X_test,
        y_test,
        target_classes
    )
    
    print("✅ Results visualizations completed!")
    
    # 5. Generate summary report
    print("\n📄 STEP 5: GENERATING FINAL SUMMARY")
    print("-"*40)
    
    # Create final project summary
    ensure_dir('results/reports/')
    
    best_model = max(evaluation_results, key=lambda x: x['accuracy'])
    
    final_summary = f"""# Drug Classification Project - Final Summary

## Project Overview
This project implements and compares multiple machine learning models for drug classification
based on patient characteristics including age, sex, blood pressure, cholesterol levels, and Na/K ratio.

## Dataset Summary
- **Total Samples:** {len(df)}
- **Features:** Age, Sex, BP, Cholesterol, Na_to_K
- **Target Classes:** {len(target_classes)} drug types
- **Training/Validation/Test Split:** {len(preprocessed_data['X_train'])}/{len(preprocessed_data['X_val'])}/{len(X_test)}

## Models Evaluated
{chr(10).join([f"- {result['model_name']}: {result['accuracy']:.4f} accuracy" for result in evaluation_results])}

## Best Performing Model
**{best_model['model_name']}**
- Accuracy: {best_model['accuracy']:.4f}
- Precision: {best_model['precision_macro']:.4f}
- Recall: {best_model['recall_macro']:.4f}
- F1-Score: {best_model['f1_macro']:.4f}

## Key Findings
1. {best_model['model_name']} achieved the best overall performance
2. Average model accuracy: {np.mean([r['accuracy'] for r in evaluation_results]):.4f}
3. All models showed {('good' if min(r['accuracy'] for r in evaluation_results) > 0.8 else 'varying')} performance

## Generated Outputs
### Data Exploration
- Overview dashboard and feature analysis
- Target variable analysis and correlations
- Data quality assessment report

### Model Results
- Performance comparison charts
- Confusion matrices for all models
- ROC curves and feature importance analysis
- Interactive dashboards for detailed exploration

## Recommendations
1. **Production Model:** Use {best_model['model_name']} for drug classification
2. **Monitoring:** Track model performance on new patient data
3. **Updates:** Retrain models when new data becomes available
4. **Validation:** Conduct clinical validation before deployment

## Files Generated
- `/results/plots/data_exploration/` - Data analysis visualizations
- `/results/plots/model_performance/` - Model comparison plots
- `/results/evaluation/` - Detailed evaluation metrics
- `/results/reports/` - Comprehensive analysis reports
"""
    
    with open('results/reports/final_project_summary.md', 'w') as f:
        f.write(final_summary)
    
    print("✅ Visualization pipeline completed!")
    print(f"📁 All visualizations saved to: results/plots/")
    print(f"📄 Final summary: results/reports/final_project_summary.md")
    
    print(f"\n🎉 PROJECT COMPLETE!")
    print("="*60)
    print("📊 Data exploration visualizations created")
    print("🏆 Model performance analysis completed")
    print("📈 Interactive dashboards generated")
    print("📄 Comprehensive reports available")
    
    return evaluation_results

if __name__ == "__main__":
    main()