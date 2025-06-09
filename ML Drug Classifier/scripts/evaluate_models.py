"""
Main script to evaluate all trained models for drug classification.

This script loads trained models and evaluates their performance.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import joblib
import numpy as np
from pathlib import Path

from src.data.load_data import load_drug_data
from src.data.preprocess import preprocess_drug_data
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.model_comparison import ModelComparison
from src.models.neural_network import NeuralNetworkModel
from src.utils.config import config
from src.utils.helpers import ensure_dir, save_json, get_timestamp

def load_trained_models():
    """Load all trained models from disk."""
    models_dir = Path('models/saved_models')
    
    if not models_dir.exists():
        print("❌ No saved models found! Please run train_models.py first.")
        return {}
    
    loaded_models = {}
    model_files = list(models_dir.glob('*.pkl'))
    
    # Load traditional ML models (pickle files)
    for model_file in model_files:
        try:
            if 'neural_network' not in model_file.name:
                model = joblib.load(model_file)
                model_name = getattr(model, 'model_name', model_file.stem)
                loaded_models[model_name] = model
                print(f"✅ Loaded {model_name}")
        except Exception as e:
            print(f"⚠️  Could not load {model_file.name}: {str(e)}")
    
    # Load neural network models (HDF5 files)
    nn_files = list(models_dir.glob('*.h5'))
    for nn_file in nn_files:
        try:
            model = NeuralNetworkModel.load_model(nn_file)
            loaded_models[model.model_name] = model
            print(f"✅ Loaded {model.model_name}")
        except Exception as e:
            print(f"⚠️  Could not load {nn_file.name}: {str(e)}")
    
    return loaded_models

def main():
    """Main evaluation pipeline."""
    print("📊 DRUG CLASSIFICATION - MODEL EVALUATION PIPELINE")
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
    
    print(f"✅ Test data loaded: {X_test.shape}")
    
    # 2. Load trained models
    print("\n🤖 STEP 2: LOADING TRAINED MODELS")
    print("-"*40)
    
    models = load_trained_models()
    
    if not models:
        print("❌ No models found! Please run train_models.py first.")
        return
    
    print(f"✅ Loaded {len(models)} models")
    
    # 3. Evaluate each model
    print("\n🔍 STEP 3: EVALUATING MODELS")
    print("-"*40)
    
    evaluator = ModelEvaluator()
    evaluation_results = []
    
    for model_name, model in models.items():
        print(f"\n📊 Evaluating {model_name}...")
        
        try:
            result = evaluator.evaluate_model(
                model, X_test, y_test, model_name, target_classes
            )
            evaluation_results.append(result)
            
            print(f"✅ {model_name} evaluation completed!")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            continue
    
    # 4. Compare models
    print("\n🏆 STEP 4: COMPARING MODELS")
    print("-"*40)
    
    if len(evaluation_results) >= 2:
        comparator = ModelComparison()
        
        # Add models to comparator
        for model_name, model in models.items():
            comparator.add_model(model, model_name)
        
        # Perform comparison
        comparison_results = comparator.compare_models(
            X_test, y_test, target_classes, save_results=True
        )
        
        # Generate recommendation
        recommendation = comparator.generate_model_recommendation(comparison_results)
        comparator.print_recommendation(recommendation)
        
        # Export comprehensive report
        report_path = comparator.export_comparison_report(comparison_results)
        
    else:
        print("⚠️  Need at least 2 models for comparison")
        comparison_results = None
        recommendation = None
    
    # 5. Save evaluation results
    print("\n💾 STEP 5: SAVING EVALUATION RESULTS")
    print("-"*40)
    
    ensure_dir('results/evaluation/')
    
    # Save individual results
    for result in evaluation_results:
        model_name_clean = result['model_name'].replace(' ', '_').lower()
        result_path = f'results/evaluation/{model_name_clean}_evaluation.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            else:
                serializable_result[key] = value
        
        save_json(serializable_result, result_path)
    
    # Save summary
    summary = {
        'timestamp': get_timestamp(),
        'models_evaluated': [r['model_name'] for r in evaluation_results],
        'test_set_size': len(X_test),
        'best_model': max(evaluation_results, key=lambda x: x['accuracy'])['model_name'],
        'average_accuracy': np.mean([r['accuracy'] for r in evaluation_results]),
        'evaluation_summary': {
            r['model_name']: {
                'accuracy': r['accuracy'],
                'f1_macro': r['f1_macro'],
                'precision_macro': r['precision_macro'],
                'recall_macro': r['recall_macro']
            }
            for r in evaluation_results
        }
    }
    
    summary_path = 'results/evaluation/evaluation_summary.json'
    save_json(summary, summary_path)
    
    print(f"✅ Evaluation pipeline completed!")
    print(f"📁 Results saved to: results/evaluation/")
    print(f"🏆 Best model: {summary['best_model']}")
    print(f"📊 Average accuracy: {summary['average_accuracy']:.4f}")
    
    print("\n📋 Next steps:")
    print("   1. Run 'python scripts/generate_plots.py' to create visualizations")
    print("   2. Check 'results/reports/' for detailed analysis")
    
    return evaluation_results, comparison_results

if __name__ == "__main__":
    main()