"""
Training script for all machine learning models.
This script loads preprocessed data and trains all models.
"""

import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_processed_data
from src.models.logistic_regression import LogisticRegressionModel
from src.models.knn import KNNModel
from src.models.svm import SVMModel
from src.models.random_forest import RandomForestModel
from src.models.neural_network import NeuralNetworkModel
from src.utils.helpers import save_json, print_section_header, print_step_header

def main():
    print_section_header("DRUG CLASSIFICATION - MODEL TRAINING PIPELINE")
    
    # Step 1: Load preprocessed data
    print_step_header("STEP 1", "LOADING AND PREPROCESSING DATA")
    try:
        # Load preprocessed data
        train_data, val_data, test_data = load_processed_data()
        
        # Extract features and labels
        X_train = train_data.drop(['Drug'], axis=1)
        y_train = train_data['Drug']
        X_val = val_data.drop(['Drug'], axis=1)
        y_val = val_data['Drug']
        X_test = test_data.drop(['Drug'], axis=1)
        y_test = test_data['Drug']
        
        print(f"✅ Data preprocessing completed!")
        print(f"   Training set: {X_train.shape}")
        print(f"   Validation set: {X_val.shape}")
        print(f"   Test set: {X_test.shape}")
        
    except Exception as e:
        print(f"❌ Error loading preprocessed data: {e}")
        return
    
    # Step 2: Initialize models
    print_step_header("STEP 2", "INITIALIZING MODELS")
    models = {
        'Logistic Regression': LogisticRegressionModel(),
        'K-Nearest Neighbors': KNNModel(),
        'Support Vector Machine': SVMModel(),
        'Random Forest': RandomForestModel(),
        'Neural Network': NeuralNetworkModel()
    }
    print(f"✅ Initialized {len(models)} models")
    
    # Step 3: Train models
    print_step_header("STEP 3", "TRAINING MODELS")
    
    training_results = {}
    successful_models = {}
    
    for model_name, model in models.items():
        print(f"\n🔄 Training {model_name}...")
        try:
            # Train the model
            result = model.train(X_train, y_train, X_val, y_val)
            
            # Save the model - FIX: Use correct base path
            model_save_path = model.save("models/saved_models")
            
            # Store results
            training_results[model_name] = {
                'train_accuracy': float(result['train_accuracy']),
                'val_accuracy': float(result['val_accuracy']) if result['val_accuracy'] is not None else None,
                'model_path': model_save_path,
                'training_info': getattr(model, 'training_info', {})
            }
            
            successful_models[model_name] = model
            print(f"✅ {model_name} trained successfully!")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            training_results[model_name] = {
                'error': str(e),
                'train_accuracy': None,
                'val_accuracy': None,
                'model_path': None
            }
    
    # Step 4: Save training summary
    print_step_header("STEP 4", "SAVING TRAINING SUMMARY")
    
    # Create training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_trained': len(successful_models),
        'total_models': len(models),
        'training_data_shape': list(X_train.shape),
        'validation_data_shape': list(X_val.shape),
        'test_data_shape': list(X_test.shape),
        'results': training_results
    }
    
    # Save summary
    summary_path = f"results/training/training_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        save_json(summary, summary_path)
        print(f"✅ Training summary saved to: {summary_path}")
    except Exception as e:
        print(f"❌ Error saving training summary: {e}")
    
    # Print final summary
    print_section_header("TRAINING COMPLETE")
    print(f"✅ Successfully trained: {len(successful_models)}/{len(models)} models")
    
    if successful_models:
        print("\n📊 Training Results:")
        for model_name in successful_models:
            result = training_results[model_name]
            train_acc = result['train_accuracy']
            val_acc = result['val_accuracy']
            print(f"   {model_name}:")
            print(f"     Training Accuracy: {train_acc:.4f}" if train_acc else "     Training Accuracy: N/A")
            print(f"     Validation Accuracy: {val_acc:.4f}" if val_acc else "     Validation Accuracy: N/A")
    
    if len(successful_models) < len(models):
        print(f"\n⚠️  {len(models) - len(successful_models)} models failed to train")
        failed_models = set(models.keys()) - set(successful_models.keys())
        for model_name in failed_models:
            error = training_results[model_name].get('error', 'Unknown error')
            print(f"   {model_name}: {error}")
    
    return successful_models, training_results

if __name__ == "__main__":
    main()