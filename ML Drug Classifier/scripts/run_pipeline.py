"""
Complete pipeline script that runs the entire drug classification project.

This script orchestrates the full workflow:
1. Data loading and preprocessing
2. Model training
3. Model evaluation
4. Visualization generation

Run this script to execute the complete project pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import subprocess
import time
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        
        duration = time.time() - start_time
        print(f"✅ {description} completed successfully!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        
        # Print any output
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_prerequisites():
    """Check if all required files and directories exist."""
    print("🔍 CHECKING PREREQUISITES")
    print("="*40)
    
    required_files = [
        'data/raw/drug200.csv',
        'config.yaml',
        'requirements.txt'
    ]
    
    required_dirs = [
        'src',
        'scripts'
    ]
    
    missing_items = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_items.append(f"Missing file: {file_path}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_items.append(f"Missing directory: {dir_path}")
    
    if missing_items:
        print("❌ Prerequisites not met:")
        for item in missing_items:
            print(f"   {item}")
        return False
    
    print("✅ All prerequisites met!")
    return True

def create_directory_structure():
    """Create all necessary directories."""
    print("\n📁 CREATING DIRECTORY STRUCTURE")
    print("="*40)
    
    directories = [
        'data/processed',
        'models/saved_models',
        'models/hyperparameters',
        'results/plots/data_exploration',
        'results/plots/model_performance',
        'results/plots/feature_importance',
        'results/metrics',
        'results/reports',
        'results/training',
        'results/evaluation'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print("📁 Directory structure ready!")

def main():
    """Run the complete pipeline."""
    print("🚀 DRUG CLASSIFICATION PROJECT - COMPLETE PIPELINE")
    print("="*80)
    print("This pipeline will:")
    print("1. Check prerequisites")
    print("2. Set up directory structure")
    print("3. Train all models")
    print("4. Evaluate model performance")
    print("5. Generate comprehensive visualizations")
    print("="*80)
    
    # Record start time
    pipeline_start = time.time()
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Pipeline aborted due to missing prerequisites")
        return False
    
    # Step 2: Create directory structure
    create_directory_structure()
    
    # Step 3: Run training script
    success = run_script('scripts/train_models.py', 'MODEL TRAINING')
    if not success:
        print("\n❌ Pipeline aborted due to training failure")
        return False
    
    # Step 4: Run evaluation script
    success = run_script('scripts/evaluate_models.py', 'MODEL EVALUATION')
    if not success:
        print("\n❌ Pipeline aborted due to evaluation failure")
        return False
    
    # Step 5: Run visualization script
    success = run_script('scripts/generate_plots.py', 'VISUALIZATION GENERATION')
    if not success:
        print("\n⚠️  Visualization generation failed, but core pipeline completed")
    
    # Calculate total time
    total_duration = time.time() - pipeline_start
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"⏱️  Total Duration: {total_duration/60:.2f} minutes")
    print(f"📁 Results Directory: results/")
    print(f"🏆 Model Files: models/saved_models/")
    print(f"📊 Visualizations: results/plots/")
    print(f"📄 Reports: results/reports/")
    
    print(f"\n📋 NEXT STEPS:")
    print("1. Review results in results/reports/final_project_summary.md")
    print("2. Explore interactive dashboards in results/plots/")
    print("3. Check model performance in results/evaluation/")
    print("4. Use best model for drug classification predictions")
    
    return True

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)