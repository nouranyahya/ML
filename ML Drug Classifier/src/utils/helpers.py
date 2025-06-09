import json
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict

def create_directories(directories):
    """
    Create directories if they don't exist.
    
    Parameters:
    - directories: List of directory paths to create
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created: {directory}")
        else:
            print(f"📁 Exists: {directory}")

def get_timestamp():
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def convert_to_serializable(obj):
    """
    Convert numpy arrays and other non-serializable objects to JSON-serializable format.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def save_json(data, filepath):
    """
    Save data to JSON file with proper serialization handling.
    
    Parameters:
    - data: Data to save
    - filepath: Path to save the JSON file
    """
    # Convert data to serializable format
    serializable_data = convert_to_serializable(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)

def load_json(filepath):
    """
    Load data from JSON file.
    
    Parameters:
    - filepath: Path to the JSON file
    
    Returns:
    - Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def save_pickle(data, filepath):
    """
    Save data using pickle.
    
    Parameters:
    - data: Data to save
    - filepath: Path to save the pickle file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath):
    """
    Load data using pickle.
    
    Parameters:
    - filepath: Path to the pickle file
    
    Returns:
    - Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def print_section_header(title, char="=", length=60):
    """
    Print a formatted section header.
    
    Parameters:
    - title: Title of the section
    - char: Character to use for the line
    - length: Length of the line
    """
    print("\n" + char * length)
    print(f"🚀 {title}")
    print(char * length)

def print_step_header(step_num, title, char="-", length=40):
    """
    Print a formatted step header.
    
    Parameters:
    - step_num: Step number
    - title: Title of the step
    - char: Character to use for the line
    - length: Length of the line
    """
    print(f"\n{step_num}. {title}")
    print(char * length)

def format_accuracy(accuracy):
    """
    Format accuracy for display.
    
    Parameters:
    - accuracy: Accuracy value (0-1)
    
    Returns:
    - Formatted accuracy string
    """
    if accuracy is None:
        return "N/A"
    return f"{accuracy:.4f}"

def format_percentage(value):
    """
    Format value as percentage.
    
    Parameters:
    - value: Value (0-1)
    
    Returns:
    - Formatted percentage string
    """
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"

def safe_divide(numerator, denominator):
    """
    Safely divide two numbers, returning 0 if denominator is 0.
    
    Parameters:
    - numerator: Numerator
    - denominator: Denominator
    
    Returns:
    - Division result or 0 if denominator is 0
    """
    if denominator == 0:
        return 0
    return numerator / denominator

def validate_file_exists(filepath):
    """
    Validate that a file exists.
    
    Parameters:
    - filepath: Path to the file
    
    Returns:
    - True if file exists, False otherwise
    """
    return os.path.exists(filepath)

def get_file_size(filepath):
    """
    Get file size in bytes.
    
    Parameters:
    - filepath: Path to the file
    
    Returns:
    - File size in bytes
    """
    if validate_file_exists(filepath):
        return os.path.getsize(filepath)
    return 0

def format_file_size(size_bytes):
    """
    Format file size in human readable format.
    
    Parameters:
    - size_bytes: Size in bytes
    
    Returns:
    - Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def clean_filename(filename):
    """
    Clean filename by removing invalid characters.
    
    Parameters:
    - filename: Original filename
    
    Returns:
    - Cleaned filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def ensure_directory_exists(directory):
    """
    Ensure a directory exists, create if it doesn't.
    
    Parameters:
    - directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✅ Created directory: {directory}")

def log_error(error_message, log_file="error.log"):
    """
    Log error message to file.
    
    Parameters:
    - error_message: Error message to log
    - log_file: Log file path
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] ERROR: {error_message}\n")

def log_info(info_message, log_file="info.log"):
    """
    Log info message to file.
    
    Parameters:
    - info_message: Info message to log
    - log_file: Log file path
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] INFO: {info_message}\n")

# Additional functions that are imported by other modules
def ensure_dir(directory):
    """
    Ensure a directory exists, create if it doesn't.
    (Alias for ensure_directory_exists for backward compatibility)
    
    Parameters:
    - directory: Directory path
    """
    ensure_directory_exists(directory)

def save_model(model, filepath):
    """
    Save a model using pickle.
    
    Parameters:
    - model: Model to save
    - filepath: Path to save the model
    """
    save_pickle(model, filepath)

def load_model(filepath):
    """
    Load a model using pickle.
    
    Parameters:
    - filepath: Path to the model file
    
    Returns:
    - Loaded model
    """
    return load_pickle(filepath)

def print_dataset_info(data, title="DATASET INFORMATION"):
    """
    Print comprehensive information about a dataset.
    
    Parameters:
    - data: pandas DataFrame
    - title: Title for the information display
    """
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)
    
    print(f"📊 Shape: {data.shape} (rows × columns)")
    print(f"📋 Columns: {list(data.columns)}")
    
    print(f"\n🔍 Data Types:")
    for col, dtype in data.dtypes.items():
        print(f"   {col}: {dtype}")
    
    print(f"\n❓ Missing Values:")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values found ✅")
    else:
        for col, count in missing.items():
            if count > 0:
                print(f"   {col}: {count} ({count/len(data)*100:.1f}%)")
    
    print(f"\n📈 Statistical Summary:")
    print(data.describe(include='all'))

def calculate_class_distribution(target_column):
    """
    Calculate and display the distribution of target classes.
    
    Parameters:
    - target_column: pandas Series containing target values
    """
    print(f"\n🎯 Target Variable Distribution:")
    value_counts = target_column.value_counts()
    total = len(target_column)
    
    for class_name, count in value_counts.items():
        percentage = (count / total) * 100
        print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
    
    print(f"   Total: {total} samples")

def setup_plotting_style():
    """
    Set up a consistent plotting style for visualizations.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        
        print("✅ Plotting style configured")
    except ImportError:
        print("⚠️  Matplotlib/Seaborn not available for plotting style setup")

def get_timestamp():
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")