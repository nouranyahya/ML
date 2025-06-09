import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from datetime import datetime
import numpy as np
from .base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    """Neural Network classifier implementation using TensorFlow/Keras."""
    
    def __init__(self, hidden_layers=[64, 32, 16], activation='relu', 
                 dropout_rate=0.3, learning_rate=0.001, epochs=100, batch_size=32):
        """
        Initialize Neural Network model.
        
        Parameters:
        - hidden_layers: List of hidden layer sizes
        - activation: Activation function for hidden layers
        - dropout_rate: Dropout rate for regularization
        - learning_rate: Learning rate for optimizer
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        """
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.model_name = "Neural Network"
        self.input_features = None
        self.num_classes = None
        
    def create_model(self, input_features, num_classes):
        """Create and return a neural network model."""
        self.input_features = input_features
        self.num_classes = num_classes
        
        print(f"🔧 Creating Neural Network with architecture:")
        print(f"   Input features: {input_features}")
        print(f"   Hidden layers: {self.hidden_layers}")
        print(f"   Activation: {self.activation}")
        print(f"   Dropout rate: {self.dropout_rate}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Output classes: {num_classes}")
        
        # Create the model
        model = keras.Sequential()
        
        # Add input layer and first hidden layer
        model.add(keras.layers.Dense(
            self.hidden_layers[0], 
            activation=self.activation, 
            input_shape=(input_features,),
            name='hidden_1'
        ))
        model.add(keras.layers.Dropout(self.dropout_rate))
        
        # Add remaining hidden layers
        for i, layer_size in enumerate(self.hidden_layers[1:], 2):
            model.add(keras.layers.Dense(
                layer_size, 
                activation=self.activation,
                name=f'hidden_{i}'
            ))
            model.add(keras.layers.Dropout(self.dropout_rate))
        
        # Add output layer
        model.add(keras.layers.Dense(num_classes, activation='softmax', name='output'))
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n🏗️  Model Architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the neural network model.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - X_val: Validation features (optional)
        - y_val: Validation labels (optional)
        """
        if self.model is None:
            input_features = X_train.shape[1]
            num_classes = len(set(y_train))
            self.create_model(input_features, num_classes)
            
        print(f"🚀 Training Neural Network...")
        print(f"   Training data shape: {X_train.shape}")
        print(f"   Number of classes: {len(set(y_train))}")
        print(f"   Epochs: {self.epochs}")
        print(f"   Batch size: {self.batch_size}")
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Set up callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate final accuracies
        train_pred = self.model.predict(X_train, verbose=0)
        train_pred_classes = np.argmax(train_pred, axis=1)
        train_accuracy = accuracy_score(y_train, train_pred_classes)
        
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val, verbose=0)
            val_pred_classes = np.argmax(val_pred, axis=1)
            val_accuracy = accuracy_score(y_val, val_pred_classes)
        
        print(f"✅ Training completed!")
        print(f"   Epochs trained: {len(history.history['loss'])}")
        print(f"   Final training accuracy: {train_accuracy:.4f}")
        if val_accuracy is not None:
            print(f"   Final validation accuracy: {val_accuracy:.4f}")
        
        # Store training information (convert numpy arrays to lists for JSON serialization)
        self.training_info = {
            'train_accuracy': float(train_accuracy),
            'val_accuracy': float(val_accuracy) if val_accuracy is not None else None,
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'input_features': self.input_features,
            'num_classes': self.num_classes,
            # Convert history to lists for JSON serialization
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history.get('accuracy', [])],
                'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])]
            }
        }
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'model': self.model,
            'history': history
        }
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X, verbose=0)
    
    def save(self, filepath):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        # Create timestamped filename with .keras extension
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"neural_network_{timestamp}.keras"
        full_path = f"{filepath}/{model_filename}"
        
        # Ensure directory exists
        import os
        os.makedirs(filepath, exist_ok=True)
        
        # Save the model in native Keras format
        self.model.save(full_path)
        print(f"🧠 Neural Network model saved to: {full_path}")
        
        return full_path
    
    def load(self, filepath):
        """Load a trained model."""
        self.model = keras.models.load_model(filepath)
        return self.model
    
    def get_params(self):
        """Get model parameters."""
        return {
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }