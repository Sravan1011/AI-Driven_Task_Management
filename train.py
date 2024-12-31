import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tf_keras import layers, regularizers
import joblib
import os
import json
import numpy as np

PRIORITY_CONFIG = {
    'prediction_threshold': 0.7,  # Higher threshold for "High" priority
    'importance_weights': {
        'Critical': 1.0,
        'High': 0.8,
        'Medium': 0.4
    },
    'workload_impact': {
        'confidence_boost': 0.1,  # Boost confidence for high workload
        'threshold_adjust': 0.05  # Adjust threshold based on workload
    }
}
class PriorityPredictor:
    def __init__(self, model_dir='model_output'):
        """Initialize predictor with synchronized file checking"""
        try:
            if not os.path.exists(model_dir):
                raise Exception(f"Model directory not found: {model_dir}")
            
            required_files = [
                'model_weights.h5',
                'vectorizer.pkl',
                'text_scaler.pkl',
                'meta_scaler.pkl',
                'model_data.pkl'
            ]
            
            missing_files = [f for f in required_files 
                           if not os.path.exists(os.path.join(model_dir, f))]
            
            if missing_files:
                raise Exception(f"Missing required files: {missing_files}")
            
            # Load model data and components
            self.model_data = joblib.load(os.path.join(model_dir, 'model_data.pkl'))
            self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
            self.text_scaler = joblib.load(os.path.join(model_dir, 'text_scaler.pkl'))
            self.meta_scaler = joblib.load(os.path.join(model_dir, 'meta_scaler.pkl'))
            
            # Create and load model
            self.model = create_model(
                self.model_data['text_input_dim'],
                self.model_data['meta_input_dim']
            )
            self.model.load_weights(os.path.join(model_dir, 'model_weights.h5'))
            
            # Initialize mappings
            self.time_map = {
                'Morning': 9 * 60,
                'Afternoon': 13 * 60,
                'Evening': 18 * 60,
                'Night': 21 * 60
            }
            
            # Get importance categories from training
            self.importance_categories = ['Critical', 'High', 'Medium']
            self.time_categories = list(self.time_map.keys())
            
            print("Model and components loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def predict_priority(self, task_data):
        """Predict priority for a task with fixed feature preparation"""
        try:
            # Prepare text features
            text = f"{task_data['title']}. {task_data['description']}"
            text_features = self.vectorizer.transform([text]).toarray()
            text_features_scaled = self.text_scaler.transform(text_features)
            
            # Create importance one-hot encoding
            importance_features = np.zeros(len(self.importance_categories))
            try:
                importance_idx = self.importance_categories.index(task_data['importance'])
                importance_features[importance_idx] = 1
            except ValueError:
                print(f"Warning: Unknown importance category: {task_data['importance']}")
                
            # Create time one-hot encoding
            time_features = np.zeros(len(self.time_categories))
            try:
                time_idx = self.time_categories.index(task_data['preferred_time'])
                time_features[time_idx] = 1
            except ValueError:
                print(f"Warning: Unknown time category: {task_data['preferred_time']}")
            
            # Combine meta features
            meta_features = np.concatenate([
                [task_data['estimated_time'] / 24.0],
                [task_data['workload']],
                importance_features,
                time_features
            ]).reshape(1, -1)
            
            # Scale features
            meta_features_scaled = self.meta_scaler.transform(meta_features)
            
            raw_score = self.model.predict(
                [text_features_scaled, meta_features_scaled],
                verbose=0
            )[0][0]
            
            # Adjust threshold based on importance and workload
            base_threshold = PRIORITY_CONFIG['prediction_threshold']
            importance_weight = PRIORITY_CONFIG['importance_weights'].get(
                task_data['importance'], 0.5
            )
            
            # Adjust threshold down for Critical/High importance
            adjusted_threshold = base_threshold - (importance_weight * 0.2)
            
            # Further adjust based on workload
            if task_data['workload'] > 0.7:  # High workload
                adjusted_threshold -= PRIORITY_CONFIG['workload_impact']['threshold_adjust']
            
            # Determine priority and confidence
            is_high_priority = raw_score > adjusted_threshold
            
            # Calculate confidence with importance weighting
            base_confidence = raw_score if is_high_priority else (1 - raw_score)
            weighted_confidence = base_confidence * importance_weight
            
            # Adjust confidence based on workload for high priority tasks
            if is_high_priority and task_data['workload'] > 0.7:
                weighted_confidence += PRIORITY_CONFIG['workload_impact']['confidence_boost']
            
            # Ensure confidence stays in [0,1] range
            final_confidence = min(max(weighted_confidence, 0.0), 1.0)
            
            return {
                'priority': 'High' if is_high_priority else 'Normal',
                'confidence': float(final_confidence),
                'raw_score': float(raw_score),
                'adjusted_threshold': float(adjusted_threshold),
                'features_used': {
                    'importance': task_data['importance'],
                    'time': task_data['preferred_time'],
                    'estimated_time': task_data['estimated_time'],
                    'workload': task_data['workload']
                }
            }
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            raise


def create_model(text_input_dim, meta_input_dim):
    """Create the model architecture"""
    # Text input branch
    text_input = layers.Input(shape=(text_input_dim,))
    text_features = layers.Dense(64, activation='relu', 
                               kernel_regularizer=regularizers.l2(0.01))(text_input)
    text_features = layers.BatchNormalization()(text_features)
    text_features = layers.Dropout(0.4)(text_features)
    
    # Metadata input branch
    meta_input = layers.Input(shape=(meta_input_dim,))
    meta_features = layers.Dense(32, activation='relu',
                               kernel_regularizer=regularizers.l2(0.01))(meta_input)
    meta_features = layers.BatchNormalization()(meta_features)
    meta_features = layers.Dropout(0.3)(meta_features)
    
    # Combine branches
    combined = layers.Concatenate()([text_features, meta_features])
    
    # Deep layers
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Skip connection
    skip = x
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Add()([x, skip])
    
    # Output
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[text_input, meta_input], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.AUC(), 
                tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def prepare_training_data(csv_file):
    """Prepare and clean the data for training"""
    print("Loading and preparing training data...")
    
    # Read CSV data
    df = pd.read_csv(csv_file)
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Create formatted dataset
    formatted_data = {
        'description': [],
        'importance': [],
        'estimated_time': [],
        'preferred_time': [],
        'workload': [],
        'priority': []
    }
    
    # Workload mapping
    workload_map = {
        'Overwhelmed': 1.0,
        'Heavy': 0.8,
        'Moderate': 0.5,
        'Light': 0.2
    }
    
    # Process each row
    for _, row in df.iterrows():
        # Combine title, description, and keywords
        description = f"{row['Task Title']}. {row['Task Description']}. Keywords: {row['Task Keywords and Description']}"
        
        # Convert time string to float
        try:
            time_str = row['Estimated Time of Completion(Hours)']
            if ':' in str(time_str):
                hours, minutes = map(float, str(time_str).split(':'))
                estimated_time = hours + minutes/60
            else:
                estimated_time = float(time_str)
        except:
            estimated_time = 0.0
        
        # Map workload to numerical value
        workload = workload_map.get(row['Current Workload'], 0.5)
        
        # Determine priority
        priority = 1 if row['Task Importance'].strip() in ['Critical', 'High'] else 0
        
        # Add to formatted data
        formatted_data['description'].append(description)
        formatted_data['importance'].append(row['Task Importance'].strip())
        formatted_data['estimated_time'].append(estimated_time)
        formatted_data['preferred_time'].append(row['Preferred Time of Day to work on Tasks'].strip())
        formatted_data['workload'].append(workload)
        formatted_data['priority'].append(priority)
    
    # Create DataFrame
    formatted_df = pd.DataFrame(formatted_data)
    formatted_df.to_csv('formatted_training_data.csv', index=False)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(formatted_df)}")
    print(f"High priority tasks: {formatted_df['priority'].sum()}")
    print(f"Normal priority tasks: {len(formatted_df) - formatted_df['priority'].sum()}")
    print("\nImportance distribution:")
    print(formatted_df['importance'].value_counts())
    print("\nPreferred time distribution:")
    print(formatted_df['preferred_time'].value_counts())
    print("\nEstimated time statistics:")
    print(formatted_df['estimated_time'].describe())
    
    return formatted_df

def save_trained_model(model, components, output_dir='model_output'):
    """Save model with category information"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        vectorizer, text_scaler, meta_scaler = components
        
        # Save components
        joblib.dump(vectorizer, os.path.join(output_dir, 'vectorizer.pkl'))
        joblib.dump(text_scaler, os.path.join(output_dir, 'text_scaler.pkl'))
        joblib.dump(meta_scaler, os.path.join(output_dir, 'meta_scaler.pkl'))
        
        # Save model weights
        model.save_weights(os.path.join(output_dir, 'model_weights.h5'))
        
        # Save model dimensions and info
        model_data = {
            'text_input_dim': vectorizer.get_feature_names_out().shape[0],
            'meta_input_dim': meta_scaler.n_features_in_,
            'importance_categories': ['Critical', 'High', 'Medium'],
            'time_categories': ['Morning', 'Afternoon', 'Evening', 'Night']
        }
        joblib.dump(model_data, os.path.join(output_dir, 'model_data.pkl'))
        
        print(f"Model and components saved successfully to {output_dir}/")
        return True
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False
def train_model(data_file, output_dir='model_output', n_splits=5):
    """Train the model using cross-validation"""
    try:
        print("Loading data...")
        df = pd.read_csv(data_file)
        
        # Prepare features
        X_text = df['description'].values
        df['estimated_time'] = df['estimated_time'] / df['estimated_time'].max()
        
        # Create meta features with fixed order
        importance_categories = ['Critical', 'High', 'Medium']
        time_categories = ['Morning', 'Afternoon', 'Evening', 'Night']
        
        # Create importance dummies with fixed order
        importance_dummies = np.zeros((len(df), len(importance_categories)))
        for i, importance in enumerate(importance_categories):
            importance_dummies[:, i] = (df['importance'] == importance).astype(float)
        
        # Create time dummies with fixed order
        time_dummies = np.zeros((len(df), len(time_categories)))
        for i, time in enumerate(time_categories):
            time_dummies[:, i] = (df['preferred_time'] == time).astype(float)
        
        # Combine meta features
        X_meta = np.hstack([
            df[['estimated_time']].values,
            df[['workload']].values,
            importance_dummies,
            time_dummies
        ])
        
        y = df['priority'].values
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = {'accuracy': [], 'auc': [], 'precision': [], 'recall': []}
        
        best_auc = 0
        best_fold_model = None
        best_fold_components = None
        
        # TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=100,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        print("\nStarting cross-validation training...")
        
        # Train on each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_text, y), 1):
            print(f"\nFold {fold}/{n_splits}")
            print("-" * 30)
            
            # Split data
            X_text_train, X_text_val = X_text[train_idx], X_text[val_idx]
            X_meta_train, X_meta_val = X_meta[train_idx], X_meta[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Process features
            print("Processing features...")
            X_text_train_tfidf = vectorizer.fit_transform(X_text_train).toarray()
            X_text_val_tfidf = vectorizer.transform(X_text_val).toarray()
            
            text_scaler = StandardScaler()
            meta_scaler = StandardScaler()
            
            X_text_train_scaled = text_scaler.fit_transform(X_text_train_tfidf)
            X_meta_train_scaled = meta_scaler.fit_transform(X_meta_train)
            
            X_text_val_scaled = text_scaler.transform(X_text_val_tfidf)
            X_meta_val_scaled = meta_scaler.transform(X_meta_val)
            
            # Create and train model
            print("Training model...")
            model = create_model(
                X_text_train_scaled.shape[1],
                X_meta_train_scaled.shape[1]
            )
            
            # Calculate class weights
            total_samples = len(y_train)
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            
            class_weights = {
                0: total_samples / (2 * n_neg) if n_neg > 0 else 1.0,
                1: total_samples / (2 * n_pos) if n_pos > 0 else 1.0
            }
            
            # Train
            history = model.fit(
                [X_text_train_scaled, X_meta_train_scaled],
                y_train,
                validation_data=([X_text_val_scaled, X_meta_val_scaled], y_val),
                epochs=30,
                batch_size=16,
                class_weight=class_weights,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_auc',
                        patience=5,
                        restore_best_weights=True,
                        mode='max'
                    )
                ],
                verbose=1
            )
            
            # Evaluate
            scores = model.evaluate(
                [X_text_val_scaled, X_meta_val_scaled],
                y_val,
                verbose=0
            )
            
            # Store scores
            cv_scores['accuracy'].append(scores[1])
            cv_scores['auc'].append(scores[2])
            cv_scores['precision'].append(scores[3])
            cv_scores['recall'].append(scores[4])
            
            # Update best model
            if scores[2] > best_auc:
                best_auc = scores[2]
                best_fold_model = model
                best_fold_components = (vectorizer, text_scaler, meta_scaler)
            
            print(f"\nFold {fold} Results:")
            print(f"Accuracy: {scores[1]:.4f}")
            print(f"AUC: {scores[2]:.4f}")
            print(f"Precision: {scores[3]:.4f}")
            print(f"Recall: {scores[4]:.4f}")
        
        # Print final results
        print("\nFinal Cross-validation Results:")
        for metric, scores in cv_scores.items():
            print(f"\n{metric.capitalize()}:")
            print(f"  Mean: {np.mean(scores):.4f}")
            print(f"  Std:  {np.std(scores):.4f}")
            print(f"  Min:  {np.min(scores):.4f}")
            print(f"  Max:  {np.max(scores):.4f}")
        
        return best_fold_model, best_fold_components
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None
    
def export_model_for_extension(model, components, output_dir='extension/model'):
    """Export model and components for Chrome extension using direct serialization"""
    try:
        if output_dir is None:
            # Get the project root directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, 'chrome-extension', 'model')
        
        os.makedirs(output_dir, exist_ok=True)
        
        vectorizer, text_scaler, meta_scaler = components
        
        # Save model architecture
        model_json = model.to_json()
        with open(os.path.join(output_dir, 'model_architecture.json'), 'w') as f:
            f.write(model_json)
        
        # Save weights
        model.save_weights(os.path.join(output_dir, 'model_weights.h5'))
        
        # Save preprocessing parameters
        preprocessing_params = {
            'vocabulary': vectorizer.get_feature_names_out().tolist(),
            'max_features': vectorizer.max_features,
            'importance_categories': ['Critical', 'High', 'Medium'],
            'time_categories': ['Morning', 'Afternoon', 'Evening', 'Night'],
            'text_scaler': {
                'mean': text_scaler.mean_.tolist(),
                'scale': text_scaler.scale_.tolist()
            },
            'meta_scaler': {
                'mean': meta_scaler.mean_.tolist(),
                'scale': meta_scaler.scale_.tolist()
            },
            'model_config': {
                'text_input_dim': len(vectorizer.get_feature_names_out()),
                'meta_input_dim': meta_scaler.n_features_in_
            },
            'priority_config': PRIORITY_CONFIG
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        preprocessing_params = convert_to_serializable(preprocessing_params)
        
        with open(os.path.join(output_dir, 'preprocessing_params.json'), 'w') as f:
            json.dump(preprocessing_params, f, indent=2)
            
        # Save additional model metadata
        model_metadata = {
            'input_shapes': {
                'text_input': model.inputs[0].shape.as_list(),
                'meta_input': model.inputs[1].shape.as_list()
            },
            'output_shape': model.outputs[0].shape.as_list(),
            'layer_config': [
                {
                    'name': layer.name,
                    'type': layer.__class__.__name__,
                    'config': layer.get_config()
                }
                for layer in model.layers
            ]
        }
        
        with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
            json.dump(model_metadata, f, indent=2)
            
        print(f"Model exported for extension in {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error exporting model for extension: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False
    
def main():
    """Main execution function with extension export"""
    try:
        # Step 1: Prepare training data
        print("\nStep 1: Preparing training data...")
        print("=" * 50)
        formatted_df = prepare_training_data('Task Prioritization.csv')
        
        # Step 2: Train model
        print("\nStep 2: Training model...")
        print("=" * 50)
        result = train_model('formatted_training_data.csv')
        if result is None or result[0] is None:
            raise Exception("Model training failed")
        model, components = result
        
        # Step 3: Save model (original format)
        print("\nStep 3: Saving model...")
        print("=" * 50)
        save_success = save_trained_model(model, components)
        
        if not save_success:
            raise Exception("Failed to save model")
        
        # Step 4: Export for extension
        print("\nStep 4: Exporting model for extension...")
        print("=" * 50)
        export_success = export_model_for_extension(model, components)
        
        if not export_success:
            raise Exception("Failed to export model for extension")
        
        # Step 5: Test predictions
        print("\nStep 5: Testing the model...")
        print("=" * 50)
        predictor = PriorityPredictor()
        
        test_tasks = [
            {
                'title': 'Critical production system down',
                'description': 'Production system affecting all users needs immediate attention. Multiple services are impacted.',
                'importance': 'Critical',
                'estimated_time': 4.0,
                'preferred_time': 'Morning',
                'workload': 0.8
            },
            {
                'title': 'Update documentation',
                'description': 'Regular documentation update for new features and improvements',
                'importance': 'Medium',
                'estimated_time': 2.0,
                'preferred_time': 'Afternoon',
                'workload': 0.3
            },
            {
                'title': 'Security patch deployment',
                'description': 'Critical security vulnerability needs patching across all systems',
                'importance': 'High',
                'estimated_time': 3.0,
                'preferred_time': 'Morning',
                'workload': 0.7
            }
        ]
        
        print("\nPrediction Results:")
        print("=" * 70)
        for task in test_tasks:
            result = predictor.predict_priority(task)
            print(f"\nTask: {task['title']}")
            print(f"Description: {task['description'][:100]}...")
            print(f"Importance: {task['importance']}")
            print(f"Estimated Time: {task['estimated_time']} hours")
            print(f"Preferred Time: {task['preferred_time']}")
            print(f"Current Workload: {task['workload']}")
            print("\nPrediction Details:")
            print(f"Priority: {result['priority']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Raw Model Score: {result['raw_score']:.4f}")
            print(f"Adjusted Threshold: {result['adjusted_threshold']:.4f}")
            print("-" * 70)
        
        print("\nExecution completed successfully!")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        print(traceback.format_exc())

def test_saved_model():
    """Function to test a previously saved model"""
    try:
        predictor = PriorityPredictor('model_output')
        
        test_task = {
            'title': 'New feature implementation',
            'description': 'Implement new user authentication system',
            'importance': 'High',
            'estimated_time': 5.0,
            'preferred_time': 'Morning',
            'workload': 0.6
        }
        
        result = predictor.predict_priority(test_task)
        
        print("\nTest Prediction Result:")
        print("-" * 50)
        print(f"Task: {test_task['title']}")
        print(f"Importance: {test_task['importance']}")
        print(f"Predicted Priority: {result['priority']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # Setup
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
    EXTENSION_DIR = os.path.join(BASE_DIR, 'chrome-extension')

    directories = [
        os.path.join(BACKEND_DIR, 'model_output'),
        os.path.join(EXTENSION_DIR, 'model')
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
    # Run main function
    print("\n=== Task Priority Prediction System ===")
    print("Version: 1.0")
    print("Starting execution...")
    main()