# predict_priority.py
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
from tf_keras import layers, regularizers

def create_model(text_input_dim, meta_input_dim):
    """Create the model architecture - same as training script"""
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
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class PriorityPredictor:
    def __init__(self, model_dir='model_output'):
        """Initialize predictor with improved error handling"""
        try:
            # Load model info
            model_info = joblib.load(os.path.join(model_dir, 'model_info.pkl'))
            
            # Create fresh model
            self.model = create_model(
                model_info['text_input_dim'],
                model_info['meta_input_dim']
            )
            
            # Load weights
            weights_path = os.path.join(model_dir, 'model_weights.h5')
            self.model.load_weights(weights_path)
            
            # Load preprocessors
            self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
            self.text_scaler = joblib.load(os.path.join(model_dir, 'text_scaler.pkl'))
            self.meta_scaler = joblib.load(os.path.join(model_dir, 'meta_scaler.pkl'))
            
            self.time_map = {
                'Morning': 9 * 60,
                'Afternoon': 13 * 60,
                'Evening': 18 * 60,
                'Night': 21 * 60
            }
            
            print("Model and components loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def predict_priority(self, task_data):
        """Predict priority for a single task"""
        try:
            # Prepare text features
            text = f"{task_data['title']}. {task_data['description']}"
            text_features = self.vectorizer.transform([text]).toarray()
            text_features_scaled = self.text_scaler.transform(text_features)
            
            # Prepare meta features
            importance_dummies = pd.get_dummies(
                pd.Series([task_data['importance']]), 
                prefix='importance'
            ).values
            time_dummies = pd.get_dummies(
                pd.Series([task_data['preferred_time']]), 
                prefix='time'
            ).values
            
            meta_features = np.hstack([
                [[task_data['estimated_time'] / 24.0]],
                [[task_data['workload']]],
                importance_dummies,
                time_dummies
            ])
            
            meta_features_scaled = self.meta_scaler.transform(meta_features)
            
            # Make prediction
            prediction = self.model.predict(
                [text_features_scaled, meta_features_scaled],
                verbose=0
            )[0][0]
            
            # Determine priority and confidence
            is_high_priority = prediction > 0.5
            confidence = prediction if is_high_priority else 1 - prediction
            
            return {
                'priority': 'High' if is_high_priority else 'Normal',
                'confidence': float(confidence),
                'raw_score': float(prediction)
            }
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None

def test_saved_model():
    """Test the saved model with sample tasks"""
    try:
        predictor = PriorityPredictor('model_output')
        
        test_tasks = [
            {
                'title': 'Critical production system down',
                'description': 'Production system affecting all users needs immediate attention',
                'importance': 'Critical',
                'estimated_time': 4.0,
                'preferred_time': 'Morning',
                'workload': 0.8
            },
            {
                'title': 'Update documentation',
                'description': 'Regular documentation update for new features',
                'importance': 'Medium',
                'estimated_time': 2.0,
                'preferred_time': 'Afternoon',
                'workload': 0.3
            },
            {
                'title': 'Security patch deployment',
                'description': 'Deploy critical security updates to production',
                'importance': 'High',
                'estimated_time': 3.0,
                'preferred_time': 'Morning',
                'workload': 0.7
            }
        ]
        
        print("\nTesting Saved Model:")
        print("-" * 50)
        
        for task in test_tasks:
            result = predictor.predict_priority(task)
            if result:
                print(f"\nTask: {task['title']}")
                print(f"Description: {task['description']}")
                print(f"Importance: {task['importance']}")
                print(f"Estimated Time: {task['estimated_time']} hours")
                print(f"Preferred Time: {task['preferred_time']}")
                print(f"Current Workload: {task['workload']}")
                print(f"Predicted Priority: {result['priority']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Raw Score: {result['raw_score']:.4f}")
                print("-" * 30)
            else:
                print(f"Failed to get prediction for task: {task['title']}")
        
        print("\nTesting completed!")
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_saved_model()