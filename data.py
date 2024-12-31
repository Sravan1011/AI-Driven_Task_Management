import pandas as pd
import numpy as np
from datetime import datetime

def prepare_training_data(csv_data):
    """
    Prepare and clean the data for the priority prediction model
    """
    # Read CSV data
    df = pd.read_csv(csv_data)
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Create the formatted dataset
    formatted_data = {
        'description': [],
        'importance': [],
        'estimated_time': [],
        'preferred_time': [],
        'workload': [],
        'priority': []  # Target variable
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
        # Description: Combine title, description, and keywords
        description = f"{row['Task Title']}. {row['Task Description']}. Keywords: {row['Task Keywords and Description']}"
        
        # Convert time string to float (assuming format HH:MM)
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
        
        # Determine priority (High for Critical and High importance)
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
    
    # Save formatted data
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

# Example usage
if __name__ == "__main__":
    # Prepare the data
    formatted_df = prepare_training_data('Task Prioritization.csv')
    print("\nData preparation completed! Formatted data saved to 'formatted_training_data.csv'")