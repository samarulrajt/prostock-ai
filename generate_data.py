import pandas as pd
import numpy as np

def generate_churn_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'tenure': np.random.randint(0, 60, n_samples), # months with company
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 7000, n_samples),
        'support_calls': np.random.randint(0, 10, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Add some logic to make the model learnable
    # Higher support calls and month-to-month contracts increase churn probability
    df.loc[(df['support_calls'] > 5) & (df['contract_type'] == 'Month-to-month'), 'churn'] = 1
    df.loc[(df['tenure'] > 40) & (df['contract_type'] == 'Two year'), 'churn'] = 0
    
    return df

if __name__ == "__main__":
    df = generate_churn_data()
    df.to_csv('customer_data.csv', index=False)
    print("Synthetic dataset 'customer_data.csv' created successfully!")
