import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_rows=20):
    """
    Generates a synthetic dataset of financial transactions.
    Includes normal patterns (recurring bills, daily spend) and anomalies.
    """
    np.random.seed(42)
    random.seed(42)

    categories = {
        'Food & Drink': ['Starbucks', 'McDonalds', 'Whole Foods', 'Local Cafe', 'Uber Eats'],
        'Groceries': ['Walmart', 'Target', 'Kroger', 'Safeway'],
        'Utilities': ['Electric Co', 'Water Dept', 'Internet Provider', 'Gas Company'],
        'Transport': ['Uber', 'Lyft', 'Shell Gas', 'Subway Ticket'],
        'Shopping': ['Amazon', 'Nike', 'H&M', 'Best Buy'],
        'Entertainment': ['Netflix', 'Spotify', 'Cinema XYZ', 'Steam'],
        'Health': ['CVS', 'Doctor Visit', 'Gym Membership'],
        'Income': ['Salary', 'Freelance', 'Refund']
    }

    data = []
    
    # Start date 1 year ago
    start_date = datetime.now() - timedelta(days=365)

    for _ in range(num_rows):
        # 1. Random Date
        random_days = random.randint(0, 365)
        date = start_date + timedelta(days=random_days)
        
        # 2. Pick Category & Merchant
        cat = random.choice(list(categories.keys()))
        merchant = random.choice(categories[cat])
        
        # 3. Generate Amount based on Category (INR Context)
        if cat == 'Income':
            amount = random.uniform(30000, 150000) # Salary range typical for freshers/mid-level
            is_expense = False
        elif cat == 'Food & Drink':
            amount = random.uniform(100, 1500) # Tea to fancy dinner
            is_expense = True
        elif cat == 'Groceries':
            amount = random.uniform(500, 5000) # Weekly grocery run
            is_expense = True
        elif cat == 'Utilities':
            amount = random.uniform(500, 3000) # Electricity, Internet
            is_expense = True
        elif cat == 'Transport':
            amount = random.uniform(50, 800) # Auto, Uber, Petrol
            is_expense = True
        elif cat == 'Shopping':
            amount = random.uniform(1000, 10000) # Clothes, Electronics
            is_expense = True
        elif cat == 'Entertainment':
            amount = random.uniform(300, 2000) # Movies, Netflix
            is_expense = True
        else: # Health and others
            amount = random.uniform(500, 5000)
            is_expense = True

        # 4. Inject some "Anomalies" (High spending spikes)
        # 2% chance of a high expense anomaly
        is_anomaly = False
        if is_expense and random.random() < 0.02:
            amount = amount * random.uniform(3, 10)
            is_anomaly = True
            
        dataset_row = {
            'Date': date.strftime('%Y-%m-%d'),
            'Merchant': merchant,
            'Category': cat,
            'Amount': round(amount, 2),
            'Is_Expense': is_expense,
            'Is_Anomaly': is_anomaly # This is our ground truth for training
        }
        data.append(dataset_row)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Save
    output_path = 'data/raw/transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {num_rows} transactions to {output_path}")
    print(df.head())
    print("\nAnomaly Count:")
    print(df['Is_Anomaly'].value_counts())

if __name__ == "__main__":
    generate_synthetic_data()
