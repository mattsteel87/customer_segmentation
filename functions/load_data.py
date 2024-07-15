import pandas as pd 

def load_data(filename):
    data = pd.read_csv(filename)
    data.columns = ['CustomerID',
                    'Gender',
                    'Age',
                    'Annual Income',
                    'Spending Score']
    return data