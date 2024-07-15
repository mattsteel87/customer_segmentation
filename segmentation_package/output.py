import pandas as pd

def output_data(data, model):
    data['cluster'] = model.labels_
    final_data = data[['CustomerID','Gender','Age','Annual Income','Spending Score','cluster']]
    final_data.to_csv('data/output.csv')
    return