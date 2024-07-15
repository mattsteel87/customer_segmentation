import pandas as pd

def output_data(data, model, output_file):
    data['cluster'] = model.labels_
    final_data = data[['CustomerID','Gender','Age','Annual Income','Spending Score','cluster']]
    final_data.to_csv(output_file)
    return