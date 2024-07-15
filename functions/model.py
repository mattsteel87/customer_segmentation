from sklearn.cluster import KMeans

def train_model(data, clusters):
    model = KMeans(clusters)
    model.fit(data[['Age T', 'Annual Income T','Spending Score T']])
    return model