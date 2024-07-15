from sklearn.cluster import KMeans

def train_model(data):
    model = KMeans(n_clusters = 4)
    model.fit(data[['Age T', 'Annual Income T','Spending Score T']])
    return model