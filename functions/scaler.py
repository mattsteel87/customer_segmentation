from sklearn.preprocessing import StandardScaler

def scale_data(data):
        scaler = StandardScaler()
        data[['Age T','Annual Income T','Spending Score T']] = scaler.fit_transform(data[['Age','Annual Income','Spending Score']])
        return data