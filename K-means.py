#%%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#%%
# 3.1
df_train_raw = pd.read_csv("dow_jones_index.csv")

#3.2
# Removing the given three categorical values
columns = ["quarter", "stock", "date"]
df_train = df_train_raw.drop(columns,1)
df_train.dtypes

#%%
# 3.2
#Data Cleaning
#striping the $ sign for few columns
for col in df_train.select_dtypes([np.object]):
    df_train[col] = df_train[col].str.lstrip('$')

# Rounding the data for few columns
for col in df_train.select_dtypes([np.float64]):
    df_train[col] = round(df_train[col],4)

df_train.dropna(inplace=True)
print(df_train.iloc[1])
#%%
#Pre-processing
#Normalization and outlier removal
# normalize the data attributes
# We will need to normalize data to the range 0 to 1 because, here we use distance measures so we to scale
# the input attributes for the model that relies on the magnitude of the values
# Since the data need not be normalized we are not using the data standardization.
# For this problem we are just using normalization
colnames = list(df_train.columns.values)
normalize = MinMaxScaler()
df_train[colnames] = normalize.fit_transform(df_train[colnames])

#%%

#df_train.to_csv("cleaned_df.csv",sep=',')

#%%
def kMeans(X, K):

    # Selecting random values as centroids
    idx = np.arange(2,4)
    newcentpoint= np.asarray(X[np.random.choice(len(X),K, replace=False),:])
    sse = np.zeros((K,1))
    oldcentpoint = np.zeros((K,13))   ##X[np.random.choice(np.arange(len(X)), K), :]
    
    i=0
    while (oldcentpoint!=newcentpoint).any():
        oldcentpoint = newcentpoint.copy()
        cluster = np.zeros((720,3))
        # Cluster Assignment step, assigning points to cluster. C will contain, which
        # cluster the point belongs
        for x_i in range(len(X)): # For each point in the data set
            min_dist = np.Inf
            for index,y_k in enumerate(newcentpoint):# For each center point from the value K
                distance = np.sqrt(np.sum((X[x_i,:]-y_k)**2))  
                if distance<min_dist: # Calculating the min distance of that point to which cluster
                    min_dist=distance
                    idx=index
            ##Assigning point to particular cluster
            cluster[x_i]=idx, min_dist, min_dist**2
        # count iterations   
        i = i+1
        # Move centroids step; Calculating the new centroids
        for k in range(K):
            newcentpoint[k] = X[np.where(cluster[:,0]==k), :].mean(axis = 1)
            sse[k] = np.sum(cluster[np.where(cluster[:,0]==k), 2])
        tot_sse = np.sum(sse)
    #print('Iterations:',i)
    return newcentpoint, oldcentpoint, cluster,sse,tot_sse,i
#%%
k_vals = [2,3,4,5,6,7]
tsse_values_list = []
df_train = df_train.as_matrix()
for k_val in k_vals:    
    print(k_val)
    newcentpoint, oldcentpoint, cluster,sse,tot_sse,num_iter= kMeans(df_train, k_val)
    print('K_value: ', k_val)
    print('Number of iterations: ', num_iter)
    print('tot_sse: ', tot_sse)
    print('Centroid: ', newcentpoint)
    
    for clust in range(k_val):
        ids = np.where(cluster[:,0]==clust)
        print("SSE of current cluster: ", sse[clust])
        print(" Cluster ID's for kvalue = ", clust,':')
        print('Numbers of IDs in this cluster: ', np.shape(ids)[1])
        print(ids)
    tsse_values_list.append(tot_sse)

#%%
#Elbow/knee method
import matplotlib.pyplot as plt
plt.plot(k_vals, tsse_values_list, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
    
