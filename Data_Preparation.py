# coding: utf-8
# In[ ]:

# Import libraries

import math
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from itertools import chain, combinations
import random

# In[ ]:
# clean price field for all records (kepping by keeping aside the 7500: rows)

az_price22=pd.read_csv('Amazon_Price.csv')
az_price22=pd.DataFrame(az_price22) 
az_price=az_price22.copy()
del az_price['Unnamed: 0']
del az_price['First_Bought_Product']

Price1=az_price.iloc[:7500]
Price2=az_price.iloc[7500:]
Price2.tail()

def cleanprice(flat_price):
    for i in range(0,len(flat_price)):
        for j in range(0,len(flat_price.iloc[i])):
            if (str(flat_price.iloc[i][j]).find('£')>=0):
                start=(str(flat_price.iloc[i][j]).find('£'))+1
                end=len(str(flat_price.iloc[i][j]))
                flat_price.iloc[i][j]=str(flat_price.iloc[i][j][start:end])
            else:
                flat_price.iloc[i][j]=''
    return flat_price

Price3=cleanprice(Price1)
Price4=pd.concat([Price3,Price2], axis=0)

# Generate new file for price data
Price4.to_csv('Amazon_Association_Total Data_Price.csv')
price_df = pd.read_csv('Amazon_Association_Total Data_Price.csv')
rating_df = pd.read_csv('Amazon_Average_Rating.csv')
names_df = pd.read_csv('Amazon_Product_Names.csv')

###### Delet non required columns
del names_df['Unnamed: 0']
del rating_df['Unnamed: 0']
del price_df['Unnamed: 0']
del names_df['6']
del rating_df['First_Bought_Product']

# Create Names vector or separate data frame with single vector
agg_df = pd.DataFrame(names_df.values.ravel(), columns=['Names'])

# Remove duplicates and Nans

old=list(agg_df.iloc[:,0])
new_prd=set(old)
#print(len(new_prd))
new_prd=list(new_prd)
#print(len(new_prd))
names_noNA=list()

for i in range(0,len(new_prd)):
    if (pd.isna(new_prd[i])):
        continue
    else:
        names_noNA.append(new_prd[i])

len(new_prd), len(names_noNA)


# Get indexes of data from data frame

ind_n=list()
type(ind_n)
for i in range(0,len(names_noNA)):
    num_i=old.index(names_noNA[i])
    ind_n.append(num_i)

#print(ind_n)

# Final names vector for clustering
Final_names_Clustering=agg_df.iloc[ind_n,:]
Final_names_Clustering.head()
len(Final_names_Clustering)

# Price vector for clustering
agg_price_df = pd.DataFrame(price_df.values.ravel(), columns=['Price'])
Final_price_Clustering=agg_price_df.iloc[ind_n,:]
Final_price_Clustering.head()

# Price rating for clustering
agg_rating_df = pd.DataFrame(rating_df.values.ravel(), columns=['Rating'])
Final_rating_Clustering=agg_rating_df.iloc[ind_n,:]
Final_rating_Clustering.head()

# Create new data frame by merging these three frames

final_clustering_df=pd.concat([Final_names_Clustering,Final_price_Clustering,Final_rating_Clustering], axis=1)
final_clustering_df.head()

# Handle values in price column
def cleanrating_clustering(flat_price):
    for i in range(0,len(flat_price)):
        if(pd.isna(flat_price.iloc[i][2])):
            flat_price.iloc[i,2]=0
        else:
            if(str(flat_price.iloc[i][2]).find('o')==0):
                flat_price.iloc[i,2]=0
            elif(str(flat_price.iloc[i][2]).find('.')<0):
                flat_price.iloc[i,2]=str(flat_price.iloc[i][2])[0:1]
            else:
                flat_price.iloc[i,2]=flat_price.iloc[i][2]
    return flat_price

# Handle values in price column
def cleanprice_clustering(flat_price):
    for i in range(0,len(flat_price)):
        if (str(flat_price.iloc[i][1]).find('<')>=0):
            start=str(flat_price.iloc[i][1]).find('<')
            flat_price.iloc[i,1]=str(flat_price.iloc[i][1][0:start])
        elif(str(flat_price.iloc[i][1]).find('£')>=0):
            y=(str(flat_price.iloc[i][1]).find('£'))-3
            flat_price.iloc[i,1]=str(flat_price.iloc[i][1][0:y])
        elif(str(flat_price.iloc[i][1]).find(',')>=0):
            x=(str(flat_price.iloc[i][1]).find(','))+1
            yy=len(str(flat_price.iloc[i][1]))
            flat_price.iloc[i,1]=str(flat_price.iloc[i][1][x:yy])
        else:
            flat_price.iloc[i,1]=flat_price.iloc[i][1]
    return flat_price

# Execute function to handle price values and convert price to float value

final_clustering_df=cleanprice_clustering(final_clustering_df)
final_clustering_df=cleanprice_clustering(final_clustering_df)
final_clustering_df['Price'] = final_clustering_df['Price'].astype(float)

# Get float values for rating by cleaning it first
final_clustering_df=cleanrating_clustering(final_clustering_df)
final_clustering_df=cleanrating_clustering(final_clustering_df)
final_clustering_df['Rating'] = final_clustering_df['Rating'].astype(float)

# Mean value imputation

mean_value=final_clustering_df['Price'].mean()
final_clustering_df['Price']=final_clustering_df['Price'].fillna(mean_value)
final_clustering_df.isna().sum()
final_clustering_df.head()

#Testing data on built in K-Modes algorithm/ K-mode clustering with built-in package
from kmodes.kmodes import KModes
km = KModes(n_clusters=10, init='Huang', n_init=11, verbose=1)
clusters = km.fit_predict(final_clustering_df)
labels = km.predict(final_clustering_df)

kmodes = km.cluster_centroids_
shape = kmodes.shape
clusters, kmodes, shape
print(kmodes)

#Print Clusters
cluster_map = pd.DataFrame()
cluster_map['data_index'] = final_clustering_df.index.values
cluster_map['cluster'] = km.labels_

#Print clusters
cluster_map[cluster_map.cluster == 0]

