# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import math
import random

# COMMAND ----------

def calculatekClusters(pdataset,k,maxIter):
    #10 point,3,5
    k_indexes = random.sample(range(0,len(pdataset)-1),k)
    results = {} 
    k_centriods=[]
    k_colors=['red','green','blue','brown','yellow']
    index = 0
    for k_index in k_indexes:
        k_centriods.append(pdataset[int(k_index)])
        index = index + 1        
    print(k_indexes)
    plt.scatter(pdataset[:],pdataset[:],s=50,c='blue')
    plt.scatter(k_centriods[:],k_centriods[:],s=50,c='green')

    #print(pdataset[0])
    counter = 1
    while counter<=maxIter:
        results={}
        number_of_shuffles = 0
        print("******** Iteration-"+str(counter)+" *************************")
        for x_value in pdataset:
            min_distance = 1000.0
            s_k_index = -1
            cluster_id = 0
            m_cluster_id = 0
            x_index = 0
            for x_centriod in k_centriods:                
                if np.abs((x_value-x_centriod))<min_distance:                    
                    min_distance = np.abs((x_value-x_centriod))
                    s_k_index = cluster_id
                    m_cluster_id=cluster_id+1
                    number_of_shuffles = number_of_shuffles + 1
                cluster_id = cluster_id + 1    
            results[x_value]="Cluster-"+str(m_cluster_id)
            
        counter = counter + 1
        
        print(results)
        print("Number of Shuffles are ",number_of_shuffles)
        print("******** centriods *************************")
        print(k_centriods)        
        if counter<=maxIter:
          plt.scatter(k_centriods[:],k_centriods[:],s=100,c=k_colors[counter-1])
        k_centriods=recalculateCentriod(results)        
    return results 

# COMMAND ----------

def recalculateCentriod(p_results):    
    k_centriods = []
    k_clusters = [] # storing Tags into k_clusters
    
    i_index = 0
    for i_key,i_value in p_results.items():        
        if len(k_clusters)==0:
            k_clusters.insert(0,i_value)
            i_index=k_clusters.index(i_value)
            k_centriods.insert(i_index,i_key)

        else:
            if i_value in k_clusters:                
                i_index=k_clusters.index(i_value)
                k_centriods[i_index]=(k_centriods[i_index]+i_key)/2
            else:
                k_clusters.append(i_value)
                i_index=k_clusters.index(i_value)
                k_centriods.insert(i_index,i_key)       

    return k_centriods    

# COMMAND ----------

pdataset = (np.random.rand(10)*100)

# COMMAND ----------

#print(type(pdataset))
print(pdataset)

# COMMAND ----------

k_means_result=calculatekClusters(pdataset,3,5)

# COMMAND ----------

print(k_means_result)

# COMMAND ----------

plt.scatter(pdataset[:],pdataset[:],s=50,c='blue')

# COMMAND ----------


