import pickle
import numpy as np
import random 

def sample_data(train_path, sample_number):
    sampled_data = []

    train = []
    with open(train_path, "rb") as f:
        train = pickle.load(f)

    if "ADFA" in train_path:
        # random sampling
        
        n_samples = int(sample_number/2)
        
        normal_indices = [i for i, t in enumerate(train) if t["Label"] == 0]
        anomalous_indices = [i for i, t in enumerate(train) if t["Label"] == 1]
    
        # Sample normal instances
        sampled_normal_indices = np.random.choice(normal_indices, n_samples, replace = True)
        sampled_normal_data = [train[i] for i in sampled_normal_indices]
        sampled_normal_labels = [train[i]["Label"] for i in sampled_normal_indices]
    
        # Sample anomalous instances
        sampled_anomalous_indices = np.random.choice(anomalous_indices, n_samples, replace = True)
        sampled_anomalous_data = [train[i] for i in sampled_anomalous_indices]
        sampled_anomalous_labels = [train[i]["Label"] for i in sampled_anomalous_indices]
    
        # Combine the sampled data and labels
        sampled_data = sampled_normal_data + sampled_anomalous_data
        sampled_labels = sampled_normal_labels + sampled_anomalous_labels
    
        sampled_data = []
        for i in range(n_samples):
          sampled_data.append(train[sampled_normal_indices[i]])
          sampled_data.append(train[sampled_anomalous_indices[i]])
        return sampled_data

    else:
        # sample with 1 and 0.2 ratio
        unique_normal = set()
        unique_anomalous = set()
        normal_indexes = []

        for i, t in enumerate(train):
            if t["Label"] and tuple(t["EventId"]) not in unique_anomalous:
                sampled_data.append(t)
                unique_anomalous.add(tuple(t["EventId"]))
            if t["Label"]==0 :
                unique_normal.add(tuple(t["EventId"]))
                normal_indexes.append(i)

        n_normal = int(unique_normal*0.2)
        sampled_normal_indices = np.random.choice(normal_indexes, n_normal, replace = False)
        for i in sampled_normal_indices:
            sampled_data.append(train[i])

        return sampled_data
        
        
                


        
    