import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def process_vectors(train_data, test_data, dedup=True):
    
    # Convert the list of lists into a list of strings, where each string represents a sequence.
    sequences = [' '.join(seq["EventId"]) for seq in train_data]
    
    # Create a CountVectorizer object.
    vectorizer = CountVectorizer()
    
    # Fit the vectorizer to the sequences. This will learn the vocabulary of unique EventIds.
    vectorizer.fit(sequences)
    
    # Transform the sequences into count vectors.
    count_vectors = vectorizer.transform(sequences)
    
    # The 'count_vectors' variable now contains the count vectors.  You can access them like a sparse matrix:
    # print(count_vectors.toarray())  # toarray() converts it to a dense matrix for easier viewing
    
    # To get the feature names (i.e., the unique EventIds), use vectorizer.get_feature_names_out():
    feature_names = vectorizer.get_feature_names_out()

    train_processed = []
    train_labeled = []
    for i in train_data:
      count_vector = np.array([0]*len(feature_names))
      for event in i['EventId']:
        count_vector[np.where(feature_names == event)[0]] += 1
      train_processed.append(tuple(count_vector))
      train_labeled.append(i['Label'])
    
    test_processed = []
    for j in test_data:
      count_vector = np.array([0]*len(feature_names))
      for event in j['EventId']:
        if event in feature_names:
          count_vector[np.where(feature_names == event)[0]] += 1
      test_processed.append(tuple(count_vector))
    test_labeled = []
    for i in test_data:
        test_labeled.append(i['Label'])
    
    ## dedup the test data

    if dedup == True:
        test_dedup_processed = []
        test_dedup_labeled = []
        for j in test_data:
          count_vector = np.array([0]*len(feature_names))
          if tuple(j["EventId"]) not in unique_seqs:
            for event in j['EventId']:
              if event in feature_names:
                count_vector[np.where(feature_names == event)[0]] += 1
            test_dedup_processed.append(tuple(count_vector))
            test_dedup_labeled.append(j['Label'])
        
        test_processed = test_dedup_processed
        test_labeled = test_dedup_labeled

    return train_processed, train_labeled, test_processed, test_labeled