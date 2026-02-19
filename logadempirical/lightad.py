import os
import pickle
import pandas as pd
import time
import numpy as np

from lightad.models.DT import decision_tree
from lightad.models.KNN import KNN
from lightad.models.SLFN import MLP
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.feature_extraction.text import CountVectorizer

def run_lightad(options):

    #load the datasets
    with open(os.path.join(options["output_dir"], "test.pkl"), "rb") as f:
        test_data = pickle.load(f)

    with open(os.path.join(options["output_dir"], "train.pkl"), "rb") as f:
        train_data = pickle.load(f)
        
    # process dataset
    sequences = [' '.join(seq["EventId"]) for seq in train_data]
    vectorizer = CountVectorizer()
    vectorizer.fit(sequences)
    count_vectors = vectorizer.transform(sequences)
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

    # train and infer the model
    if options["lightad_model"] == "DT":
         prediction, train_time, infer_time = DT(train_processed, test_processed,train_labeled, 
                                                 criterion = options["DT_criterion"], 
                                                 min_samples_leaf = options["DT_min_samples_leaf"],
                                                 max_depth = options["DT_max_depth"],
                                                 min_samples_split = options["DT_min_samples_split"]
                                                )
    elif options["lightad_model"] == "KNN":
        prediction, train_time, infer_time = KNN(train_processed, test_processed,train_labeled,
                                                 n_neighbors = options["KNN_n"],
                                                 metric = options["KNN_metric"]
                                                )
    else: #SLFN
        prediction, train_time, infer_time = MLP(train_processed, test_processed,train_labeled,
                                                 hidden_layer_sizes = (int(options["SLFN_hidden_neurons"]),),
                                                 solver = options["SLFN_solver"],
                                                 activation = options["SLFN_activation"],
                                                 tol = options["SLFN_tol"],
                                                 max_iter = options["SLFN_max_iter"]
                                                )

    # print the evaluation results
    print('Train time:', train_time)
    print('Infer time:', infer_time)
    y_true = test_labeled
    p, r, f1 = precision_score(y_true, prediction), recall_score(y_true, prediction), f1_score(y_true, prediction)
    print('precision: ', p,' recall: ', r,'f1-score: ', f1)