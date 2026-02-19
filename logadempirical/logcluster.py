import os

from logadempirical.loglizer import dataloader, preprocessing
from logadempirical.loglizer.models import LogClustering
import pickle
import pandas as pd
import time



def run_logcluster(options):

    (x_train, y_train), (x_test, y_test) = dataloader.load_preprocessed(options["output_dir"], options["data_dir"] + options["folder"].replace("/","") + ".log_templates.csv")

    x_train_normal = pd.Series([x for i, x in enumerate(x_train) if y_train[i]==0])
    
    feature_extractor = preprocessing.FeatureExtractor()

    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')

    x_train_normal = feature_extractor.transform(x_train_normal)

    x_test = feature_extractor.transform(x_test)

    model = LogClustering(max_dist=options["max_dist"], anomaly_threshold=options["anomaly_threshold"], num_bootstrap_samples=options["num_bootstrap_samples"])
    start_time = time.time()
    model.fit(x_train_normal)
    train_time =  time.time() - start_time
    print('Train time:', train_time)
    
    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)
    
    print('Test validation:')
    start_time = time.time()
    precision, recall, f1 = model.evaluate(x_test, y_test)
    test_time =  time.time() - start_time
    print('Test time:', test_time, len(y_test))
