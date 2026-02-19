import os

from logadempirical.loglizer import dataloader, preprocessing
from logadempirical.loglizer.models import PCA
import pickle



def run_pca(options):

    (x_train, y_train), (x_test, y_test) = dataloader.load_preprocessed(options["output_dir"], options["data_dir"] + options["folder"].replace("/","") + ".log_templates.csv")

    feature_extractor = preprocessing.FeatureExtractor()

    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')

    x_test = feature_extractor.transform(x_test)

    model = PCA()
    model.fit(x_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)
    
    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)