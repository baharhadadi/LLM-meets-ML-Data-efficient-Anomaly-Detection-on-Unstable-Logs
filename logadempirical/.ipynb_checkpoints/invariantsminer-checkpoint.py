import os

from logadempirical.loglizer import dataloader, preprocessing
from logadempirical.loglizer.models import InvariantsMiner
import pickle



def run_invariantsminer(options):

    (x_train, y_train), (x_test, y_test) = dataloader.load_preprocessed(options["output_dir"], options["data_dir"] + options["folder"].replace("/","") + ".log_templates.csv")

    feature_extractor = preprocessing.FeatureExtractor()

    x_train = feature_extractor.fit_transform(x_train)

    x_test = feature_extractor.transform(x_test)

    model = InvariantsMiner(epsilon=0.5)
    model.fit(x_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)
    
    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)