import torch
import pickle
import pandas as pd
import random
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth import is_bfloat16_supported
from argparse import ArgumentParser
import time
from sklearn.metrics import f1_score, recall_score, precision_score
import os
import random
import numpy as np
from flexlog.models.LLM import LogSequenceModel
from flexlog.preprocessing.vectorization import process_vectors
from flexlog.preprocessing.dataloader import load_data
from flexlog.preprocessing.sample import sample_data
from flexlog.tools.ensemble import mj_ensemble, snail_ensemble, metaformer_ensemble
from flexlog.models.DT import decision_tree
from flexlog.models.KNN import KNN
from flexlog.models.SLFN import MLP

def parse_arguments():
    """
    Parse input arguments provided via the command line.
    """
    parser = ArgumentParser(description="Train a Log Sequence Model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--train_set_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--test_set_path", type=str, required=True, help="Path to the testing dataset")
    parser.add_argument("--templates_file", type=str, required=True, help="Path to the templates file")
    parser.add_argument("--description_file", type=str, default = None, help="Path to the description file (pkl)")
    parser.add_argument("--results_file", type=str, required=True, help="Path to save the results")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples in random sampling")
    parser.add_argument("--max_steps", type=int, default=600, help="Maximum number of training steps")
    parser.add_argument("--max_length", type=int, default=32000, help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to the output directory")
    parser.add_argument("--save_output", type=str, default=None, help="Path to the model saved directory")
    parser.add_argument("--threshold", type=float, default=0.85, help="threshold for confidence score")
    parser.add_argument("--prediction_file", type=str, default="prediction.pkl", help="the name of the prediction file")
    parser.add_argument("--knn_neighbor", type=int, default=2, help="the number of neighbors in knn")
    parser.add_argument("--ensemble_method", type=str, default="mj", help="type of ensemble learning")
    parser.add_argument("--csv_results_flexlog", type=str, default=None, help="path to the csv file of effectiveness results")
    return parser.parse_args()


if  __name__ == "__main__":
    """
    Main function to parse arguments and initiate model training.
    """
    # Restrict to using only one GPU
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to the GPU ID you want to use

    # Check if CUDA is available and report the GPU being used
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Please check your setup.")
    args = parse_arguments()

    # load datasets
    train_data, test_data = load_data(args.train_set_path, args.test_set_path, dedup = True)

    # Sample limited data
    sampled_data = sample_data(args.train_set_path, args.num_samples)
    args.output_dir = args.output_dir if args.output_dir != "output" else args.model_name+"-output"
    args.save_output = args.output_dir if args.save_output == None else args.save_output
    args.csv_results_flexlog = args.output_dir if args.csv_results_flexlog == None else args.csv_results_flexlog
    sample_path = args.output_dir + "/train_sampled_data.pkl"
    
    with open(sample_path, "wb") as f:
        pickle.dump(sampled_data, f)

    # process the count vectors for ML models
    train_processed, train_labels, test_processed, test_labels = process_vectors(sampled_data, test_data)
    
    # Instantiate the LogSequenceModel class with parsed arguments
    log_model = LogSequenceModel(
        mode='train',
        model_name=args.model_name,
        train_set_path=sample_path,
        test_set_path=args.test_set_path,
        templates_file=args.templates_file,
        description_file=args.description_file,
        results_file=args.results_file,
        max_length=args.max_length,
        max_steps=args.max_steps,
        output_dir = args.output_dir,
        threshold = args.threshold,
        save_output = args.save_output,
        prediction = args.prediction_file
    )
    
    # Train the model
    log_model.train_model()

    #cache enhanced inference
    
    # load fine-tuned LLM
    log_model = LogSequenceModel(
        mode='infer',
        model_name=args.save_output,
        train_set_path=sample_path,
        test_set_path=args.test_set_path,
        templates_file=args.templates_file,
        description_file=args.description_file,
        results_file=args.results_file,
        max_length=args.max_length,
        max_steps=args.max_steps,
        output_dir = args.output_dir,
        threshold = args.threshold,
        save_output = args.save_output,
        prediction = args.prediction_file
    )
    
    log_model.inference()

    # load llm inference
    with open(args.prediction_file, "rb") as f:
        llm_predictions = pickle.load(f)

    prediction_llm = []
    for t in test_data:
        if tuple(t["EventId"]) in llm_predictions:
            prediction_llm.append(llm_predictions[tuple(t["EventId"])][0])

    # obtain the ML inference
    prediction_mlp, train_time_mlp, infer_time_mlp = MLP(train_processed, test_processed,train_labels)
    prediction_knn, train_time_knn, infer_time_knn = KNN(train_processed, test_processed,train_labels, n_neighbors= args.knn_neighbor)
    prediction_dt, train_time_dt, infer_time_dt = decision_tree(train_processed, test_processed,train_labels)

    # average ensemble learning
    model_predictions = [
            np.array(prediction_dt),  
            np.array(prediction_knn),  
            np.array(prediction_mlp),
            np.array(prediction_llm)]

    ensemble_results = {}

    if args.ensemble_method == "mj":
        mj_ensemble_preds = mj_ensemble(model_predictions)

        # evaluate the results
        y_true = test_labels
        ensemble_results["Majority Voting Ensemble"] = {
            "precision": precision_score(y_true, mj_ensemble_preds),
            "recall": recall_score(y_true, mj_ensemble_preds),
            "f1_score": f1_score(y_true, mj_ensemble_preds)}

    elif args.ensemble_method == "snail":

        log_model.test_set_path = log_model.train_set_path
        log_model.inference()
        with open(args.prediction_file, "rb") as f:
            llm_predictions_train = pickle.load(f)
        snail_ensemble_preds = snail_ensemble(model_predictions, llm_predictions_train, train_processed, train_labels )

        # evaluate the results
        y_true = test_labels
        ensemble_results["SNAIL Ensemble"] = {
            "precision": precision_score(y_true, snail_ensemble_preds),
            "recall": recall_score(y_true, snail_ensemble_preds),
            "f1_score": f1_score(y_true, snail_ensemble_preds)}

    elif args.ensemble_method == "metaformer":

        log_model.test_set_path = log_model.train_set_path
        log_model.inference()
        with open(args.prediction_file, "rb") as f:
            llm_predictions_train = pickle.load(f)
        mf_ensemble_preds = metaformer_ensemble(model_predictions, llm_predictions_train, train_processed, train_labels )

        # evaluate the results
        y_true = test_labels
        ensemble_results["MetaFormer Ensemble"] = {
            "precision": precision_score(y_true, mf_ensemble_preds),
            "recall": recall_score(y_true, mf_ensemble_preds),
            "f1_score": f1_score(y_true, mf_ensemble_preds)}

    for k, v in ensemble_results.items():
        file_exists = os.path.isfile(args.csv_results_flexlog)

        # Write or append results
        with open(args.csv_results_flexlog, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["precision", "recall", "f1_score"])
            writer.writerow([v["precision"], v["recall"], v["f1_score"])

        print(f"Results saved to {args.csv_results_flexlog}")
    print(ensemble_results)


    


    
    

    
