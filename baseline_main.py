from argparse import ArgumentParser
import os
import shutil

from logadempirical.logdeep.tools.utils import seed_everything, save_parameters
from logadempirical.logdeep.dataset.injection import inject_window
from logadempirical.deeplog import run_deeplog
from logadempirical.loganomaly import run_loganomaly
from logadempirical.logrobust import run_logrobust
from logadempirical.cnn import run_cnn
from logadempirical.pca import run_pca
from logadempirical.logcluster import run_logcluster
from logadempirical.invariantsminer import run_invariantsminer
from logadempirical.bert import run_logbert
from logadempirical.plelog import run_plelog
from logadempirical.neurallog import run_neuralog
from logadempirical.lightad import run_lightad
from logadempirical.dataset import process_dataset, parse_log, sample_raw_data, process_instance

import sys
import time

sys.path.append("../../")


def arg_parser():
    """
    add parser parameters
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument("--model_name", help="which model to train", choices=["lighad", "logbert", "deeplog", "loganomaly",
                                                                              "logrobust", "cnn", "neurallog",
                                                                              "pca","invariantsminer", "logcluster", "autoencoder", "plelog"])
    parser.add_argument("--dataset_name", help="which dataset to use", choices=["adfa","hdfs", "bgl", "tbird", "hdfs_2k",
                                                                                "bgl_2k", "tdb", "spirit", "bo",
                                                                                "bgl2", "hadoop", "spark"])
    parser.add_argument("--device", help="hardware device", default="cuda")
    parser.add_argument("--trained", action='store_true', help="")
    parser.add_argument("--data_dir", default="./dataset/", metavar="DIR", help="data directory")
    parser.add_argument("--output_dir", default="./experimental_results/RQ1/random/", metavar="DIR",
                        help="output directory")
    parser.add_argument("--folder", default='bgl', metavar="DIR")

    parser.add_argument('--log_file', help="log file name")
    parser.add_argument("--sample_size", default=None, help="sample raw log")
    parser.add_argument("--sample_log_file", default=None, help="if sampling raw logs, new log file name")

    parser.add_argument("--parser_type", default=None, help="parse type drain or spell")
    parser.add_argument("--log_format", default=None, help="log format",
                        metavar="<Date> <Time> <Pid> <Level> <Component>: <Content>")
    parser.add_argument("--regex", default=[], type=list, help="regex to clean log messages")
    parser.add_argument("--keep_para", action='store_true', help="keep parameters in log messages after parsing")
    parser.add_argument("--st", default=0.3, type=float, help="similarity threshold")
    parser.add_argument("--depth", default=3, type=int, help="depth of all leaf nodes")
    parser.add_argument("--max_child", default=100, type=int, help="max children in each node")
    parser.add_argument("--tau", default=0.5, type=float,
                        help="the percentage of tokens matched to merge a log message")

    parser.add_argument("--is_process", action='store_true', help="if split train and test data")
    parser.add_argument("--is_instance", action='store_true', help="if instances of log are available")
    parser.add_argument("--train_file", default="train_fixed100_instances.pkl", help="train instances file name")
    parser.add_argument("--test_file", default="test_fixed100_instances.pkl", help="test instances file name")
    parser.add_argument("--window_type", type=str, choices=["sliding", "session"],
                        help="window for building log sequence")
    parser.add_argument("--session_level", type=str, choices=["entry", "hour"],
                        help="window for building log sequence")
    parser.add_argument('--window_size', default=5, type=float, help='window size(mins)')
    parser.add_argument('--step_size', default=1, type=float, help='step size(mins)')
    parser.add_argument('--train_size', default=0.4, type=float, help="train size", metavar="float or int")

    parser.add_argument("--train_ratio", default=1, type=float)
    parser.add_argument("--valid_ratio", default=0.1, type=float)
    parser.add_argument("--test_ratio", default=1, type=float)

    parser.add_argument("--max_epoch", default=200, type=int, help="epochs")
    parser.add_argument("--n_epochs_stop", default=10, type=int,
                        help="training stops after n epochs without improvement")
    parser.add_argument("--n_warm_up_epoch", default=10, type=int, help="save model parameters after n warm-up epoch")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")

    # features
    parser.add_argument("--is_logkey", action='store_true', help="is logkey included in features")
    parser.add_argument("--random_sample", action='store_true', help="is logkey included in features")
    parser.add_argument("--is_time", action='store_true', help="is time duration included in features")

    parser.add_argument("--min_freq", default=1, type=int, help="min frequency of logkey")
    # logbert
    parser.add_argument("--seq_len", default=10, type=int, help="max length of sequence")
    parser.add_argument("--min_len", default=10, type=int, help="min length of sequence")
    parser.add_argument("--max_len", default=512, type=int, help="for position embedding in bert")
    parser.add_argument("--mask_ratio", default=0.5, type=float, help="mask ratio in bert")
    parser.add_argument("--adaptive_window", action='store_true',
                        help="if true, window size is the length of sequences")

    parser.add_argument("--deepsvdd_loss", action='store_true', help="if calculate deepsvdd loss")
    parser.add_argument("--deepsvdd_loss_test", action='store_true', help="if use deepsvdd for prediction")

    parser.add_argument("--scale", default=None, help="sklearn normalization methods")

    parser.add_argument("--hidden", type=int, default=256, help="hidden size in logbert")
    parser.add_argument("--layers", default=4, type=int, help="number of layers in bert")
    parser.add_argument("--attn_heads", default=4, type=int, help="number of attention heads")

    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--adam_weight_decay", default=0.00, type=float)

    # deeplog, loganomaly & logrobust
    parser.add_argument("--sample", default="sliding_window", help="split sequences by sliding window")
    parser.add_argument("--history_size", default=10, type=int, help="window size for deeplog and log anomaly")
    parser.add_argument("--embeddings", default="embeddings.json", help="template embedding json file")

    # Features
    parser.add_argument("--sequentials", default=True, help="sequences of logkeys")
    parser.add_argument("--quantitatives", default=True, help="logkey count vector")
    parser.add_argument("--semantics", default=False, action='store_true', help="logkey embedding with semantics "
                                                                                "vectors")
    parser.add_argument("--parameters", default=False, help="include paramters in logs after parsing such time")

    parser.add_argument("--input_size", default=1, type=int, help="input size in lstm")
    parser.add_argument("--hidden_size", default=128, type=int, help="hidden size in lstm")
    parser.add_argument("--num_layers", default=2, type=int, help="num of lstm layers")
    parser.add_argument("--embedding_dim", default=300, type=int, help="embedding dimension of logkeys")

    parser.add_argument("--accumulation_step", default=1, type=int, help="let optimizer steps after several batches")
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--lr_decay_ratio", default=0.1, type=float)

    parser.add_argument("--num_candidates", default=9, type=int, help="top g candidates are normal")
    parser.add_argument("--log_freq", default=100, type=int, help="logging frequency of the batch iteration")
    parser.add_argument("--resume_path", action='store_true')

    parser.add_argument("--inject", action='store_true', help="is to inject unstable log")
    parser.add_argument("--inject_rate", default=30, type=int, help="injection rate")
    parser.add_argument("--inject_type", default="all", help="injection type either all, shuffle, duplicate, or delete" )
    parser.add_argument("--inject_path", default="", metavar="DIR", help="data directory")

    # neural_log
    parser.add_argument("--num_encoder_layers", default=1, type=int, help="number of encoder layers")
    parser.add_argument("--num_decoder_layers", default=1, type=int, help="number of decoder layers")
    parser.add_argument("--dim_model", default=300, type=int, help="model's dim")
    parser.add_argument("--num_heads", default=8, type=int, help="number of attention heads")
    parser.add_argument("--dim_feedforward", default=2048, type=int, help="feed-forward network's dim")
    parser.add_argument("--transformers_dropout", default=0.1, type=float, help="dropout rate of transformers model")
    
    parser.add_argument("--train_file_name", default="train_random_sampled.pkl", type=str)
    parser.add_argument("--max_dist", default=0.3, type=float, help="max_dist for logcluster")
    parser.add_argument("--anomaly_threshold", default=0.3, type=float, help="anomaly_threshold for logcluster")
    parser.add_argument("--num_bootstrap_samples", default=1000, type=int, help="num_bootstrap_samples for logcluster")

    parser.add_argument("--lightad_model", default="DT", type=str, choices = ["DT", "KNN", "SLFN"])
    parser.add_argument("--KNN_n", default=5, type=int, , help="number of neighbors for KNN")
    parser.add_argument("--KNN_metric", default='minkowski', type=str, , help="metric for KNN")
    parser.add_argument("--SLFN_hidden_neurons", default=100, type=int, help="hidden neurons for SLFN")
    parser.add_argument("--SLFN_solver", default='adam', type=str, help="solver for SLFN")
    parser.add_argument("--SLFN_activation", default='relu', type=str, help="activation for SLFN")
    parser.add_argument("--SLFN_max_iter", default=200, type=int, help="max_iter for SLFN")
    parser.add_argument("--SLFN_alpha", default=0.0001, type=float, help="alpha for SLFN")
    parser.add_argument("--SLFN_tol", default=0.0001, type=float, help="tol for SLFN")
    parser.add_argument("--DT_criterion", default='gini', type=str, help="the criterion for DT")
    parser.add_argument("--DT_min_samples_leaf", default=1, type=int, help="the min_samples_leaf for DT")
    parser.add_argument("--DT_max_depth", default=None, type=int, help="the max_depth for DT")
    parser.add_argument("--DT_min_samples_split", default=2, type=int, help="the min_samples_split for DT")
    
    return parser


def main():
    # seed_everything(seed=int(time.clock()))
    parser = arg_parser()
    args = parser.parse_args()

    args.inject_path = os.path.expanduser(args.data_dir + args.inject_path)

    args.data_dir = os.path.expanduser(args.data_dir + args.folder)

    args.output_dir += args.folder
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # sampling raw logs
    if args.sample_size is not None:
        sample_step_size = 10 ** 4
        sample_raw_data(args.data_dir + args.log_file, args.data_dir + args.sample_log_file, args.sample_size,
                        sample_step_size)
        args.log_file = args.sample_log_file

    # parse logs
    if args.parser_type is not None:
        args.log_format = " ".join([f"<{field}>" for field in args.log_format.split(",")])
        parse_log(args.data_dir, args.data_dir, args.log_file, args.parser_type, args.log_format, args.regex,
                  args.keep_para,
                  args.st, args.depth, args.max_child, args.tau)

    options = vars(args)
    if options['session_level'] == "entry":
        options["output_dir"] = options["output_dir"] + str(int(options["window_size"])) + "/"
    if args.is_process:
        if args.dataset_name in ["hdfs", "adfa", "hadoop", "spark"]:
            if not os.path.exists(os.path.join(options["output_dir"])):
                os.makedirs(os.path.join(options["output_dir"]), exist_ok=True)
            if args.dataset_name in ["hdfs"]:
                #shutil.copy(options["data_dir"] + "train_unique_sampled.pkl", os.path.join(options["output_dir"], "train.pkl"))
                shutil.copy(options["data_dir"] + options["train_file_name"], os.path.join(options["output_dir"], "train.pkl"))

            elif args.dataset_name in ["hadoop"]:
                #shutil.copy(options["data_dir"] + "hadoop2/train_unique_sampled.pkl", os.path.join(options["output_dir"], "train.pkl"))
                shutil.copy(options["data_dir"] + options["train_file_name"], os.path.join(options["output_dir"], "train.pkl"))
            
            elif args.dataset_name in ["spark"]:
                #shutil.copy(options["data_dir"] + "hadoop2/train_unique_sampled.pkl", os.path.join(options["output_dir"], "train.pkl"))
                shutil.copy(options["data_dir"] + options["train_file_name"], os.path.join(options["output_dir"], "train.pkl"))

            else:
                #shutil.copy(options["data_dir"] + "train_random_sampled.pkl", os.path.join(options["output_dir"], "train.pkl"))
                shutil.copy(options["data_dir"] + options["train_file_name"], os.path.join(options["output_dir"], "train.pkl"))

        else:
            process_dataset(data_dir=args.data_dir, output_dir=options["output_dir"], log_file=args.log_file,
                            dataset_name=args.dataset_name, window_type=args.window_type,
                            window_size=args.window_size, step_size=args.step_size,
                            train_size=args.train_size, random_sample=args.random_sample, session_type=args.session_level)

    if args.is_instance:
        process_instance(data_dir=args.data_dir, output_dir=args.output_dir, train_file=args.train_file,
                         test_file=args.test_file)

    # if options['session_level'] == "entry":
    #     options["output_dir"] = options["output_dir"] + str(options["window_size"]) + "/"
    options["model_dir"] = options["output_dir"] + options["model_name"] + "/"
    options["train_vocab"] = options["output_dir"] + "train.pkl"
    options["vocab_path"] = options["data_dir"] + options["model_name"] + "_vocab.pkl"  # pickle file
    #options["inject_path"] = options["data_dir"] + "inject.pkl"
    options["model_path"] = options["model_dir"] + options["model_name"] + ".pth"
    options["scale_path"] = options["model_dir"] + "scale.pkl"

    if not os.path.exists(options["model_dir"]):
        os.mkdir(options["model_dir"])

    if args.inject:
      print("Inject unstable logs")
      shutil.copy(options["inject_path"], os.path.join(options["output_dir"], "test.pkl"))
      #inject_window(options["output_dir"]+"test.pkl", options["inject_path"], injection_rate = args.inject_rate, inject_type = options["inject_type"])

    print("Save options parameters")
    save_parameters(options, options["model_dir"] + "parameters.txt")

    if args.model_name == "logbert":
        run_logbert(options)
    elif args.model_name == "deeplog":
        run_deeplog(options)
    elif args.model_name == "loganomaly":
        run_loganomaly(options)
    elif args.model_name == "logrobust":
        run_logrobust(options)
    elif args.model_name == "cnn":
        run_cnn(options)
    elif args.model_name == "plelog":
        run_plelog(options)
    elif args.model_name == "pca":
        run_pca(options)
    elif args.model_name == "logcluster":
        run_logcluster(options)
    elif args.model_name == "invariantsminer":
        run_invariantsminer(options)
    elif args.model_name == "neurallog":
        run_neuralog(options)
    elif args.model_name == "lightad":
        run_lightad(options)
    else:
        raise NotImplementedError(f"Model {args.model_name} is not defined")


if __name__ == "__main__":
    main()
