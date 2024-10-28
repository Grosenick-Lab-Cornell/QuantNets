import os
import sys
import math
import statistics
import wget
import yaml
import torch

from copy import deepcopy
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# Empty cache
torch.cuda.empty_cache()

# current file directory
current_file_dirpath = os.path.dirname(os.path.realpath('__file__'))

# device type for all models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using cuda: ", torch.cuda.is_available(), "device")

# Add parent directory to sys.path
def get_root_dir(base: str = "."):
    if any([ os.path.isdir(os.path.join(base, child_dir)) and child_dir == "qgrn" for child_dir in os.listdir(base) ]):
        return base
    return get_root_dir(base=str(Path(base).parent.resolve()))

root_dir = get_root_dir()
sys.path.append(root_dir)

# Load in relevant modules
from qgrn.architectures import QGRN
from gnn.architectures import GCNConvNet
from graph_classification.experiment import Experiment


"""
Read in the datasets, splits and run settings
"""
def read_input_file(filename: str):
    import yaml

    # Open the YAML file
    with open(f'{filename}', 'r') as file:
        data = yaml.safe_load(file)

    return data

"""
Confirms whether dataset exists else downloads from dropbox
"""
def check_dataset_split_exists_else_download(dataset_split: dict, selected_dataset_config: dict):  
    dataset_par_dirpath = os.path.join(root_dir, "Dataset")
    if not os.path.exists(dataset_par_dirpath):
        os.mkdir(dataset_par_dirpath)
    rawgraph_subfolder_dirpath = os.path.join(dataset_par_dirpath, "RawGraph")
    if not os.path.exists(rawgraph_subfolder_dirpath):
        os.mkdir(rawgraph_subfolder_dirpath)
    # extract the dataset split
    train_size = dataset_split["train"]
    test_size = dataset_split["test"]
    dataset_split_folder = os.path.join(rawgraph_subfolder_dirpath, f"train_{train_size}_test_{test_size}")
    if not os.path.exists(dataset_split_folder):
        os.mkdir(dataset_split_folder)
    # get the filename and constr full filepath for downloaded file
    full_filepath = os.path.join(dataset_split_folder, f"{selected_dataset_config['dataset_name']}.pkl")
    if not os.path.exists(full_filepath):
        try:
            # Otherwise: attempt to download prepared pkts
            print(f"Dataset {selected_dataset_config['dataset_name'].upper()} for split: train-{train_size}, test-{test_size} doesn't exist")
            print(f"Downloading dataset ...")
            # Else: download the file before running experiment
            full_data_url = selected_dataset_config["download_url"].get(f"train_{train_size}_test_{test_size}", None)
            if full_data_url is None: 
                # if image_to_graph conversion exists, then we default to that
                if selected_dataset_config.get("image_to_graph_supported", False):
                    return True
                else: 
                    return False
            # Attempt downloading dataset
            wget.download(full_data_url, full_filepath)
            print(f"\nDownload complete ...")
            # Post-process
            if selected_dataset_config.get("image_to_graph_supported", False): selected_dataset_config["image_to_graph_supported"] = False
            return True
        except:
            print("Couldn't download dataset! Handing control to parent function")
            return False
    # Post-process
    if selected_dataset_config.get("image_to_graph_supported", False): selected_dataset_config["image_to_graph_supported"] = False
    # Return state
    return True

"""
Helper function for collating results
Define function for handling collation
"""
def collate_stats(stats_name, max_stats, smoothened_stats):
  # collate the results to cache
  collated_stats_keys = [ f"{stats_name}_max_of_maxs",
                          f"{stats_name}_avg_of_maxs",
                          f"{stats_name}_std_of_maxs",
                          f"{stats_name}_max_of_smaxs",
                          f"{stats_name}_avg_of_smaxs",
                          f"{stats_name}_std_of_smaxs"  ]
  cnn_collated_results = { x: 0 for x in collated_stats_keys} 
  sgcn_collated_results = { x: 0 for x in collated_stats_keys} 
  qgcn_collated_results = { x: 0 for x in collated_stats_keys} 

  # save the results
  cnn_collated_results[f"{stats_name}_max_of_maxs"]  = round(max(max_stats["cnn"]), 5)
  cnn_collated_results[f"{stats_name}_max_of_smaxs"] = round(max(smoothened_stats["cnn"]), 5)
  cnn_collated_results[f"{stats_name}_avg_of_maxs"]  = round(statistics.mean(max_stats["cnn"]), 5)
  cnn_collated_results[f"{stats_name}_avg_of_smaxs"] = round(statistics.mean(smoothened_stats["cnn"]), 5)
  cnn_collated_results[f"{stats_name}_std_of_maxs"]  = round(0 if (len(max_stats["cnn"]) <= 1) else statistics.stdev(max_stats["cnn"]), 5)
  cnn_collated_results[f"{stats_name}_std_of_smaxs"] = round(0 if (len(smoothened_stats["cnn"]) <= 1) else statistics.stdev(smoothened_stats["cnn"]), 5)

  sgcn_collated_results[f"{stats_name}_max_of_maxs"]  = round(max(max_stats["sgcn"]), 5)
  sgcn_collated_results[f"{stats_name}_max_of_smaxs"] = round(max(smoothened_stats["sgcn"]), 5)
  sgcn_collated_results[f"{stats_name}_avg_of_maxs"]  = round(statistics.mean(max_stats["sgcn"]), 5)
  sgcn_collated_results[f"{stats_name}_avg_of_smaxs"] = round(statistics.mean(smoothened_stats["sgcn"]), 5)
  sgcn_collated_results[f"{stats_name}_std_of_maxs"]  = round(0 if (len(max_stats["sgcn"]) <= 1) else statistics.stdev(max_stats["sgcn"]), 5)
  sgcn_collated_results[f"{stats_name}_std_of_smaxs"] = round(0 if (len(smoothened_stats["sgcn"]) <= 1) else statistics.stdev(smoothened_stats["sgcn"]), 5)

  qgcn_collated_results[f"{stats_name}_max_of_maxs"]  = round(max(max_stats["qgcn"]), 5)
  qgcn_collated_results[f"{stats_name}_max_of_smaxs"] = round(max(smoothened_stats["qgcn"]), 5)
  qgcn_collated_results[f"{stats_name}_avg_of_maxs"]  = round(statistics.mean(max_stats["qgcn"]), 5)
  qgcn_collated_results[f"{stats_name}_avg_of_smaxs"] = round(statistics.mean(smoothened_stats["qgcn"]), 5)
  qgcn_collated_results[f"{stats_name}_std_of_maxs"]  = round(0 if (len(max_stats["qgcn"]) <= 1) else statistics.stdev(max_stats["qgcn"]), 5)
  qgcn_collated_results[f"{stats_name}_std_of_smaxs"] = round(0 if (len(smoothened_stats["qgcn"]) <= 1) else statistics.stdev(smoothened_stats["qgcn"]), 5)

  # Return results
  return cnn_collated_results, sgcn_collated_results, qgcn_collated_results

"""
Main function that runs experiments
Initiates experiments run on standard vs custom vs all datasets
"""
def run_experiments(dataset_groups: list[str] = ['standard', 'custom'], average_smoothening_width: float = 0.05, notraineval: bool = False, profile_models: bool = True):
    """
    SWEEPING Logic below
    Loops through the different sweep parameters to train different models
    """
    # Load run details
    dataset_mapping = read_input_file(filename="datasets.yaml")
    datasets = read_input_file(filename="data.splits.yaml") 
    lrs = read_input_file(filename="run.settings.yaml")["lrs"]
    runs = read_input_file(filename="run.settings.yaml")["runs"]
    epochs = read_input_file(filename="run.settings.yaml")["epochs"]

    # Run data collection loop
    for dataset_group in dataset_groups:
        for dataset_split in datasets[dataset_group]: # loop over datasets
            # extract batch size which is peculiar to dataset split
            train_size = dataset_split.get('train', 0)
            test_size  = dataset_split.get('test', 0)
            batch_size = dataset_split.get('batch_size', 64)
            
            print(f"Dataset stats: train-{train_size}, test-{test_size}, batch_size-{batch_size}")
            # Inner loop goes over all datasets
            for selected_dataset, selected_dataset_config in dataset_mapping.items():
                # Skip all datasets that do not match the dataset group key
                selected_dataset_config = deepcopy(selected_dataset_config)
                if selected_dataset_config['dataset_group'] != dataset_group: continue

                # Check if dataset exists, if not then download
                if not check_dataset_split_exists_else_download(dataset_split, selected_dataset_config): continue
                
                # Prep experiment name
                experiment_name = f"BATCH-RESULTS-ALL-DATASETS-{selected_dataset.capitalize()}_GCN_QGRN_Summary"
                experiment_result_filepath = os.path.join(os.path.join(root_dir, "Experiments"), f'{"_".join(experiment_name.split(" "))}.yaml')
                results = {} # to hold results for saving
                if os.path.exists(experiment_result_filepath):
                    with open(experiment_result_filepath, "r") as file_stream:
                        results = yaml.safe_load(file_stream)
                        if results:
                            results = dict(results)
                        else:
                            results = {}
                            
                # Load the required params for this dataset
                dataset_name       = selected_dataset_config["dataset_name"]
                layers_num         = selected_dataset_config["layers_num"]
                out_dim            = selected_dataset_config["out_dim"]
                in_channels        = selected_dataset_config["in_channels"]
                hidden_channels    = selected_dataset_config["hidden_channels"]
                out_channels       = selected_dataset_config["out_channels"]
                num_sub_kernels    = selected_dataset_config["num_sub_kernels"]
                edge_attr_dim      = selected_dataset_config["edge_attr_dim"]
                pos_descr_dim      = selected_dataset_config["pos_descr_dim"]

                # Model specific params
                gcn_out_sf      = selected_dataset_config.get("gcn_out_sf", 1)
                gcn_hidden_sf   = selected_dataset_config.get("gcn_hidden_sf", 1)

                print(f"Selected Dataset: {selected_dataset.upper()}")
            
                # Inner-Inner loop
                for i, lr in enumerate(lrs[dataset_group]): # loop over learning rates
                    optim_params = { "lr": lr }
                    num_epochs   = epochs[dataset_group][i]
                    num_runs     = runs[dataset_group][i]
                    # create the key for hashing into results
                    results_hash_key = f'train_{dataset_split["train"]}_test_{dataset_split["test"]}_lr_{lr}'
                    results[results_hash_key] = {}
                    # run stats
                    mean_train_loss = { "cnn": [], "sgcn": [], "qgcn": []}
                    smoothened_train_loss = { "cnn": [], "sgcn": [], "qgcn": []}
                    max_train_acc = { "cnn": [], "sgcn": [], "qgcn": []}
                    smoothened_train_acc = { "cnn": [], "sgcn": [], "qgcn": []}
                    max_test_acc = { "cnn": [], "sgcn": [], "qgcn": []}
                    smoothened_test_acc = { "cnn": [], "sgcn": [], "qgcn": []}

                    for run in range(num_runs): # loop over num runs
                        # initialize the models
                        gcnconv_model = GCNConvNet(out_dim=out_dim, 
                                                   input_features=in_channels, 
                                                   output_channels=out_channels, 
                                                   layers_num=layers_num, 
                                                   model_dim=hidden_channels,
                                                   hidden_sf=gcn_hidden_sf, 
                                                   out_sf=gcn_out_sf)
                        
                        qgcn_model = QGRN(out_dim=out_dim,
                                          in_channels=in_channels,
                                          hidden_channels=hidden_channels,
                                          out_channels=out_channels,
                                          layers_num=layers_num,
                                          num_sub_kernels=num_sub_kernels,
                                          edge_attr_dim=edge_attr_dim,
                                          pos_descr_dim=pos_descr_dim,
                                          device=device)
                        
                        # setup experiments to run
                        num_train, num_test = dataset_split["train"], dataset_split["test"]
                        experiment_id = f'_{run}_train_{num_train}_test_{num_test}_lr_{lr}_num_epochs_{num_epochs}'
                        experiment = Experiment(sgcn_model = gcnconv_model,
                                                qgcn_model = qgcn_model,
                                                cnn_model = None,
                                                optim_params = optim_params,
                                                base_path = root_dir, 
                                                num_train = num_train,
                                                num_test = num_test,
                                                dataset_name = dataset_name,
                                                train_batch_size = batch_size,
                                                test_batch_size = batch_size,
                                                train_shuffle_data = True,
                                                test_shuffle_data = False,
                                                profile_run=profile_models,
                                                id = experiment_id) # mark this experiment ...
                        
                        if profile_models: break

                        # run the experiment ...
                        experiment.run(num_epochs=num_epochs, eval_training_set=(not notraineval)) # specify num epochs ...

                        # load collected stats during runs ...
                        (train_cnn_loss_array, train_qgcn_loss_array, train_sgcn_loss_array, \
                        train_cnn_acc_array, train_qgcn_acc_array, train_sgcn_acc_array, \
                        test_cnn_acc_array, test_qgcn_acc_array, test_sgcn_acc_array) = experiment.load_cached_results() # only accuracies on train and test sets ...
                        
                        # get the mean stats
                        mean_train_loss["cnn"].append(statistics.mean(train_cnn_loss_array))
                        mean_train_loss["sgcn"].append(statistics.mean(train_sgcn_loss_array))
                        mean_train_loss["qgcn"].append(statistics.mean(train_qgcn_loss_array))
                        
                        max_train_acc["cnn"].append(max(train_cnn_acc_array))
                        max_train_acc["sgcn"].append(max(train_sgcn_acc_array))
                        max_train_acc["qgcn"].append(max(train_qgcn_acc_array))
                        
                        max_test_acc["cnn"].append(max(test_cnn_acc_array))
                        max_test_acc["sgcn"].append(max(test_sgcn_acc_array))
                        max_test_acc["qgcn"].append(max(test_qgcn_acc_array))

                        # get the smoothened max test acc
                        num_averaging_window = int(math.ceil(average_smoothening_width * num_epochs))
                        smoothened_train_loss["cnn"].append(statistics.mean(train_cnn_loss_array[-num_averaging_window:]))
                        smoothened_train_loss["sgcn"].append(statistics.mean(train_sgcn_loss_array[-num_averaging_window:]))
                        smoothened_train_loss["qgcn"].append(statistics.mean(train_qgcn_loss_array[-num_averaging_window:]))
                        
                        smoothened_train_acc["cnn"].append(statistics.mean(train_cnn_acc_array[-num_averaging_window:]))
                        smoothened_train_acc["sgcn"].append(statistics.mean(train_sgcn_acc_array[-num_averaging_window:]))
                        smoothened_train_acc["qgcn"].append(statistics.mean(train_qgcn_acc_array[-num_averaging_window:]))
                        
                        smoothened_test_acc["cnn"].append(statistics.mean(test_cnn_acc_array[-num_averaging_window:]))
                        smoothened_test_acc["sgcn"].append(statistics.mean(test_sgcn_acc_array[-num_averaging_window:]))
                        smoothened_test_acc["qgcn"].append(statistics.mean(test_qgcn_acc_array[-num_averaging_window:]))
                    
                    if profile_models: break

                    # get collated stats
                    train_loss_cnn_results, train_loss_sgcn_results, train_loss_qgcn_results = collate_stats("train_loss", mean_train_loss, smoothened_train_loss)
                    train_acc_cnn_results,  train_acc_sgcn_results,  train_acc_qgcn_results  = collate_stats("train_acc", max_train_acc, smoothened_train_acc)
                    test_acc_cnn_results,   test_acc_sgcn_results,   test_acc_qgcn_results   = collate_stats("test_acc", max_test_acc, smoothened_test_acc)
                    all_cnn_stats  = {**train_loss_cnn_results,  **train_acc_cnn_results,  **test_acc_cnn_results}
                    all_sgcn_stats = {**train_loss_sgcn_results, **train_acc_sgcn_results, **test_acc_sgcn_results}
                    all_qgcn_stats = {**train_loss_qgcn_results, **train_acc_qgcn_results, **test_acc_qgcn_results}

                    # save results into results obj
                    results[results_hash_key]["cnn"] = all_cnn_stats
                    results[results_hash_key]["sgcn"] = all_sgcn_stats
                    results[results_hash_key]["qgcn"] = all_qgcn_stats

                    # pickle the results
                    with open(experiment_result_filepath, "w") as file_stream:
                        yaml.dump(results, file_stream)

        if profile_models: break


"""
Args prep
"""
import argparse
parser = argparse.ArgumentParser(description='Reproduce experimental results for CNN vs QGCN vs QGRN on standard and custom datasets')
parser.add_argument('-d', '--dataset', required=True, help='Type of dataset, i.e., standard / custom / all, to run experiment on')
parser.add_argument('--notraineval', action="store_true", default=False, help='Disables evaluating models for training accuracy')
parser.add_argument('--profilemodels', action="store_true", default=False, help='Enables profiling models for speed (model inference wall-clock)')
parser.add_argument('--avgsmoothwidth', default=0.05, help='Sets the %-width of window over which collected perf stats should be averaged')

args = parser.parse_args()
notraineval = args.notraineval
profile_models = args.profilemodels
average_smoothening_width = args.avgsmoothwidth
dataset = args.dataset.strip().lower()
assert any([ dataset == t for t in ['standard', 'custom', 'all'] ])
dataset_groups = ['standard', 'custom'] if dataset.lower() == 'all' else [dataset]

# Validate dataset type is correct
print(f"Running experiments on {dataset.capitalize()} Datasets")
run_experiments(dataset_groups=dataset_groups, average_smoothening_width=average_smoothening_width, notraineval=notraineval, profile_models=profile_models)
