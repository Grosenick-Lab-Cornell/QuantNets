# Main class called to train different models from CNN, QGCN, QGRN, SGCN
import shutil
import os
import sys
import torch
import pickle
import torch.nn.functional as F
import torch
import torch_geometric
import time
import statistics
import matplotlib
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader as RawDataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader

from util.data_processing import *


# define a wrapper time_it decorator function
def time_it(func):
    def wrapper_function(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        stop = time.time()
        print(f'Function {func.__name__} took: {stop-start}s')
        return res
    return wrapper_function


class Experiment:
  # list of static variables ...
  experiment_id = 1 # static variable for creating experiments dir ...

  def __init__(self, 
               sgcn_model = None, 
               qgcn_model = None, 
               cnn_model = None,
               optim_params = None, 
               base_path = ".", 
               num_train = None,
               num_test = None,
               dataset_name = None,
               train_batch_size = 64,
               test_batch_size = 64,
               train_shuffle_data = True,
               test_shuffle_data = False,
               profile_run = False,
               image_to_graph_supported = False,
               walk_clock_num_runs = 10,
               id = None):
    
    # Controls whether we want to print runtime per model
    self.profile_run = profile_run
    self.walk_clock_num_runs = walk_clock_num_runs

    if image_to_graph_supported:
      # Graph Datasets
      geometric_qgcn_train_data = CustomImageGraphDataset(dataset_name=dataset_name, train=True)
      geometric_qgcn_test_data  = CustomImageGraphDataset(dataset_name=dataset_name, train=False)
      geometric_sgcn_train_data = CustomImageGraphDataset(dataset_name=dataset_name, train=True)
      geometric_sgcn_test_data  = CustomImageGraphDataset(dataset_name=dataset_name, train=False)

      # create a dataset class around datasets ...
      raw_train_data = (geometric_qgcn_train_data or geometric_sgcn_train_data).get_raw_image_dataset()
      raw_test_data  = (geometric_qgcn_test_data or geometric_sgcn_test_data).get_raw_image_dataset()

    else:
      # load in the dataset ...
      data_struct = read_cached_graph_dataset(num_train=num_train, num_test=num_test, dataset_name=dataset_name, parent_dir = base_path)

      # save the references to the datasets ...
      self.data_struct = data_struct
      raw_x_train_data = data_struct["raw"]["x_train_data"]
      raw_y_train_data = data_struct["raw"]["y_train_data"] 
      raw_x_test_data  = data_struct["raw"]["x_test_data"]  
      raw_y_test_data  = data_struct["raw"]["y_test_data"]

      # create a dataset class around datasets ...
      raw_train_data = ImageDatasetWrapper(raw_x_train_data, raw_y_train_data)
      raw_test_data  = ImageDatasetWrapper(raw_x_test_data, raw_y_test_data)

      # get the geometric data ...
      geometric_qgcn_train_data = data_struct["geometric"].get("gcn_train_data", None)
      geometric_qgcn_test_data  = data_struct["geometric"].get("gcn_test_data", None)
      geometric_qgcn_train_data = geometric_qgcn_train_data or data_struct["geometric"].get("qgcn_train_data", None)
      geometric_qgcn_test_data  = geometric_qgcn_test_data or data_struct["geometric"].get("qgcn_test_data", None)
      geometric_sgcn_train_data = data_struct["geometric"]["sgcn_train_data"]
      geometric_sgcn_test_data  = data_struct["geometric"]["sgcn_test_data"]
      # geometric_qgcn_train_data = torch_geometric.datasets.TUDataset(root="./data", name="PROTEINS", use_node_attr=True, use_edge_attr=True)
      # geometric_qgcn_test_data  = torch_geometric.datasets.TUDataset(root="./data", name="PROTEINS", use_node_attr=True, use_edge_attr=True)
      # geometric_sgcn_train_data = torch_geometric.datasets.TUDataset(root="./data", name="PROTEINS", use_node_attr=True, use_edge_attr=True)
      # geometric_sgcn_test_data  = torch_geometric.datasets.TUDataset(root="./data", name="PROTEINS", use_node_attr=True, use_edge_attr=True)
      # import pdb; pdb.set_trace()
      # torch_geometric.datasets.TUDataset(root="./data", name="PROTEINS", use_node_attr=True, use_edge_attr=True)

    # init the dataloaders ...
    shuffle_raw_train_data = (len(raw_train_data) != 0) and train_shuffle_data
    raw_train_loader = RawDataLoader(raw_train_data, 
                                    batch_size=train_batch_size, 
                                    shuffle=shuffle_raw_train_data)
    shuffle_raw_test_data = (len(raw_test_data) != 0) and test_shuffle_data
    raw_test_loader  = RawDataLoader(raw_test_data, 
                                    batch_size=test_batch_size, 
                                    shuffle=shuffle_raw_test_data)

    # init the dataloaders ...
    shuffle_qgcn_geo_train_data = (len(geometric_qgcn_train_data) != 0) and train_shuffle_data
    geometric_qgcn_train_loader = GraphDataLoader(geometric_qgcn_train_data, 
                                                  batch_size=train_batch_size, 
                                                  shuffle=shuffle_qgcn_geo_train_data,)
    shuffle_qgcn_geo_test_data  = (len(geometric_qgcn_test_data) != 0) and test_shuffle_data
    geometric_qgcn_test_loader  = GraphDataLoader(geometric_qgcn_test_data,
                                                  batch_size=test_batch_size,
                                                  shuffle=shuffle_qgcn_geo_test_data)
    shuffle_sgcn_geo_train_data = (len(geometric_sgcn_train_data) != 0) and train_shuffle_data
    geometric_sgcn_train_loader = GraphDataLoader(geometric_sgcn_train_data, 
                                                  batch_size=train_batch_size, 
                                                  shuffle=shuffle_sgcn_geo_train_data)
    shuffle_sgcn_geo_test_data  = (len(geometric_sgcn_test_data) != 0) and test_shuffle_data
    geometric_sgcn_test_loader  = GraphDataLoader(geometric_sgcn_test_data,
                                                  batch_size=test_batch_size,
                                                  shuffle=shuffle_sgcn_geo_test_data)
    
    print('raw dataset train: ', len(raw_train_loader.dataset))
    print('raw dataset test: ',  len(raw_test_loader.dataset))
    print('geometric qgcn dataset train: ', len(geometric_qgcn_train_loader.dataset))
    print('geometric qgcn dataset test: ', len(geometric_qgcn_test_loader.dataset))
    print('geometric sgcn dataset train: ', len(geometric_sgcn_train_loader.dataset))
    print('geometric sgcn dataset test: ', len(geometric_sgcn_test_loader.dataset))

    # save the ref here ...    
    self.raw_train_dataloader = raw_train_loader
    self.raw_test_dataloader = raw_test_loader
    self.sp_qgcn_train_dataloader = geometric_qgcn_train_loader
    self.sp_qgcn_test_dataloader = geometric_qgcn_test_loader
    self.sp_sgcn_train_dataloader = geometric_sgcn_train_loader
    self.sp_sgcn_test_dataloader = geometric_sgcn_test_loader

    # define what device we are using ...
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True

    # define the experiments folder and add the directory for this run ...
    self.cache_run = True
    self.specific_run_dir = None
    self.cnn_specific_run_dir = None
    self.qgcn_specific_run_dir = None
    self.sgcn_specific_run_dir = None
    if base_path == None:
      self.cache_run = False
    else:
      local_experiment_id = Experiment.experiment_id
      if id == None:
        Experiment.experiment_id += 1 # increment the id ...
      else:
        local_experiment_id = id
      self.local_experiment_id = local_experiment_id # save ref to experiment id ...
      # create the folder structure for this run ...
      self.__create_experiment_folder_structure(base_path, local_experiment_id)

    # if this is a cache_run then load models if they exist ...
    # NB: the run directories are used by the __load_models function below ...
    # loaded_cnn_model, loaded_qgcn_model, loaded_sgcn_model = self.__load_models() 
    loaded_cnn_model, loaded_qgcn_model, loaded_sgcn_model = None, None, None

    # # save the model inside experiments ...
    self.cnn_model = loaded_cnn_model
    self.qgcn_model = loaded_qgcn_model
    self.sgcn_model = loaded_sgcn_model
    if cnn_model != None:
      self.cnn_model = cnn_model
    if qgcn_model != None:
      self.qgcn_model = qgcn_model
    if sgcn_model != None:
      self.sgcn_model = sgcn_model

    # assert that at least one of the models if not None --> at least we train 1 model ...
    assert any([self.cnn_model, self.qgcn_model, self.sgcn_model]), "There must be at least 1 model to train"
    self.cnn_model_exists = cnn_model != None
    self.qgcn_model_exists = qgcn_model != None
    self.sgcn_model_exists = sgcn_model != None

    # put the model on the device: cuda or cpu ...
    if self.cnn_model_exists:
      self.cnn_model.to(self.device)
    if self.qgcn_model_exists:
      self.qgcn_model.to(self.device)
    if self.sgcn_model_exists:
      self.sgcn_model.to(self.device)

    # define the optimizers for the different models ...
    self.optim_params = optim_params
    learning_rate = 0.005 # default 
    if optim_params != None and "lr" in optim_params.keys():
      learning_rate = optim_params["lr"] 
    self.cnn_model_optimizer = None
    self.qgcn_model_optimizer = None
    self.sgcn_model_optimizer = None
    if self.cnn_model_exists:
      self.cnn_model_optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=learning_rate)
    if self.qgcn_model_exists:
      self.qgcn_model_optimizer = torch.optim.Adam(self.qgcn_model.parameters(), lr=learning_rate)
    if self.sgcn_model_exists:
      self.sgcn_model_optimizer = torch.optim.Adam(self.sgcn_model.parameters(), lr=learning_rate)

    # Print model flops and num parameters:
    if self.profile_run: self.__print_models_stats()
  
  
  def __print_models_stats(self):
    from flops_counter.ptflops import get_model_complexity_info
    if self.cnn_model_exists:
      data_sample = self.data_struct["raw"]["x_train_data"][0].clone().detach().unsqueeze(dim=0).to(self.device)
      model = self.cnn_model
      model.eval() # put model in eval mode so no params are updated
      macs, params = get_model_complexity_info(model, data_sample, as_strings=False, print_per_layer_stat=False, verbose=False)
      # Assuming 1 MAC = 0.5x FLOPs
      flops, macs, params = round(2*macs / 1e3, 3), round(macs / 1e3, 3), round(params / 1e3, 3)
      # Profile Inference Wall Time
      wall_times = []
      for _ in range(self.walk_clock_num_runs):
         start_time = time.time()
         _ = model(self.data_struct["raw"]["x_train_data"][0].clone().detach().unsqueeze(dim=0).to(self.device))
         end_time = time.time()
         wall_times.append(end_time - start_time)
      wall_time_mean = statistics.mean(wall_times)
      wall_time_std = statistics.stdev(wall_times)
      # print stat
      print("\n-----------------")
      print("CNN Model Stats:")
      print("-------------------------------------------------------------------------------------------")
      print(f'Number of parameters: {params} k')
      print(f'Theoretical Computational Complexity (FLOPs): {flops} kFLOPs')
      print(f'Theoretical Computational Complexity (MACs):  {macs} kMACs')
      print(f'Wall Clock  Computational Complexity (s):     {wall_time_mean} +/- {wall_time_std} s ')
      print("-------------------------------------------------------------------------------------------")
    
    if self.sgcn_model_exists:
      # pick the largest data (by node degree) to profile all the models
      crit_lst = [ data.x.numel() + data.edge_index.numel() for data in self.data_struct["geometric"]["sgcn_train_data"]]
      _, max_crit_index = max(crit_lst), crit_lst.index(max(crit_lst))
      data_sample = self.data_struct["geometric"]["sgcn_train_data"][max_crit_index].clone().detach().to(self.device) # get data at that index for profiling
      model = self.sgcn_model
      model.eval() # put model in eval mode so no params are updated
      macs, params = get_model_complexity_info(model, data_sample, as_strings=False, print_per_layer_stat=False, verbose=False)
      # Assuming 1 MAC = 0.5x FLOPs
      flops, macs, params = round(2*macs / 1e3, 3), round(macs / 1e3, 3), round(params / 1e3, 3)
      # Profile Inference Wall Time
      wall_times = []
      for _ in range(self.walk_clock_num_runs):
         start_time = time.time()
         _ = model(self.data_struct["geometric"]["sgcn_train_data"][max_crit_index].clone().detach().to(self.device))
         end_time = time.time()
         wall_times.append(end_time - start_time)
      wall_time_mean = statistics.mean(wall_times)
      wall_time_std = statistics.stdev(wall_times)
      # print stats
      print("\n-----------------")
      print("SGCN Model Stats:")
      print(f"Profiling data sample: {self.data_struct['geometric']['sgcn_train_data'][max_crit_index]}")
      print("-------------------------------------------------------------------------------------------")
      print(f'Number of parameters: {params} k')
      print(f'Theoretical Computational Complexity (FLOPs): {flops} kFLOPs')
      print(f'Theoretical Computational Complexity (MACs):  {macs} kMACs')
      print(f'Wall Clock  Computational Complexity (s):     {wall_time_mean} +/- {wall_time_std} s ')
      print("-------------------------------------------------------------------------------------------")

    if self.qgcn_model_exists:
      # pick the largest data (by node degree) to profile all the models
      crit_lst = [ data.x.numel() + data.edge_index.numel() for data in self.data_struct["geometric"]["qgcn_train_data"]]
      _, max_crit_index = max(crit_lst), crit_lst.index(max(crit_lst))
      data_sample = self.data_struct["geometric"]["qgcn_train_data"][max_crit_index].clone().detach().to(self.device) # get data at that index for profiling
      model = self.qgcn_model
      model.eval() # put model in eval mode so no params are updated
      macs, params = get_model_complexity_info(model, data_sample, as_strings=False, print_per_layer_stat=False, verbose=False)
      # Assuming 1 MAC = 0.5x FLOPs
      flops, macs, params = round(2*macs / 1e3, 3), round(macs / 1e3, 3), round(params / 1e3, 3)
      # Profile Inference Wall Time
      wall_times = []
      for _ in range(self.walk_clock_num_runs):
         start_time = time.time()
         _ = model(self.data_struct["geometric"]["qgcn_train_data"][max_crit_index].clone().detach().to(self.device))
         end_time = time.time()
         wall_times.append(end_time - start_time)
      wall_time_mean = statistics.mean(wall_times)
      wall_time_std = statistics.stdev(wall_times)
      # print stats
      print("\n-----------------")
      print("QGCN Model Stats:")
      print(f"Profiling data sample: {self.data_struct['geometric']['qgcn_train_data'][max_crit_index]}")
      print("-------------------------------------------------------------------------------------------")
      print(f'Number of parameters: {params} k')
      print(f'Theoretical Computational Complexity (FLOPs): {flops} kFLOPs')
      print(f'Theoretical Computational Complexity (MACs):  {macs} kMACs')
      print(f'Wall Clock  Computational Complexity (s):     {wall_time_mean} +/- {wall_time_std} s ')
      print("-------------------------------------------------------------------------------------------")


  def __load_models(self):
    loaded_cnn_model, loaded_qgcn_model, loaded_sgcn_model = None, None, None
    if self.cnn_specific_run_dir != None:
      cnn_model_filepath = os.path.join(self.cnn_specific_run_dir, "model.pth")
      if os.path.exists(cnn_model_filepath):
        loaded_cnn_model = torch.load(cnn_model_filepath)
    if self.qgcn_specific_run_dir != None:
      qgcn_model_filepath = os.path.join(self.qgcn_specific_run_dir, "model.pth")
      if os.path.exists(qgcn_model_filepath):
        loaded_qgcn_model = torch.load(qgcn_model_filepath)
    if self.sgcn_specific_run_dir != None:
      sgcn_model_filepath = os.path.join(self.sgcn_specific_run_dir, "model.pth")
      if os.path.exists(sgcn_model_filepath):
        loaded_sgcn_model = torch.load(sgcn_model_filepath)
    return loaded_cnn_model, loaded_qgcn_model, loaded_sgcn_model


  def __cache_models(self):
    if self.cnn_specific_run_dir != None:
      cnn_model_filepath = os.path.join(self.cnn_specific_run_dir, "model.pth")
      torch.save(self.cnn_model, cnn_model_filepath) # override if any exists ...
    if self.qgcn_specific_run_dir != None:
      qgcn_model_filepath = os.path.join(self.qgcn_specific_run_dir, "model.pth")
      torch.save(self.qgcn_model, qgcn_model_filepath) # override if any exists ...
    if self.sgcn_specific_run_dir != None:
      sgcn_model_filepath = os.path.join(self.sgcn_specific_run_dir, "model.pth")
      torch.save(self.sgcn_model, sgcn_model_filepath) # override if any exists ...


  def __cache_results(self, train_cnn_loss_array, train_qgcn_loss_array, train_sgcn_loss_array, 
                      train_cnn_acc_array, train_qgcn_acc_array, train_sgcn_acc_array,
                      test_cnn_acc_array, test_qgcn_acc_array, test_sgcn_acc_array):
    # define the paths to save the results ...
    if self.cnn_specific_run_dir != None:
      train_cnn_loss_filepath = os.path.join(self.cnn_specific_run_dir, "train_loss.pk")
      train_cnn_acc_filepath  = os.path.join(self.cnn_specific_run_dir, "train_acc.pk")
      test_cnn_acc_filepath   = os.path.join(self.cnn_specific_run_dir, "test_acc.pk")
      with open(train_cnn_loss_filepath, 'wb') as train_cnn_loss_file:
          pickle.dump(train_cnn_loss_array, train_cnn_loss_file)
      with open(train_cnn_acc_filepath, 'wb') as train_cnn_acc_file:
          pickle.dump(train_cnn_acc_array, train_cnn_acc_file)
      with open(test_cnn_acc_filepath, 'wb') as test_cnn_acc_file:
          pickle.dump(test_cnn_acc_array, test_cnn_acc_file)
    if self.qgcn_specific_run_dir != None:
      train_qgcn_loss_filepath = os.path.join(self.qgcn_specific_run_dir, "train_loss.pk")
      train_qgcn_acc_filepath  = os.path.join(self.qgcn_specific_run_dir, "train_acc.pk")
      test_qgcn_acc_filepath   = os.path.join(self.qgcn_specific_run_dir, "test_acc.pk")
      with open(train_qgcn_loss_filepath, 'wb') as train_qgcn_loss_file:
          pickle.dump(train_qgcn_loss_array, train_qgcn_loss_file)
      with open(train_qgcn_acc_filepath, 'wb') as train_qgcn_acc_file:
          pickle.dump(train_qgcn_acc_array, train_qgcn_acc_file)
      with open(test_qgcn_acc_filepath, 'wb') as test_qgcn_acc_file:
          pickle.dump(test_qgcn_acc_array, test_qgcn_acc_file)
    if self.sgcn_specific_run_dir != None:
      train_sgcn_loss_filepath = os.path.join(self.sgcn_specific_run_dir, "train_loss.pk")
      train_sgcn_acc_filepath  = os.path.join(self.sgcn_specific_run_dir, "train_acc.pk")
      test_sgcn_acc_filepath   = os.path.join(self.sgcn_specific_run_dir, "test_acc.pk")
      with open(train_sgcn_loss_filepath, 'wb') as train_sgcn_loss_file:
          pickle.dump(train_sgcn_loss_array, train_sgcn_loss_file)
      with open(train_sgcn_acc_filepath, 'wb') as train_sgcn_acc_file:
          pickle.dump(train_sgcn_acc_array, train_sgcn_acc_file)
      with open(test_sgcn_acc_filepath, 'wb') as test_sgcn_acc_file:
          pickle.dump(test_sgcn_acc_array, test_sgcn_acc_file)


  def load_cached_results(self):
    train_cnn_loss_array, train_qgcn_loss_array, train_sgcn_loss_array = None, None, None
    train_cnn_acc_array, train_qgcn_acc_array, train_sgcn_acc_array = None, None, None
    test_cnn_acc_array, test_qgcn_acc_array, test_sgcn_acc_array = None, None, None

    # define the paths to save the results ...
    if self.cnn_specific_run_dir != None and os.path.exists(self.cnn_specific_run_dir):
      train_cnn_loss_filepath = os.path.join(self.cnn_specific_run_dir, "train_loss.pk")
      train_cnn_acc_filepath  = os.path.join(self.cnn_specific_run_dir, "train_acc.pk")
      test_cnn_acc_filepath   = os.path.join(self.cnn_specific_run_dir, "test_acc.pk")
      with open(train_cnn_loss_filepath, 'rb') as train_cnn_loss_file:
          train_cnn_loss_array = pickle.load(train_cnn_loss_file)
      with open(train_cnn_acc_filepath, 'rb') as train_cnn_acc_file:
          train_cnn_acc_array = pickle.load(train_cnn_acc_file)
      with open(test_cnn_acc_filepath, 'rb') as test_cnn_acc_file:
          test_cnn_acc_array = pickle.load(test_cnn_acc_file)
    if self.qgcn_specific_run_dir != None and os.path.exists(self.qgcn_specific_run_dir):
      train_qgcn_loss_filepath = os.path.join(self.qgcn_specific_run_dir, "train_loss.pk")
      train_qgcn_acc_filepath  = os.path.join(self.qgcn_specific_run_dir, "train_acc.pk")
      test_qgcn_acc_filepath   = os.path.join(self.qgcn_specific_run_dir, "test_acc.pk")
      with open(train_qgcn_loss_filepath, 'rb') as train_qgcn_loss_file:
          train_qgcn_loss_array = pickle.load(train_qgcn_loss_file)
      with open(train_qgcn_acc_filepath, 'rb') as train_qgcn_acc_file:
          train_qgcn_acc_array = pickle.load(train_qgcn_acc_file)
      with open(test_qgcn_acc_filepath, 'rb') as test_qgcn_acc_file:
          test_qgcn_acc_array = pickle.load(test_qgcn_acc_file)
    if self.sgcn_specific_run_dir != None and os.path.exists(self.sgcn_specific_run_dir):
      train_sgcn_loss_filepath = os.path.join(self.sgcn_specific_run_dir, "train_loss.pk")
      train_sgcn_acc_filepath  = os.path.join(self.sgcn_specific_run_dir, "train_acc.pk")
      test_sgcn_acc_filepath   = os.path.join(self.sgcn_specific_run_dir, "test_acc.pk")
      with open(train_sgcn_loss_filepath, 'rb') as train_sgcn_loss_file:
          train_sgcn_loss_array = pickle.load(train_sgcn_loss_file)
      with open(train_sgcn_acc_filepath, 'rb') as train_sgcn_acc_file:
          train_sgcn_acc_array = pickle.load(train_sgcn_acc_file)
      with open(test_sgcn_acc_filepath, 'rb') as test_sgcn_acc_file:
          test_sgcn_acc_array = pickle.load(test_sgcn_acc_file)

    # return format ---> cnn, qgcn, sgcn ...
    return (train_cnn_loss_array, train_qgcn_loss_array, train_sgcn_loss_array, \
            train_cnn_acc_array, train_qgcn_acc_array, train_sgcn_acc_array, \
            test_cnn_acc_array, test_qgcn_acc_array, test_sgcn_acc_array)


  def __create_experiment_folder_structure(self, base_path, experiment_id):
    if not os.path.exists(base_path):
      print("Ensure that your base path exists -> {}".format(base_path))
      sys.exit(1)
    experiments_dir = os.path.join(base_path, "Experiments")
    if not os.path.exists(experiments_dir):
      os.mkdir(experiments_dir)
    underscored_experiment_id = "_".join(str(experiment_id).strip().split(" "))
    specific_run_dir = os.path.join(experiments_dir, "run_" + underscored_experiment_id)
    if not os.path.exists(specific_run_dir):
      os.mkdir(specific_run_dir)
    self.specific_run_dir = specific_run_dir

    # create the respective folders for this run ...
    cnn_specific_run_dir = os.path.join(specific_run_dir, "cnn")
    if not os.path.exists(cnn_specific_run_dir):
        os.mkdir(cnn_specific_run_dir)
    self.cnn_specific_run_dir = cnn_specific_run_dir
    qgcn_specific_run_dir = os.path.join(specific_run_dir, "qgcn")
    if not os.path.exists(qgcn_specific_run_dir):
        os.mkdir(qgcn_specific_run_dir)
    self.qgcn_specific_run_dir = qgcn_specific_run_dir
    sgcn_specific_run_dir = os.path.join(specific_run_dir, "sgcn")
    if not os.path.exists(sgcn_specific_run_dir):
        os.mkdir(sgcn_specific_run_dir)
    self.sgcn_specific_run_dir = sgcn_specific_run_dir

    # copy the architecture files into the relevant folders ...
    cnn_source_filepath = os.path.join(base_path, "cnn", "architectures.py")
    cnn_destination_filepath = os.path.join(self.cnn_specific_run_dir, "architectures.py")
    if not os.path.exists(cnn_destination_filepath):
      shutil.copyfile(cnn_source_filepath, cnn_destination_filepath)
    # copy the architecture files into the relevant folders ...
    qgcn_source_filepath = os.path.join(base_path, "qgcn", "architectures.py")
    qgcn_destination_filepath = os.path.join(self.qgcn_specific_run_dir, "architectures.py")
    if not os.path.exists(qgcn_destination_filepath):
      shutil.copyfile(qgcn_source_filepath, qgcn_destination_filepath)
    # copy the architecture files into the relevant folders ...
    sgcn_source_filepath = os.path.join(base_path, "sgcn", "src", "architectures.py")
    sgcn_destination_filepath = os.path.join(self.sgcn_specific_run_dir, "architectures.py")
    if not os.path.exists(sgcn_destination_filepath):
      shutil.copyfile(sgcn_source_filepath, sgcn_destination_filepath)


  # Moves the graph data to target device
  # Expects Data to have keys in list[x, y, pos, batch, edge_index]
  def __move_graph_data_to_device(self, data):
    if hasattr(data, 'x') and data.x is not None:
      data.x = data.x.to(self.device)
    if hasattr(data, 'edge_index') and data.edge_index is not None:
      data.edge_index = data.edge_index.to(self.device)
    if hasattr(data, 'y') and data.y is not None:
      data.y = data.y.to(self.device)
    if hasattr(data, 'pos') and data.pos is not None:
      data.pos = data.pos.to(self.device)
    if hasattr(data, 'batch') and data.batch is not None:
      data.batch = data.batch.to(self.device)
    return data


  # Trains models sequentially
  # CNN -> QGCN -> SGCN
  def __train(self):
    # put all the models in training mode ...
    # for the cnn training ...
    cnn_loss_all, cnn_total_graphs = 0, 0
    if self.cnn_model_exists:
      self.cnn_model.train()
      if self.profile_run: start_time = time.time()
      for data in self.raw_train_dataloader:
        x_data, y_data = data
        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)
        cnn_total_graphs += len(x_data)
        # zero out the gradients in the optim object ...
        # forward pass through the model ...
        # compute loss ...
        # update gradients ...
        self.cnn_model_optimizer.zero_grad()
        cnn_output = self.cnn_model(x_data)
        cnn_loss = F.cross_entropy(cnn_output, y_data.squeeze())
        cnn_loss.backward()
        cnn_loss_all += len(x_data) * cnn_loss.item()
        self.cnn_model_optimizer.step()
      if self.profile_run: stop_time = time.time()
      if self.profile_run: profile_stats = f"; Epoch took a total of {stop_time - start_time}s"
      print(f"Single epoch training... cnn_model done{ profile_stats if self.profile_run else '' }")
    
    # for the qgcn training ...
    qgcn_loss_all, qgcn_total_graphs = 0, 0
    if self.qgcn_model_exists:
      self.qgcn_model.train()
      # begin training ...
      if self.profile_run: start_time = time.time()
      for data in self.sp_qgcn_train_dataloader:
        qgcn_data = self.__move_graph_data_to_device(data)
        qgcn_total_graphs += qgcn_data.num_graphs
        # zero out the gradients in the optim object ...
        # forward pass through the model ...
        # compute loss ...
        # update gradients ...
        self.qgcn_model_optimizer.zero_grad()
        qgcn_output = self.qgcn_model(qgcn_data)
        qgcn_loss = F.cross_entropy(qgcn_output, qgcn_data.y)
        qgcn_loss.backward()
        qgcn_loss_all += qgcn_data.num_graphs * qgcn_loss.item()
        self.qgcn_model_optimizer.step()
        # print("optimizer done")
      if self.profile_run: stop_time = time.time()
      if self.profile_run: profile_stats = f"; Epoch took a total of {stop_time - start_time}s"
      print(f"Single epoch training... qgcn_model done{ profile_stats if self.profile_run else '' }")

    # for the sgcn training ...
    sgcn_loss_all, sgcn_total_graphs = 0, 0
    if self.sgcn_model_exists:
      self.sgcn_model.train()
      # begin training ...
      if self.profile_run: start_time = time.time()
      for data in self.sp_sgcn_train_dataloader:
        sgcn_data = self.__move_graph_data_to_device(data)
        sgcn_total_graphs += sgcn_data.num_graphs
        # zero out the gradients in the optim object ...
        # forward pass through the model ...
        # compute loss ...
        # update gradients ...
        self.sgcn_model_optimizer.zero_grad()
        sgcn_output = self.sgcn_model(sgcn_data)
        sgcn_loss = F.cross_entropy(sgcn_output, sgcn_data.y)
        sgcn_loss.backward()
        sgcn_loss_all += sgcn_data.num_graphs * sgcn_loss.item()
        self.sgcn_model_optimizer.step()
      if self.profile_run: stop_time = time.time()
      if self.profile_run: profile_stats = f"; Epoch took a total of {stop_time - start_time}s"
      print(f"Single epoch training... sgcn_model done{ profile_stats if self.profile_run else '' }")
      
    # if we want to cache the run, then cache them here ...
    # if self.cache_run:
    #   self.__cache_models() # cache the models ...

    # normalize loss by the total length of training set ...
    if self.cnn_model_exists:
      cnn_loss_all /= cnn_total_graphs
    if self.qgcn_model_exists:
      qgcn_loss_all /= qgcn_total_graphs
    if self.sgcn_model_exists:
      sgcn_loss_all /= sgcn_total_graphs

    # output provided in the form -> cnn, qgcn, sgcn ...
    return cnn_loss_all, qgcn_loss_all, sgcn_loss_all


  def __evaluate(self, eval_train_data=False):
    raw_dataset_loader = None
    sp_qgcn_dataset_loader = None 
    sp_sgcn_dataset_loader = None
    # put all the models in evaluation mode ...
    cnn_correct = 0 
    if self.cnn_model_exists:
      if eval_train_data:
        raw_dataset_loader = self.raw_train_dataloader
      else:
        raw_dataset_loader = self.raw_test_dataloader
      # set model in eval mode ...
      self.cnn_model.eval()
      # begin evaluation ...
      if self.profile_run: start_time = time.time()
      for data in raw_dataset_loader:
        x_data, y_data = data
        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)
        pred = self.cnn_model(x_data).max(dim=1)[1].reshape((-1,1))
        cnn_correct += pred.eq(y_data).sum().item()
      if self.profile_run: stop_time = time.time()
      if self.profile_run: profile_stats = f"; Epoch took a total of {stop_time - start_time}s"
      print(f"{'train' if eval_train_data else 'test'} data: cnn eval done{ profile_stats if self.profile_run else '' }")

    # perform the evaluation for the pixels ...
    qgcn_correct = 0
    if self.qgcn_model_exists: 
      if eval_train_data:
        sp_qgcn_dataset_loader = self.sp_qgcn_train_dataloader
      else:
        sp_qgcn_dataset_loader = self.sp_qgcn_test_dataloader
      self.qgcn_model.eval()
      # begin evaluation ...
      if self.profile_run: start_time = time.time()
      for data in sp_qgcn_dataset_loader:
        qgcn_data = self.__move_graph_data_to_device(data)
        pred = self.qgcn_model(qgcn_data).max(dim=1)[1]
        qgcn_correct += pred.eq(qgcn_data.y).sum().item()
      if self.profile_run: stop_time = time.time()
      if self.profile_run: profile_stats = f"; Epoch took a total of {stop_time - start_time}s"
      print(f"{'train' if eval_train_data else 'test'} data: qgcn eval done{ profile_stats if self.profile_run else '' }")

    # perform the evaluation for the pixels ...
    sgcn_correct = 0
    if self.sgcn_model_exists: 
      if eval_train_data:
        sp_sgcn_dataset_loader = self.sp_sgcn_train_dataloader
      else:
        sp_sgcn_dataset_loader = self.sp_sgcn_test_dataloader
      self.sgcn_model.eval()
      # begin evaluation ...
      if self.profile_run: start_time = time.time()
      for data in sp_sgcn_dataset_loader:
        sgcn_data = self.__move_graph_data_to_device(data)
        pred = self.sgcn_model(sgcn_data).max(dim=1)[1]
        sgcn_correct += pred.eq(sgcn_data.y).sum().item()
      if self.profile_run: stop_time = time.time()
      if self.profile_run: profile_stats = f"; Epoch took a total of {stop_time - start_time}s"
      print(f"{'train' if eval_train_data else 'test'} data: sgcn eval done{ profile_stats if self.profile_run else '' }")
    
    # return total num correct as a percentage [fraction] ...
    if self.cnn_model_exists:
      cnn_correct /= len(raw_dataset_loader.dataset)
    if self.qgcn_model_exists:
      qgcn_correct /= len(sp_qgcn_dataset_loader.dataset)
    if self.sgcn_model_exists:
      sgcn_correct /= len(sp_sgcn_dataset_loader.dataset)

    # output provided in the form -> cnn, qgcn, sgcn ...
    return cnn_correct, qgcn_correct, sgcn_correct


  @time_it
  def run(self, num_epochs=None, eval_training_set=True):
    if num_epochs == None or num_epochs <= 0:
      print("num_epochs ({}) in [Experiments.run] is invalid".format(num_epochs))
      sys.exit(1)

    # define the variables to hold to the stats ...
    test_cnn_acc_array,  test_qgcn_acc_array,  test_sgcn_acc_array  = [], [], []
    train_cnn_acc_array, train_qgcn_acc_array, train_sgcn_acc_array = [], [], []
    train_cnn_loss_array, train_qgcn_loss_array, train_sgcn_loss_array = [], [], []
    for epoch in range(1, num_epochs):
        # Time epoch ops
        start_time = time.time()
        print("training... epoch {}".format(epoch))
        cnn_loss, qgcn_loss, sgcn_loss = self.__train()
        train_cnn_loss_array.append(cnn_loss)
        train_qgcn_loss_array.append(qgcn_loss)
        train_sgcn_loss_array.append(sgcn_loss)
        train_cnn_acc, train_qgcn_acc, train_sgcn_acc = 0, 0, 0
        if eval_training_set:
            train_cnn_acc, train_qgcn_acc, train_sgcn_acc = self.__evaluate(eval_train_data=True)
        train_cnn_acc_array.append(train_cnn_acc)
        train_qgcn_acc_array.append(train_qgcn_acc)
        train_sgcn_acc_array.append(train_sgcn_acc)
        test_cnn_acc,  test_qgcn_acc,  test_sgcn_acc = self.__evaluate(eval_train_data=False)
        test_cnn_acc_array.append(test_cnn_acc)
        test_qgcn_acc_array.append(test_qgcn_acc)
        test_sgcn_acc_array.append(test_sgcn_acc)
        stop_time = time.time()

        # build the display string ...
        epoch_str = "Epoch: {:03d}, ".format(epoch)
        loss_str = "CNN_Loss: {:.5f}, QGCN_Loss: {:.5f}, SGCN_Loss: {:.5f}, ".format(cnn_loss, qgcn_loss, sgcn_loss)
        train_acc_str = "CNN_Train_Acc: {:.5f}, QGCN_Train_Acc: {:.5f}, SGCN_Train Acc: {:.5f}, ".format(train_cnn_acc, train_qgcn_acc, train_sgcn_acc)
        test_acc_str = "CNN_Test_Acc: {:.5f}, QGCN_Test_Acc: {:.5f}, SGCN_Test_Acc: {:.5f}, ".format(test_cnn_acc,  test_qgcn_acc,  test_sgcn_acc)
        
        # print out the results ...
        print("{}".format("".join([epoch_str, loss_str, train_acc_str, test_acc_str])))
        print(f"Epoch took a total of {stop_time - start_time}s")

    # cache results if we need it ...
    if self.cache_run:
      self.__cache_results( train_cnn_loss_array, train_qgcn_loss_array, train_sgcn_loss_array,
                            train_cnn_acc_array, train_qgcn_acc_array, train_sgcn_acc_array,
                            test_cnn_acc_array, test_qgcn_acc_array, test_sgcn_acc_array)


  # plots both accuracies and losses on the same graph ...
  @staticmethod
  def plot_history(data, labels):
    # override the font size of plotting tool
    font = {'weight':'bold', 'size':8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams.update({'font.size':8})

    indicators = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    leftover_count = len(data) % 2
    num_rows  = len(data) // 2
    col_count = 2
    row_count = num_rows + leftover_count
    fig, ax = plt.subplots(nrows=row_count, ncols=col_count)
    fig.tight_layout(pad=0.8)
    plt.subplots_adjust(wspace=0.4, hspace=0.5)

    # loop and plot the data and their labels ...
    for i, row_plts in enumerate(ax):
      for j, row_col_plt in enumerate(row_plts):
        data_index = i * col_count + j
        xdata = list(range(1, len(data[data_index]) + 1))
        ydata = data[data_index]
        data_label = labels[data_index]
        data_indicator = indicators[data_index]
        row_col_plt.plot(xdata, ydata, color=data_indicator, label=data_label)
        row_col_plt.set_xticks(xdata)
        row_col_plt.set_ylim(0, 1)
        row_col_plt.legend(loc="lower right")
        row_col_plt.set_xlabel('Epoch')
        row_col_plt.set_ylabel(data_label)
        row_col_plt.set_title('{} vs. No. of epochs'.format(data_label))
    # show the plot ...
    plt.show()
