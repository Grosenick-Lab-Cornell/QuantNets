##########################################################
#                      MNIST RAW DATASET                 #
##########################################################
import os
import numpy as np
import torch
from torch.utils.data import Dataset as RawDataset
import struct
from array import array
import pickle

import torchvision
import torchvision.transforms as transforms
import torch_geometric
from torch_geometric.data import Data
from typing import Any, Tuple


#######################################################
#            MNIST Dataset Wrapper & Loader           #
#######################################################
class ImageDatasetWrapper(RawDataset):
  """
  Functionality: We use the original dataset as is but after we hit the last
  index, we begin to sample based on inverse probability distribution obtained
  by keeping a count of each class proportion ...
  """
  def __init__(self, dataset, labels):
    self.dataset = dataset
    self.labels = labels

  def __getitem__(self, index):
    data, label = self.dataset[index], self.labels[index]
    return data, label

  def __len__(self):
    return len(self.dataset)


# https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
class ImageDataLoader(object):
  def __init__(self, training_images_filepath,training_labels_filepath,
                test_images_filepath, test_labels_filepath):
      self.training_images_filepath = training_images_filepath
      self.training_labels_filepath = training_labels_filepath
      self.test_images_filepath = test_images_filepath
      self.test_labels_filepath = test_labels_filepath
  
  def read_images_labels(self, images_filepath, labels_filepath):        
      labels = []
      with open(labels_filepath, 'rb') as file:
          magic, size = struct.unpack(">II", file.read(8))
          if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
          labels = array("B", file.read())        
      
      with open(images_filepath, 'rb') as file:
          magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
          if magic != 2051:
              raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
          image_data = array("B", file.read())        
      images = []
      for i in range(size):
          images.append([0] * rows * cols)
      for i in range(size):
          img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
          img = img.reshape(28, 28)
          images[i][:] = img

      # convert the images and labels ...
      images = np.array(images, dtype=float) # float type ...
      N, H, W = images.shape # extract shape details ...
      images = images.reshape((N, 1, H, W))
      # same for the labels ...
      labels = np.array(labels).reshape( (-1, 1) )
      # convert to torch ...
      images = torch.from_numpy(images).type(torch.FloatTensor)
      labels = torch.from_numpy(labels).type(torch.LongTensor)
      print("images dim:", images.size())
      print("labels dim:", labels.size())
      # return the result ...
      return images, labels

  def load_data(self):
      x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
      x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
      return (x_train, y_train), (x_test, y_test)


# Wrapper around Image torch Dataset class to output ImageGraphs in torch.geo data format
class CustomImageGraphDataset(RawDataset):
    __supported_datasets_mapping = {
        "mnist"        : "MNIST",
        "fashionmnist" : "FashionMNIST",
        "cifar10"      : "CIFAR10"
    }

    __supported_normalization_transforms = {
        "MNIST": transforms.Normalize((0.5,), (0.5,)),
        "FashionMNIST": transforms.Normalize((0.5,), (0.5,)),
        "CIFAR10": transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    }

    def __init__(self, dataset_name: str, train: bool = True, data_transforms: transforms = None, normalize: bool = False):
        self.train = train
        self.torch_vision_dataset_name = CustomImageGraphDataset.__supported_datasets_mapping[dataset_name.strip().lower()]
        assert self.torch_vision_dataset_name in torchvision.datasets.__all__, "Dataset by name: {torch_vision_dataset_name}, must be inside torchvision.datasets and name must exactly match torchvision's dataset name"
        default_transforms = transforms.Compose([ transforms.ToTensor(), *([CustomImageGraphDataset.__supported_normalization_transforms[self.torch_vision_dataset_name]] if normalize else []) ])
        self.transforms =  data_transforms if (data_transforms is not None) else default_transforms
        self.raw_target_dataset = getattr(torchvision.datasets, self.torch_vision_dataset_name)(root='./data', train=self.train, download=True, transform=self.transforms)

    def convert_image_to_graph(self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Code snippet from Tomasz: author of SGCN paper
        """
        sample_shape = sample[0].shape
        image_size = sample_shape[1]
        img = sample[0].clone()
        mask = sample[1]
        y = sample[2]
        img[mask == 1] = np.nan
        if len(sample_shape) == 2:
            x = img.view(-1, 1)
            x_train = img.view(image_size, image_size, 1).numpy()
        else:
            n_channels = sample_shape[0]
            x = img.reshape(n_channels, -1).T
            x_train = x.reshape(image_size, image_size, n_channels).numpy()

        pos_grid = np.zeros(shape=(image_size * image_size, 2))

        for i in range(1, image_size + 1):
            pos_grid[image_size * (i - 1): image_size * i, 0] = i
            pos_grid[image_size * (i - 1): image_size * i, 1] = range(1, image_size + 1)

        pos1 = []
        pos1_index = []
        pos1_val_index = []
        pos_array = np.zeros(shape=(image_size, image_size))

        ind = 0
        ind_val = -1
        for i in pos_grid:
            if (i[0] < image_size + 1) and (i[1] < image_size + 1):
                ind_val = ind_val + 1
                if not np.any(np.isnan(x_train[int(i[0] - 1), int(i[1] - 1)])):
                    pos_array[int(i[0] - 1), int(i[1] - 1)] = int(ind)
                    pos1.append(i)
                    pos1_index.append(ind)
                    pos1_val_index.append(ind_val)
                    ind = ind + 1

        pos = torch.tensor(pos1, dtype=torch.float)
        edge_index = list()

        for i in range(0, image_size):
            for j in range(0, image_size - 1):
                """"
                horizontal edges:
                (0*28, 0*28+1), (0*28+1, 0*28+2), ...(0*28+26, 0*28+27)
                (1*28, 1*28+1) ... (1*28+26,1*28+27)
                (2*28, 2*28+1) ... (2*28+26, 2*28+27)
                ...
                (27*28, 27*28+1) .... (27*28+26, 27*28+27)
                """
                if (not np.any(np.isnan(x_train[i, j])) and not np.any(np.isnan(x_train[i, j + 1]))):
                    edge_index.append([pos_array[i, j], pos_array[i, j + 1]])
                    edge_index.append([pos_array[i, j + 1], pos_array[i, j]])

                """
                vertical edges:
                (0*28, 1*28), (1*28, 2*28), ...(26*28, 27*28)
                (0*28+1, 1*28+1), (1*28+1, 2*28+1)... (26*28+1, 27*28+1)
                ...
                (0*28+27, 1*28+27), (1*28+27, 2*28+27).. (26*28+27, 27*28+27)
                """
                if (not np.any(np.isnan(x_train[j, i])) and not np.any(np.isnan(x_train[j + 1, i]))):
                    edge_index.append([pos_array[j, i], pos_array[j + 1, i]])
                    edge_index.append([pos_array[j + 1, i], pos_array[j, i]])

                """
                diagonal edges I:
                (0*28, 1*28+1), (0*28+1, 1*28+2) ... (0*28+26, 1*28+27)
                ...
                (26*28, 27*28+1), (26*28+1, 27*28+2) ... (26*28+26, 27*28+27)

                """
                if i < image_size - 1:
                    if (not np.any(np.isnan(x_train[i + 1, j + 1])) and not np.any(np.isnan(x_train[i, j]))):
                        edge_index.append([pos_array[i + 1, j + 1], pos_array[i, j]])
                        edge_index.append([pos_array[i, j], pos_array[i + 1, j + 1]])
                """
                diag II:

                (0*28+1, 1*28), (0*28+2, 1*28+1) .... (0*28+27, 1*28+26)
                (1*28+1, 2*28, 1*28+2, 2*28+1) ... (1*28+27, 2*28+26)
                ...
                (26*28+1, 27*28) ... (26*28+27, 27*28+26)
                """
                if i < image_size - 1:
                    if (not np.any(np.isnan(x_train[i + 1, j])) and not np.any(np.isnan(x_train[i, j + 1]))):
                        edge_index.append([pos_array[i + 1, j], pos_array[i, j + 1]])
                        edge_index.append([pos_array[i, j + 1], pos_array[i + 1, j]])

        edge_index = torch.LongTensor(edge_index)
        edge_index = edge_index.t().contiguous()
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=len(pos1))

        data_sample = Data(x=x[pos1_val_index], pos=pos, edge_index=edge_index, y=y)
        return data_sample
    
    def get_raw_image_dataset(self) -> RawDataset:
       return getattr(torchvision.datasets, self.torch_vision_dataset_name)(root='./data', train=self.train, download=True, transform=self.transforms)

    def __getitem__(self, index: int) -> Any:
        # Get image as tuple (img, label)
        img, label = self.raw_target_dataset[index]
        # Extract the sample format img -> graph converter requires
        img, mask, label = img, torch.zeros_like(img), torch.Tensor([label]).type(torch.LongTensor)
        # Get the image graph representation
        img_graph_sample = self.convert_image_to_graph(sample= [img, mask, label])
        # Return that sample
        return img_graph_sample
    
    def __len__(self) -> int:
        return len(self.raw_target_dataset)
    

######################################################################
#                                                                    #  
#       Process Raw MNIST Data and Convert to Graph                  #
#                                                                    #
######################################################################
def read_cached_graph_dataset( num_train, num_test, dataset_name, parent_dir = None):
  # Assertion on dataset splits
  assert num_train != None and isinstance(num_train, int)
  assert num_test != None and isinstance(num_test, int)
  
  # Get data file path
  curr_path = parent_dir or os.path.dirname(os.path.realpath('__file__'))
  base_path = os.path.join(curr_path, "Dataset", "RawGraph")
  if not os.path.exists(base_path):
    print("Dataset with name: {} doesn't exist".format(dataset_name))
    quit()
  
  # get the directory name
  dir_name = "train_{}_test_{}".format(num_train, num_test)
  full_dirpath = os.path.join(base_path, dir_name)
  if not os.path.exists(full_dirpath):
    print("Dataset for num_train={}, num_test={} was not generated".format(num_train, num_test))
    quit()

  # create the dataset name ...
  full_path = os.path.join(full_dirpath, dataset_name + ".pkl")
  assert full_path != None and full_path != "", "Full Path to dataset cannot be None/''"
  print(full_path)

  # confirm that full_path exists ...
  if not os.path.exists(full_path):
    print("Dataset with name: {} doesn't exist".format(dataset_name))
    quit()

  # struct to return ...
  struct = {
      "raw": {
          "x_train_data": [],
          "y_train_data": [],
          "x_test_data" : [],
          "y_test_data" : []
      },
      "geometric": {
          "qgcn_train_data":  [],
          "qgcn_test_data" : [],
          "sgcn_train_data": [],
          "sgcn_test_data" :  [],
      }
  }

  # read data from disk ...
  with open(full_path, "rb") as f:
    struct = pickle.load(f)

  # return the struct ...
  return struct
