"""model_deap.py"""

import torch
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.transforms.pyg import ToG
import torcheeg.datasets.constants as eeg_constants
from torcheeg.model_selection import KFoldPerSubject, train_test_split
from torch import tensor
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import time
import random
import numpy as np
import sys
import os
import itertools
import pickle
import logging

# import custom module
from eeg_autoencoder import autoencoder

EEG_ROOT = "./eeg_autoencoder/"
DEAP_ROOT = EEG_ROOT+"DEAP/"
REC_LEN = 60 # seconds
FS = 128 # Hz
N_PPT = 32
N_EEG_CHANNELS = 32
DEAP_BASELINE = 21 # seconds

RESULTS_DIR = EEG_ROOT+"results/"
MODEL_SAVE_DIR = EEG_ROOT+"models/"
LOG_DIR = EEG_ROOT+"logs/"

CV_FOLDS = 5
PRETRAIN_EPOCHS = 100

SEED = 424242

def load_deap(ppt, fold,
              target='valence',
              win_len=3,
            #   test_size=0.2,
              val_size=0.2,
            #   f_start=4,
            #   f_stop=44,
            #   f_res=2,
              n_cpu=1):
    
    FULLY_CONNECTED_ADJACENCY_MATRIX = [[1]*32]*32

    # band_dict = {str(f):[f,f+f_res] for f in range(f_start, f_stop, f_res)}
    band_dict = {'theta':[4,8], 'alpha':[8,12], 'beta':[12,30], 'gamma':[30,44]}

    chunk_size = win_len * FS
    io_path = DEAP_ROOT + f".torcheeg/deap_{target}_s{ppt}_f{fold}_io"
    split_path = DEAP_ROOT + f".torcheeg/deap_{target}_s{ppt}_f{fold}_split"
    dataset = DEAPDataset(root_path=DEAP_ROOT+"data_preprocessed_python",
                            chunk_size=chunk_size,
                            num_baseline=1,
                            baseline_chunk_size=DEAP_BASELINE*FS,
                            overlap=int(FS*1.5),
                            online_transform=transforms.Compose([
                                transforms.MeanStdNormalize(axis=1),
                                transforms.BandPowerSpectralDensity(band_dict=band_dict),
                                ToG(FULLY_CONNECTED_ADJACENCY_MATRIX, binary=True)
                            ]),
                            label_transform=transforms.Compose([
                                transforms.Select(target),
                                transforms.Binary(5.0)
                            ]),
                            num_worker=n_cpu,
                            io_path=io_path)

    cv = KFoldPerSubject(n_splits=CV_FOLDS,
               shuffle=True,
               random_state=SEED,
               split_path=split_path)
    sub_id = f's{ppt:02d}.dat'
    for k, (train_dataset, test_dataset) in enumerate(cv.split(dataset, subject=sub_id)):
        if k >= fold:
            break
    
    train_dataset, val_dataset = train_test_split(train_dataset,
                                                  test_size=val_size,
                                                  shuffle=True,
                                                  random_state=SEED,
                                                  split_path=split_path+'_val')

    return train_dataset, val_dataset, test_dataset, io_path, split_path

class ModelRun:
    def __init__(self, ppt, fold, target, device, n_cpu=1):
        self.device = device

        self.geom_dataloaders(ppt=ppt, fold=fold, target=target, n_cpu=n_cpu)

    def geom_dataloaders(self, ppt, fold, target, n_cpu=1, batch_size=32):
        train_dataset, val_dataset, test_dataset, io_path, split_path = load_deap(
                                    ppt=ppt, fold=fold, target=target, n_cpu=n_cpu)
        self.io_path = io_path
        self.split_path = split_path
        self.n_bands = train_dataset[0][0].x.size(1) # save no. frequency bands

        # generate position info for each channel
        ch_position = tensor([eeg_constants.STANDARD_1020_CHANNEL_LOCATION_DICT[channel]
                            for channel in eeg_constants.DEAP_CHANNEL_LIST])
        self.pos_dim = ch_position.size(1)
        
        # create dataloader for batch training / testing
        self.train_loader = DataLoader([d[0].update({'pos':ch_position, 'y':tensor(d[1])})
                                for d in train_dataset],
                                batch_size=batch_size,
                                drop_last=True,
                                num_workers=n_cpu-1, persistent_workers=True)
        
        self.val_loader = DataLoader([d[0].update({'pos':ch_position, 'y':tensor(d[1])})
                                for d in val_dataset],
                                batch_size=batch_size,
                                drop_last=True)

        self.test_loader = DataLoader([d[0].update({'pos':ch_position, 'y':tensor(d[1])})
                                for d in test_dataset],
                                batch_size=batch_size,
                                drop_last=True)
        
        self.c_loss_fn = torch.nn.BCEWithLogitsLoss()


class SaeModelRun(ModelRun):

    def __init__(self, ppt, fold, target, device,
                 num_factors=32,
                 hidden_channels=32,
                 latent_channels=16,
                 class_hidden_units=16,
                 layers_num=3,
                 num_sub_kernels=64,
                 r_loss='mse',
                 n_cpu=1,
                 **kwargs):
        
        super().__init__(ppt, fold, target, device, n_cpu)
        
        self.num_factors = num_factors
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.layers_num = layers_num
        self.num_sub_kernels = num_sub_kernels
        self.r_loss = r_loss

        self.model = autoencoder.SAE(in_channels=self.n_bands,
                                     num_nodes=N_EEG_CHANNELS,
                                     num_factors=num_factors,
                                     hidden_channels=hidden_channels,
                                     latent_channels=latent_channels,
                                     class_hidden_units=class_hidden_units,
                                     layers_num=layers_num, 
                                     num_sub_kernels=num_sub_kernels,
                                     pos_descr_dim=self.pos_dim,
                                     edge_attr_dim=-1,
                                     device=device,
                                     **kwargs)
        self.model.to(device)

        if r_loss == 'mse':
            self.r_loss_fn = torch.nn.MSELoss()
        elif r_loss == 'kl':
            self.r_loss_fn = autoencoder.kl_div_loss

    
    def train(self, n_epochs=200, test_interval=20, lr=0.01, c_weight=100, **kwargs):
        self.lr = lr
        self.c_weight = c_weight
        print(f'Learning rate: {lr}, C-weight: {c_weight}')

        if not hasattr(self, 'optimizer'):
            self.optimizer = AdamW(self.model.parameters(), lr=lr)

        start_time = time.time()  # Start time of training

        train_r_loss,  train_c_loss,  train_auc = self.evaluate("train")
        print(f'Init. Train R-Loss: {train_r_loss:.2f} C-Loss: {train_c_loss:.4f} AUC: {train_auc:.2f}')

        self.model.train()
        for epoch in range(n_epochs):
            # total = 0
            # correct = 0
            r_loss_total = 0
            c_loss_total = 0
            for data in self.train_loader:
                data = data.to(self.device)

                x_est, y_logit = self.model(data.clone())
                r_loss = self.r_loss_fn(x_est, data.x)
                y_tensor = data.y.to(torch.float32)
                y_est = F.sigmoid(y_logit.squeeze())
                c_loss = self.c_loss_fn(y_est, y_tensor)
                r_loss_total += r_loss.item()
                c_loss_total += c_loss.item()

                # predicted = y_est.round()
                # total += y_tensor.size(0)
                # correct += (predicted == y_tensor).sum().item()

                loss = r_loss + self.c_weight * c_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()


            # train_accuracy = 100 * correct / total
            r_loss_total /= len(self.train_loader)
            c_loss_total /= len(self.train_loader)
            end_time = time.time()  # End time of the epoch
            elapsed_time = end_time - start_time  # Time elapsed for the epoch
            print(f'Epoch {epoch} R-Loss: {r_loss_total:.2f} C-Loss: {c_loss_total:.4f}'
                  f' Time Elapsed: {elapsed_time:.2f} seconds')

            if (epoch + 1) % test_interval == 0:
                test_r_loss, test_c_loss, test_auc = self.evaluate("val")
                self.model.train()
                print(f'\tVal R-Loss: {test_r_loss:.2f} C-Loss: {test_c_loss:.4f} AUC: {test_auc:.2f}')

        if not hasattr(self, 'total_epochs'):
            self.total_epochs = n_epochs
        else:
            self.total_epochs += n_epochs

        self.train_time = elapsed_time


    def save_results(self, txt_file_path, model_file_path, pkl_file_path):
        test_r_loss, test_c_loss, test_auc = self.evaluate("test")
        val_r_loss, val_c_loss, val_auc = self.evaluate("val")
        train_r_loss, train_c_loss, train_auc = self.evaluate("train")
        
        # create directories if they don't exist
        for fp in [txt_file_path, model_file_path, pkl_file_path]:
            directory = os.path.dirname(fp)
            if not os.path.exists(directory):
                os.makedirs(directory)

        with open(txt_file_path, 'w') as file:
            # file.write(f'Test R-Loss: {test_r_loss:.2f}\n')
            # file.write(f'Test C-Loss: {test_c_loss:.4f}\n')
            # file.write(f'Test Accuracy: {test_accuracy:.2f}%\n\n')
            file.write(f'Train R-Loss: {train_r_loss:.2f}\n')
            file.write(f'Train C-Loss: {train_c_loss:.4f}\n')
            file.write(f'Train AUC: {train_auc:.2f}\n\n')
            file.write(f'Val R-Loss: {val_r_loss:.2f}\n')
            file.write(f'Val C-Loss: {val_c_loss:.4f}\n')
            file.write(f'Val AUC: {val_auc:.2f}\n\n')

            file.write(f'total_epochs: {self.total_epochs}\n')
            file.write(f'num_factors: {self.num_factors}\n')
            file.write(f'hidden_channels: {self.hidden_channels}\n')
            file.write(f'latent_channels: {self.latent_channels}\n')
            file.write(f'layers_num: {self.layers_num}\n')
            file.write(f'num_sub_kernels: {self.num_sub_kernels}\n')
            file.write(f'lr: {self.lr}\n')
            file.write(f'c_weight: {self.c_weight}\n')
            file.write(f'r_loss: {self.r_loss}\n')
            file.write(f'train_time: {self.train_time:.3f}\n')

            file.write(f'\nmodel_path: {model_file_path}\n')
            file.write(f'deap io: {self.io_path}\n')
            file.write(f'deap split: {self.split_path}\n')
            file.write(f'seed: {SEED}\n')

         # Save test losses and accuracy to pickle file
        test_results = {
            'test_r_loss': test_r_loss,
            'test_c_loss': test_c_loss,
            'test_auc': test_auc
        }
        with open(pkl_file_path, 'wb') as file:
            pickle.dump(test_results, file)
        
        torch.save(self.model.state_dict(), model_file_path)


    def evaluate(self, split="train"):
        if split == "test":
            loader = self.test_loader
        elif split == "val":
            loader = self.val_loader
        elif split == "train":
            loader = self.train_loader
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.model.eval()
        test_r_loss = 0
        test_c_loss = 0
        # correct = 0
        # total = 0
        all_y_logit = []  # List to store y_logit values
        all_y = []  # List to store y values
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)

                x_est, y_logit = self.model(data.clone())
                r_loss = self.r_loss_fn(x_est, data.x)
                y_tensor = data.y.to(torch.float32)
                y_est = F.sigmoid(y_logit.squeeze())
                c_loss = self.c_loss_fn(y_est, y_tensor)
                test_r_loss += r_loss.item()
                test_c_loss += c_loss.item()
                
                # predicted = y_est.round()
                # total += y_tensor.size(0)
                # correct += (predicted == y_tensor).sum().item()
                
                all_y_logit.append(y_logit.detach().cpu().numpy())  # Append y_logit to the list
                all_y.append(data.y.detach().cpu().numpy())  # Append y to the list

        test_r_loss /= len(loader)
        test_c_loss /= len(loader)
        # test_accuracy = 100 * correct / total
        test_auc = roc_auc_score(np.concatenate(all_y), np.concatenate(all_y_logit))
        return test_r_loss, test_c_loss, test_auc  # Return y_logit_list


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    model_type = sys.argv[1]
    if len(sys.argv) > 2:
        fold = int(sys.argv[2])
    else:
        fold = 0
    if len(sys.argv) > 3:
        target = sys.argv[3]
    else:
        target = 'valence'
    if len(sys.argv) > 4:
        n_cpu = int(sys.argv[4])
    else:
        n_cpu = 1

    # set up logging
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(f'{model_type}_deap')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    file_handler = logging.FileHandler(
        os.path.join(EEG_ROOT+'logs', f'{timeticks}.log'))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'running on {device}')

    # Define the parameter grid
    param_grid = {'lr': [0.001], 'c_weight': [1000.0, 10000.0, 100.0]}

    # Iterate over all combinations of parameter values
    for p in range(1, N_PPT+1):
    
        for params in itertools.product(*param_grid.values()):
            seed_everything(SEED)
            
            # Convert the parameter values to a dictionary
            param_dict = dict(zip(param_grid.keys(), params))

            # Create an instance of ModelRun with the current parameter values
            this_run = SaeModelRun(fold=fold,
                                   ppt=p,
                                   conv_type=model_type,
                                   target=target,
                                   device=device,
                                   n_cpu=n_cpu,
                                   num_factors=64,
                                   hidden_channels=32,
                                   latent_channels=16,
                                   num_sub_kernels=32)

            # pre-train the model
            pre_train_id = model_type + f'_pretrain_epoch{PRETRAIN_EPOCHS}_sub{p}_fold{fold}'
            pre_train_model_path = MODEL_SAVE_DIR + target + '/' + pre_train_id + '.pth'
            if os.path.exists(pre_train_model_path):
                print(f'Loading pre-trained model {pre_train_model_path}')
                this_run.model.load_state_dict(torch.load(pre_train_model_path))
            else:
                this_run.train(n_epochs=PRETRAIN_EPOCHS, test_interval=20, c_weight=1e-9, lr=0.01)
                pre_train_res_path = RESULTS_DIR + target + '/' + pre_train_id + '.txt'
                pre_train_pkl_path = RESULTS_DIR + target + '/' + pre_train_id + '.pkl'
                this_run.save_results(pre_train_res_path, pre_train_model_path, pre_train_pkl_path)
                delattr(this_run, 'optimizer')

            for i in range(10):
                # Train the model with the current parameter values
                this_run.train(n_epochs=10, test_interval=5, **param_dict)

                # Save the results and model with the current parameter values
                model_id = model_type + '_wAUC'
                for key, value in param_dict.items():
                    model_id += f'_{key}{value}'
                model_id += f'_epoch{this_run.total_epochs}_sub{p}_fold{fold}'
                print(f'Saving model {model_id}')
                res_path = RESULTS_DIR + target + '/' + model_id + '.txt'
                model_path = MODEL_SAVE_DIR + target + '/' + model_id + '.pth'
                pkl_path = RESULTS_DIR + target + '/' + model_id + '.pkl'
                this_run.save_results(res_path, model_path, pkl_path)
