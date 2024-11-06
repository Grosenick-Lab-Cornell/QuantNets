Scripts & How-to for generating EEG AutoEncoder (generative and classification) results

### DATASET
DEAP dataset can be accessed at https://www.eecs.qmul.ac.uk/mmv/datasets/deap/.
The preprocessed data should be downloaded in python format and saved in the *eeg_autoencoder/DEAP* subfolder.

### TRAINING MODELS
To replicate results presented in Section 4.5 of *Generalizing CNNs to Graphs with Learnable Neighborhood Quantization* run the following from the *QuantNet* folder:  
```
python3 -m eeg_autoencoder.model_deap QGRL  
python3 -m eeg_autoencoder.model_deap SGCN  
```

The script in model_deap.py has several optional parameters. In order, they are:  
- *model_type*: Accepts the values **QGRL** or **SGCN**. Determines the type of convolutional layer used in the model.  
- *fold* (Default: 0): Integer 0-4. Can be used to perform 5-fold cross-validation of test set results.  
- *target* (Default: valence): Accepts the values **valence** or **arousal**. Used to select which type of emotional state to model in the data.  
- *n_cpu* (Default: 1): Number of CPU cores to use for loading and preprocessing data.  

### ACCESSING/INTERPRETING RESULTS
The training and validation set results for each checkpoint for each model will be saved in a *.txt* file in the *eeg_autoencoder/results* subfolder.
The corresponding test set results are saved in a *.pkl* (pickle) file in the same directory.

