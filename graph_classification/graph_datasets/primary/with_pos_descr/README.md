This folder generates the comparison between SGCN and QGRN for select datasets with positional descriptors

Target Datasets:
- Navier Stokes Binary
- Navier Stokes Denary - 1
- Navier Stokes Denary - 2
- AIDS
- Letter (high)
- Letter (low)
- Letter (med)

**NOTE**:
Folder Structure:
- data.splits.yaml: Contains the train-test split and batch size settings for the runs
- datasets.yaml: Contains the list of datasets against which target models are evaluated
- run.settings.yaml: Contains all other settings for the runs 
    i.e., learning rate (lrs: how fast learning should occur)
          epochs (epochs: number of training epochs)
          runs (runs: indicates how many times run should be repeated)
