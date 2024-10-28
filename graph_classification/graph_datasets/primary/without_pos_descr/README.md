This folder generates the comparison between QGRN and many GNNs for the IAM Graph Repository Graph Datasets (compiled without explicit positional descriptors)
The decision to train without using positional descriptors explicitly is to be fair to all GNNs (some GNNs do not consume explicit positional attributes)

Target Datasets:
- AIDS
- COIL-DEL
- Enzymes
- Frankenstein
- Letters (high)
- Letters (low)
- Letters (med)
- Mutag
- Mutagenicity
- Proteins
- Proteins (full)
- Synthie

**NOTE**:
Folder Structure:
- data.splits.yaml: Contains the train-test split and batch size settings for the runs
- datasets.yaml: Contains the list of datasets against which target models are evaluated
- run.settings.yaml: Contains all other settings for the runs 
    i.e., learning rate (lrs: how fast learning should occur)
          epochs (epochs: number of training epochs)
          runs (runs: indicates how many times run should be repeated)
