### Generating pytorch geometric datasets:
Supported Datasets:
- AIDS
- COIL-DEL
- Enzymes
- Fingerprint
- Frankenstein
- Letters (high)
- Letters (low)
- Letters (med)
- Mutag
- Mutagenicity
- Proteins
- Proteins (full)
- Synthie

**STEPS**:
1. Download the datasets from the parent repository: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
2. Unzip the files and copy the unzipped dataset folder into the project directory: <root>/Dataset/Raw/*
3. Finally run through the datasets_parsing_functions.ipynb to generate the geometric datasets (they'll be populated into <root>/Dataset/Raw/Generated/*)
4. The result of 3. are pickle files containing data split into various train:test splits, named according to the convention: train_<X>_test_<Y>_struct_<dataset-name>_graph_data_pde=<?=yes or no>; pde -> positional descriptors
5. To run models on these newly generated dataset splits, copy them to <root>/Dataset/RawGraph/* and reference them in the datasets.yaml file (i.e., dataset_group: custom, dataset_name: <dataset-split-name>), to be accessible
6. **NOTE**: The <root>/Dataset/Raw/* must only contain the raw files for these graph datasets (the ipynb will throw errors otherwise) - when running through the notebook
7. **NOTE**: Feel free to modify the `root_dir` variable in the ipynb to another directory to avoid cluttering the <root>/Dataset/Raw/* space. Your new `root_dir` path should have the structure <root_dir>/Dataset/Raw/*
8. **NOTE**: All supported datasets were generated this way and uploaded into a publicly accessible dropbox space (the `download_url` key in the datasets.yaml has links pointing to the storage locations)

##### Other Notable Information:
- data_processing.py contains helper functions used by run_experiments*.py and other scripts across the project
- It contains the `read_cached_graph_dataset` method which is used to load in the custom data structures created through via the process above
