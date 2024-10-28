Before running any of the run_experiments*.py files, please run the below scripts (paths are w.r.t. current folder):
python3 ../generate_rawgraph_data.py -d MNIST 
python3 ../generate_rawgraph_data.py -d FashionMNIST 
python3 ../generate_rawgraph_data.py -d CIFAR10

These scripts above downloads the full MNIST, FashionMNIST and CIFAR10 datasets and converts them to pytorch geometric files.
The geometric datasets will be populated in the <root>/Dataset/Raw/* and <root>/Dataset/RawGraph/* directories respectively.
The Dataset/Raw/* will contain the raw dataset downloaded from online repositories.
The Dataset/RawGraph/* will contain the pytorch geometric dataset converted from the raw datasets downloaded into Dataset/Raw/*.

The results of the runs are populated into the Experiments directory (path: <root>/Experiments/*)

Use python3 ./run_experiment*.py -h to get more details on how to trigger the runs for standard datasets results
Primary flags:
  --dataset DATASET     Type of dataset, i.e., standard / custom / all, to run experiment on        [Default = None]
                        *** For standard dataset results, always pass in --dataset=standard as input
  --notraineval         Disables evaluating models for training accuracy                            [Default = False]
  --profilemodels       Enables profiling models for speed (model inference wall-clock)             [Default = False]
  --avgsmoothwidth      Sets the width of window over which collected perf stats should be averaged [Default = 5%]

**NOTE**
To generate the loss, learning rate and accuracy curves/charts (in the Appendix of the manuscript), you can extend the run.settings.yaml with enough learning rates, as depicted in the charts we show and plot the ensuing results
