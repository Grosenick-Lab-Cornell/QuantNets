### NOTE:
The following files are at root (can be located from anywhere in the codebase):
- experiment.py
- generate_rawgraph_data.py

These files can be (are) invoked from anywhere within the project

Folder Structure:
- standard_datasets: tracks experiments performed on standard image datasets (and their torch equivalent conversions)
- graph_datasets: tracks all experiments performed with classic graph datasets (converted into torch geometric formats)
- - Further divided into primary & supplementary - distinguishing where the results are located in the manuscript
- - - primary: results generated here can be found in the main paper (parts of these results may be pulled into Appendix for further elaboration)
- - - supplementary: sensitivity studies of the models, QGCN and QGRN (results are found only in Appendix)
