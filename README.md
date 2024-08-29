# Increasing Robustness of Data-Driven Fault Diagnostics with Knowledge Graphs

This code repository includes the code for the paper [Increasing Robustness of Data-Driven Fault Diagnostics with Knowledge Graphs](https://papers.phmsociety.org/index.php/phmconf/article/view/3552). Abstract:

> In the realm of Prognostics and Health Management (PHM), it is common to possess not only process data but also domain knowledge, which, if integrated into data-driven algorithms, can aid in solving specific tasks.
This paper explores the integration of Knowledge Graphs (KGs) into deep learning models to develop a more resilient approach capable of handling domain shifts, such as variations in machine operation conditions.
We present and assess a KG-enhanced deep learning approach in a representative PHM use case, demonstrating its effectiveness by incorporating domain-invariant knowledge through the KG.
Furthermore, we provide guidance for constructing a comprehensive hierarchical KG representation that preserves semantic information while facilitating numerical representation.
The experimental results showcase the improved performance and domain shift robustness of the KG-enhanced approach in fault diagnostics.

## Setup and data

To to run the code install all required packages.
```
pip install -r requirments.txt
```
Then download the [CWRU bearing dataset](https://engineering.case.edu/bearingdatacenter) and save it to *path/to/data/x*, where x is either 12k_DE, 12k_FE or Normal for the respective data.

## Case Study 1: Different Motor Loads

Adjust the paths in `load_config.yaml` and other hyperparameters to your respective settings. Run
```
python load_main.py
```
for a single model evaluation.

For the complete results of case study 1 (**this takes some time**) run
```
python load_run_experiments.py
```

All results are saved in *path/to/model_dir* specified in either `load_config.yaml` for `load_main.py` or directly in `load_run_experiments.py`.

## Case Study 2: Different Bearings

Adjust the paths in `bearing_config.yaml` and other hyperparameters to your respective settings. Run
```
python bearing_main.py
```
for a single model evaluation.

For the complete results of case study 2 (**this takes even longer**) run
```
python bearing_run_experiments.py
```

All results are saved in *path/to/model_dir* specified in either `bearing_config.yaml` for `bearing_main.py` or directly in `bearing_run_experiments.py`.
