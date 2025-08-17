# A fair experimental comparison of four generations of machine learning methods for single-task activity prediction

# Code

## Dependencies

### Python packages
You can find all the versions we used in the requirements.txt

## Execution of source code

### run.py
You can execute all the code using the run.py script. Usage is as follows:

    python run.py <dataset name> <model name> <descriptors(optional)>

If you are using a graph based model, leave the descriptors argument out. 
Datasets you want to run obviously need to exist. Some of the datasets used in 
our the study are not available in this repository. All datasets other than 
MOE datasets can be generated using this repository. More instructions on that 
in [Data](#Data).

# Data

## original_datasets
Datasets from the original paper

## other datasets
All other datasets have to be generated first using the dataset generation 
script.

## rdkit_datasets
RDKit descriptors for original datasets
## minimol_datasets
Minimol descriptors for original datasets
## moe_datasets
Not sure if we are allowed to publish those
## additional_datasets
Everything again for split multitask datasets
## scripts
Scripts to get any non-original datasets


# TODO

This should be gone soon :D.
- write a requirements.txt with all the dependencies. Also write a script to 
install a conda environment that satisfies all of them
- provide detailed instructions to get a functional environment going using 
conda
- program dataset generation script that generates all datasets (all 
descriptors, all additional_datasets with all descriptors) form one main 
script
- test all the code, make sure it runs on my machine at least. Some one 
reproducing probably won't want to just use a docker container but maybe 
provide one as well
- give clear instructions for running all the code and generating all the 
datasets
