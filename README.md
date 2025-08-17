# Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models

# Reading the results

## Tables
### Data available/Complete
These tables are relatively straight forward:
Rows are datasets, Columns are models. If a result is golden, it was the best 
result on the dataset. The second best is silver, third is bronze/brownish. In 
the first row you can see the mean ranks of the models on all datasets. Same 
color-coding for ranks applies here. No data results in worst rank for Complete
table.
### Sick plot
This is a 2D plot with time on the y-axis and rank on the x-axis. Time is given
as a mean rank of the time needed for training on and evaluating a dataset over 
all datasets. Rank is the mean rank achieved over all datasets performance wise.
Hopefully critical differences are drawn as bars over the results as well now.

## plots
The plots show the average score of each model on the given dataset. Hopefully 
also with critical differences by now.

## ttest_plots
This is a table of all the models compared with all the other models in 
corrected resampled t-tests on the results of the models on the given dataset. 
If a cell is green or red a critical difference was found. If it's green the
model on the right was better, if it's red the model on the bottom was better.

# Mogon

To run the code on mogon the 'mogon-env' git branch should be used:

    git switch mogon-env

Then go to machine_learning_praktikum/mogon_scripts/.
(None of this is supported at the moment)
There you can:
Run everything:

    ./run_everything.sh

Run any model on any dataset:

    ./run.sh <dataset> <model> <partition>

- dataset : freesolv, esol, lipop, bace, bbbp, hiv, and many more
- model : gcn, gat, mpnn, attentivefp, svm, svm-wwl, rf, xgb, dnn
- partition : a40, a100dl, a100ai

# Code

## Main branch

Source code modified so it works on mogon hardware using different versions.

## Activate python environment

    conda activate ml

## Dependencies

### Python packages

Should be installed by init.sh or init_windows.bat

## Execution of source code

### Mogon
Scripts can be found in mogon_scripts:
- run_everything.sh <partition> runs everything...
- run.sh --dataset <dataset_name> --model <model_name> --partition <partition> --descriptors(if descriptor based model) <descriptors>  
- run_sets.sh <model> <partition> <descriptors>(if descriptor based model)

# Data

## original_datasets
Datasets from the original paper
## rdkit_datasets
RDKit descriptors for original datasets
## minimol_datasets
Minimol descriptors for original datasets
## additional_datasets
Everything again for split multitask datasets
## scripts
Scripts to get any non-original datasets

# Figure

Results provided by the paper.
