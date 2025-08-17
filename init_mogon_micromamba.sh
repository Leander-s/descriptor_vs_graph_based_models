#! /bin/bash

name=$1

module load compiler/GCC

source ~/.bashrc

conda_initialize

micromamba init

echo "Creating virtual environment"
micromamba create -y -n $name python=3.11

echo "Installing packaging"
micromamba install -n ml -c conda-forge packaging

echo "Installing pydantic"
micromamba install -n ml -c conda-forge pydantic

echo "Installing pytorch"
micromamba install -y -n ml pytorch=2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Installing dgl"
micromamba install -n ml -c dglteam/label/th21_cu121 dgl

echo "Installing dgl-lifesci"
micromamba run -n ml python -m pip -q install dgllife

echo "Installing scikit-learn"
micromamba install -y -n ml scikit-learn

echo "Installing hyperopt"
micromamba run -n ml python -m pip -q install hyperopt

micromamba run -n ml python -m pip -q uninstall bson
micromamba run -n ml python -m pip -q install pymongo

echo "Installing xgboost"
micromamba install -y -n ml xgboost -c conda-forge

echo "Installing rdkit"
micromamba install -y -n ml rdkit=2023.03.2 -c conda-forge
