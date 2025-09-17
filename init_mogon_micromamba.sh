#! /bin/bash

name=$1

source ~/.bashrc

conda_initialize

micromamba init

echo "Creating virtual environment"
micromamba create -y -n "$name" python=3.11

echo "Installing packaging"
micromamba install -n "$name" -c conda-forge packaging

echo "Installing pydantic"
micromamba install -n "$name" -c conda-forge pydantic

echo "Installing pytorch"
micromamba install -y -n "$name" pytorch=2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Installing dgl"
micromamba install -n "$name" -c dglteam/label/th21_cu121 dgl

echo "Installing dgl-lifesci"
micromamba run -n "$name" python -m pip -q install dgllife

echo "Installing scikit-learn"
micromamba install -y -n "$name" scikit-learn

echo "Installing hyperopt"
micromamba run -n "$name" python -m pip -q install hyperopt

micromamba run -n "$name" python -m pip -q uninstall bson
micromamba run -n "$name" python -m pip -q install pymongo

echo "Installing xgboost"
micromamba install -y -n "$name" xgboost -c conda-forge

echo "Installing rdkit"
micromamba install -y -n "$name" rdkit=2023.03.2 -c conda-forge
