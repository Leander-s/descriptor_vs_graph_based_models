#! /bin/bash

conda init

echo "Creating virtual environment"
conda create -y -n ml python=3.11

echo "Installing pytorch"
conda install -y -n ml pytorch=2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Installing dgl"
conda install -y -n ml -c dglteam/label/th21_cu121 dgl

echo "Installing dgl-lifesci"
conda run -n ml python -m pip -q install dgllife

echo "Installing scikit-learn"
conda install -y -n ml scikit-learn

echo "Installing hyperopt"
conda run -n ml python -m pip -q install hyperopt
conda run -n ml python -m pip -q uninstall bson
conda run -n ml python -m pip -q install pymongo

echo "Installing xgboost"
conda install -y -n ml xgboost -c conda-forge

echo "Installing rdkit"
conda install -y -n ml rdkit=2023.03.2 -c conda-forge
