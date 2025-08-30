#!/bin/bash

conda init

echo "Creating runner environment"
conda create -y -n runner python=3.11

echo "Installing pytorch"
conda install -y -n runner pytorch=2.1 pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge

echo "Installing dgl"
conda install -y -n runner -c dglteam/label/th21_cu121 dgl

echo "Installing dgl-lifesci"
conda run -n runner python -m pip -q install dgllife

echo "Installing scikit-learn"
conda install -y -n runner scikit-learn

echo "Installing hyperopt"
conda run -n runner python -m pip -q install hyperopt
conda run -n runner python -m pip -q uninstall bson
conda run -n runner python -m pip -q install pymongo

echo "Installing xgboost"
conda install -y -n runner xgboost -c conda-forge

echo "Installing rdkit"
conda install -y -n runner rdkit=2023.03.2 -c conda-forge

echo "Installing pandas"
conda install -y -n runner pandas

echo "Creating generator environment"
conda create -y -n generator python=3.11

echo "Installing pytorch"
conda install -y -n generator pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge

echo "Installing pytorch extensions"
conda install -y -n generator -c conda-forge pyg pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv

echo "Installing libgcc"
conda install -y -n generator libgcc

echo "Installing pandas"
conda install -y -n generator pandas

echo "Installing minimol"
conda run -n generator python -m pip install minimol
