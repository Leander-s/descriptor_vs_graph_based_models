#!/bin/bash

conda init

echo "Creating runner environment"
conda env create -f runner.yml
echo "Installing generator environment"
conda env create -f generator.yml
