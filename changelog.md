# Descriptors
Since Linux MOE is not available to us as of right now, we used RDKit to 
calculate features from the cano-smiles representation of the given molecules. 
This means that our features may differ from those used by the authors of the 
original paper.

# Mogon
To make the code compatible with the CUDA and GCC versions on mogon, we had to 
change the versions of the used packages. Since the packages significantly 
changed with the versions, we also had to make significant changes to the code. 
We are not sure how much the performance differs between the versions (may be 
worth testing as it doesn't cost much) but results may also significantly 
differ from the original paper due to this.
## Versions
### Python : 3.6.5 -> 3.11.10
### pytorch : 1.3.1 -> 2.1.2
### cudatoolkit : 9.2 -> 12.1
### dgl : 0.4.1 -> 2.4.0
This is probably the most significant change and it also caused the most code 
changes. All the GNNs are implemented by DGL so this change could really alter 
the results.
### scikit-learn : 0.20.1 -> 1.5.1
### hyperopt : 0.2 -> 0.2.7
This might not be a significant change or a change at all since the 
installation of 0.2 might very well just install 0.2.7 anyway. The authors only 
specified 0.2.
### xgboost : 0.80 -> 2.1.3
This also seems like quite a change. The code did not have to be adapted to the 
new version here though. There could still be performance improvements in 
theory but we are not using any new/different functionality.

## Changes
