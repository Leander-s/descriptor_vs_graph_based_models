import sys
import pandas as pd
import time
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def get_descriptors(filepath):
    df = pd.read_csv(filepath)
    column_count = len(df.columns)

    representation = 'cano_smiles'

    mol_list = []

    for smile in df[representation]:
        mol = Chem.MolFromSmiles(smile)
        # mol = Chem.AddHs(mol)
        mol_list.append(mol)

    Desc_list_func = MoleculeDescriptors.MolecularDescriptorCalculator(
        x[0] for x in Descriptors._descList)

    descriptors = Desc_list_func.GetDescriptorNames()
    descriptor_list = []

    for descriptor in descriptors:
        descriptor_list.append([])

    start = time.time()
    for index, mol in enumerate(mol_list):
        features = Desc_list_func.CalcDescriptors(mol)
        for index, descriptor in enumerate(descriptors):
            descriptor_list[index].append(features[index])
    end = time.time()
    _time = round(end-start, 2)

    for index, descriptor in enumerate(descriptors):
        df.insert(column_count, descriptor, descriptor_list[index])

    return df, _time


def main():
    if (len(sys.argv) == 1):
        raise Exception("Give result data as argument")

    name = sys.argv[1]
    filepath = f"../original_datasets/{name}.csv"

    df, time = get_descriptors(filepath)

    df.to_csv(f"../rdkit_datasets/{name}-rdkit_descriptors.csv", index=False)


if __name__ == '__main__':
    main()
