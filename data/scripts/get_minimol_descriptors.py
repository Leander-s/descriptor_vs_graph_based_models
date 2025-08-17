import sys
import pandas as pd
from minimol import Minimol
import time


def get_minimol_descriptors(model, filepath):
    df = pd.read_csv(filepath)

    representation = 'cano_smiles'

    smiles = df[representation].to_numpy().tolist()

    start = time.time()
    features = model(smiles)
    end = time.time()
    _time = round(end-start, 2)

    feature_df = pd.DataFrame([tensor.tolist() for tensor in features])
    feature_df.columns = [f"feature_{i}" for i in range(feature_df.shape[1])]
    df = pd.concat([df, feature_df], axis=1)
    return df, _time


def main():
    if (len(sys.argv) == 1):
        raise Exception("Give result data as argument")

    name = sys.argv[1]
    filepath = f"../original_datasets/{name}.csv"

    model = Minimol()

    df, time = get_minimol_descriptors(model, filepath)

    df.to_csv(
        f"../minimol_datasets/{name}_minimol_descriptors.csv", index=False)


if __name__ == '__main__':
    main()
