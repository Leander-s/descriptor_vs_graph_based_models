import pandas as pd


def get_smiles(df: pd.DataFrame) -> pd.DataFrame:
    smiles = df["cano_smiles"]
    return smiles


def main():
    datasets = [
        "freesolv",
        "esol",
        "lipop",
        "bace",
        "bbbp",
        "hiv",
        "clintox",
        "sider",
        "tox21",
        "toxcast",
        "muv",
    ]

    allSmiles = pd.DataFrame()

    for dataset in datasets:
        file = f"{dataset}.csv"
        frame = pd.read_csv(file)
        smilesFrame = get_smiles(frame)
        allSmiles = pd.concat([allSmiles, smilesFrame], ignore_index=True)
        smilesFrame.to_csv(f"./smiles/{dataset}_smiles.csv", index=False)

    allSmiles.to_csv("./smiles/all.csv", index=True)


if __name__ == '__main__':
    main()
