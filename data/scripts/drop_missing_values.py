import sys
import pandas as pd


def drop_incomplete_columns(df):
    df = df.dropna(axis=1, how="any")
    return df


def main():
    if len(sys.argv) == 1:
        raise Exception("No file path given")
    path = sys.argv[1]
    df = pd.read_csv(path)
    drop_incomplete_columns(df)
    df.to_csv(path, index=False)


if __name__ == '__main__':
    main()
