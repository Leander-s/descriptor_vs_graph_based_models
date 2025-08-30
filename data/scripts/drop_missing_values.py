import sys
import pandas as pd


def drop_incomplete_columns(df):
    return df.dropna(axis=0, how="any")


def main():
    if len(sys.argv) < 3:
        raise Exception("No file path given")
    result_path = sys.argv[2]
    path = sys.argv[1]
    df = pd.read_csv(path)
    df = drop_incomplete_columns(df)
    df.to_csv(result_path, index=False)


if __name__ == '__main__':
    main()
