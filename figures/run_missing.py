import subprocess
from show_missing import get_missing
import os


def main():
    missing = get_missing(".")
    root = os.getcwd()

    missing_by_dataset = {}
    for file in missing:
        dataset, model, descriptors, *_ = file.split("_")
        if dataset in missing_by_dataset.keys():
            missing_by_dataset[dataset].append((model, descriptors))
        else:
            missing_by_dataset[dataset] = [(model, descriptors)]

    last_jobs = ["", "", "", ""]
    i = 0
    for dataset in missing_by_dataset.keys():
        for model, descriptors in missing_by_dataset[dataset]:
            if descriptors not in ["rdkit", "minimol", "moe-neutral"]:
                descriptors = ""

            args = ["--env", f"{i}", "--dataset", f"{dataset}",
                    "--model", f"{model}", "--partition", "a100dl",
                    ]

            if descriptors != "":
                args.append("--descriptors")
                args.append(f"{descriptors}")

            print(last_jobs)
            if last_jobs[i] != "":
                args.append("--dependency")
                args.append(f"afterany:{last_jobs[i]}")

            try:
                os.chdir("../../../mogon_scripts")
                result = subprocess.run(
                    ["bash", "./run.sh"] + args,
                    check=True, capture_output=True, text=True)
                last_jobs[i] = result.stdout.strip()
                if (result.stderr != ""):
                    print(result.stderr)
                os.chdir(root)
            except subprocess.CalledProcessError as e:
                os.chdir(root)
                print("Error:", e)
            i = (i + 1) % 4


if __name__ == '__main__':
    main()
