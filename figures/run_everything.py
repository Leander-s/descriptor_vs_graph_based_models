import subprocess
from lib.config import Datasets, Models
import os


def main():
    root = os.getcwd()

    last_jobs = ["", "", "", ""]
    i = 0
    for dataset in Datasets:
        for model in Models:
            if "rdkit" in model:
                descriptors = "rdkit"
            elif "minimol" in model:
                descriptors = "minimol"
            elif "moe-neutral" in model:
                descriptors = "moe-neutral"
            else:
                descriptors = ""

            model = model.split("_")[0]

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
                if (result.stderr != ""):
                    print(result.stderr)
                last_jobs[i] = result.stdout.strip()
                i = (i + 1) % 4
                os.chdir(root)
            except subprocess.CalledProcessError as e:
                os.chdir(root)
                print("Error:", e)
            except:
                print("There was an error")
                exit(1)


if __name__ == '__main__':
    main()
