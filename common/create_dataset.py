import re
import random
from pathlib import Path


def get_sub_dirs(parent_dir: Path):
    sub_dirs = [d for d in parent_dir.glob("*") if d.is_dir()]
    return sub_dirs


def get_sub_dir_files(sub_dir: Path):
    files = [f for f in sub_dir.glob(f"{str(sub_dir.name)}*") if f.is_file()]
    return files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train-txt", help="training dataset", required=True)
    parser.add_argument("--val-txt", help="validation dataset", required=True)
    parser.add_argument("--test-txt", help="test dataset", required=True)

    args = parser.parse_args()

    Path(args.train_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.test_txt).parent.mkdir(parents=True, exist_ok=True)

    sub_dirs = get_sub_dirs(Path("text"))
    pa = re.compile(r"[\n\t]")
    train_lines = []
    valid_lines = []
    test_lines = []
    with open(args.train_txt, "w") as train, open(args.val_txt, "w") as val, open(
        args.test_txt, "w"
    ) as test:
        for sub_dir in sub_dirs:
            files = get_sub_dir_files(sub_dir)
            lines = []
            for f in files:
                text = pa.sub(" ", open(f).read())
                lines.append(text)
            indexes = range(len(lines))
            train_index = random.sample(indexes, int(len(lines) * 0.8))
            rest_index = set(indexes) - set(train_index)
            valid_index = random.sample(rest_index, int(len(rest_index) * 0.5))
            test_index = list(rest_index - set(valid_index))
            # print(f"{'*' * 100}")
            # print(f"{train_index=}")
            # print(f"{valid_index=}")
            # print(f"{test_index=}")
            assert len(set(train_index) & set(valid_index) & set(test_index)) == 0

            for index in train_index:
                train.write(f"{sub_dir.name}\t{lines[index]}\n")
            for index in valid_index:
                val.write(f"{sub_dir.name}\t{lines[index]}\n")
            for index in test_index:
                test.write(f"{sub_dir.name}\t{lines[index]}\n")
