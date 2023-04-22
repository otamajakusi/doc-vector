import re
import random
from pathlib import Path


def get_sub_dirs(parent_dir: Path):
    sub_dirs = [d for d in parent_dir.glob("*") if d.is_dir()]
    return sub_dirs


def get_sub_dir_files(sub_dir: Path):
    files = [f for f in sub_dir.glob(f"{str(sub_dir.name)}*") if f.is_file()]
    return files


TRAIN_TXT = "ldcc_train.txt"
VAL_TXT = "ldcc_val.txt"
TEST_TXT = "ldcc_test.txt"

if __name__ == "__main__":
    sub_dirs = get_sub_dirs(Path("text"))
    pa = re.compile(r"[\n\t]")
    train_lines = []
    valid_lines = []
    test_lines = []
    with open(TRAIN_TXT, "w") as train, open(VAL_TXT, "w") as val, open(
        TEST_TXT, "w"
    ) as test:
        for sub_dir in sub_dirs:
            files = get_sub_dir_files(sub_dir)
            lines = []
            for f in files:
                text = pa.sub(" ", open(f).read())
                lines.append(text)
            train_index = random.sample(range(len(lines)), int(len(lines) * 0.8))
            rest_index = list(set(range(len(lines))) - set(train_index))
            valid_index = random.sample(
                range(len(rest_index)), int(len(rest_index) * 0.5)
            )
            test_index = list(set(range(len(rest_index))) - set(valid_index))
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
