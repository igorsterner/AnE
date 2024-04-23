import json

input_train_file = (
    "/data/corpora/lid/formatted/train/german-english-denglisch_train.jsonl"
)
input_dev_file = "/data/corpora/lid/formatted/dev/german-english-denglisch_dev.jsonl"
input_test_file = "/data/corpora/lid/formatted/test/german-english-denglisch_test.jsonl"


output_train_file = (
    "/data/corpora/lid/collapsed/train/german-english-denglisch-baseline_train.jsonl"
)
output_dev_file = (
    "/data/corpora/lid/collapsed/dev/german-english-denglisch-baseline_dev.jsonl"
)
output_test_file = (
    "/data/corpora/lid/collapsed/test/german-english-denglisch-baseline_test.jsonl"
)

input_files = {
    "train": input_train_file,
    "dev": input_dev_file,
    "test": input_test_file,
}
output_files = {
    "train": output_train_file,
    "dev": output_dev_file,
    "test": output_test_file,
}

output_data = {split: [] for split in input_files.keys()}

label_map = {
    "1": "English",
    "2": "notEnglish",
    "3": "notEnglish",
    "3a": "NamedEntity",
    "3a-E": "NamedEntity",
    "3a-D": "NamedEntity",
    "3a-AE": "NamedEntity",
    "3a-AD": "NamedEntity",
    "3b": "notEnglish",
    "3c": "Mixed",
    "3c-C": "Mixed",
    "3c-M": "Mixed",
    "3c-EC": "Mixed",
    "3c-EM": "Mixed",
    "3-E": "English",
    "3-D": "notEnglish",
    "3-O": "notEnglish",
    "4": "notEnglish",
    "4a": "notEnglish",
    "4b": "Other",
    "4b-E": "English",
    "4b-D": "notEnglish",
    "4c": "Other",
    "4d": "notEnglish",
    "4d-E": "English",
    "4d-D": "notEnglish",
    "4e-E": "English",
    "<url>": "Other",
    "<punct>": "Other",
    "<EOS>": "Other",
    "<EOP>": "Other",
}

for split in input_files.keys():

    with open(input_files[split], "r") as f:
        for line in f:
            line = json.loads(line)
            tokens = line["tokens"]
            labels = line["labels"]

            labels = [label_map[label] for label in labels]

            assert len(labels) == len(tokens)

            output_data[split].append({"tokens": tokens, "labels": labels})

# Create train, dev, and test sets
train_set = output_data["train"]
dev_set = output_data["dev"]
test_set = output_data["test"]

# Write to output files
with open(output_train_file, "w") as f:
    for item in train_set:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

with open(output_dev_file, "w") as f:
    for item in dev_set:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

with open(output_test_file, "w") as f:
    for item in test_set:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")
