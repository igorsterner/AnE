import json
import random


def convert_to_jsonl(data):
    jsonl_data = []
    current_entry = {"id": None, "tokens": [], "labels": []}

    for line in data:
        if line.strip() == "":
            if current_entry["tokens"]:
                jsonl_data.append(current_entry)
                current_entry = {"id": None, "tokens": [], "labels": []}
            continue

        parts = line.strip().split("\t")
        entry_id = int(parts[0])
        token = parts[1]
        label = parts[2]

        current_entry["id"] = entry_id
        current_entry["tokens"].append(token)
        current_entry["labels"].append(label)

    if current_entry["tokens"]:
        jsonl_data.append(current_entry)

    return jsonl_data


with open("all_cs_idtweet_norm_langid_pos.col", "r", encoding="utf-8") as file:
    data = file.readlines()

# Process data and include ID information
processed_data = convert_to_jsonl(data)

split_id = 81384160501047300

split = "train"

train_data = []
test_data = []

for datapoint in processed_data:
    if datapoint["id"] == split_id:
        split = "test"

    if split == "train":
        train_data.append(datapoint)
    else:
        test_data.append(datapoint)

dev_data = random.sample(train_data, k=int(0.1 * len(train_data)))
train_data = [entry for entry in train_data if entry not in dev_data]

with open(
    "data/corpora/lid/formatted/train/turkish-german_train.jsonl", "w", encoding="utf-8"
) as train_output_file:
    for entry in train_data:
        train_output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open(
    "data/corpora/lid/formatted/dev/turkish-german_dev.jsonl", "w", encoding="utf-8"
) as dev_output_file:
    for entry in dev_data:
        dev_output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open(
    "data/corpora/lid/formatted/test/turkish-german_test.jsonl", "w", encoding="utf-8"
) as test_output_file:
    for entry in test_data:
        test_output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
