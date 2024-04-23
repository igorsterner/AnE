import json
import random


def convert_to_jsonl(data):
    jsonl_data = []
    current_entry = {"tokens": [], "labels": []}

    for line in data:
        if line.strip() == "":
            if current_entry["tokens"]:
                assert len(current_entry["tokens"]) == len(current_entry["labels"])
                jsonl_data.append(current_entry)
                current_entry = {"tokens": [], "labels": []}
            continue

        parts = line.strip().split(" ")
        token = parts[0]
        label = parts[1]

        current_entry["tokens"].append(token)
        current_entry["labels"].append(label)

    if current_entry["tokens"]:
        jsonl_data.append(current_entry)

    return jsonl_data


with open("TR-EN CS Corpus-with language tags.txt", "r", encoding="utf-8") as file:
    data = file.readlines()

jsonl_data = convert_to_jsonl(data)

output_test_file = "data/corpora/lid/formatted/test/turkish-english_test.jsonl"

with open(output_test_file, "w") as f:
    for item in jsonl_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")
