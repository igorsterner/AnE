import json
import random

label_map = {
    "t": "notEnglish",
    "e": "English",
}

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

        label = label_map[label]

        current_entry["tokens"].append(token)
        current_entry["labels"].append(label)

    if current_entry["tokens"]:
        jsonl_data.append(current_entry)

    return jsonl_data

# Read the data from the .col file
with open("TR-EN CS Corpus-with language tags.txt", "r", encoding="utf-8") as file:
    data = file.readlines()

# Convert data to JSONL format
jsonl_data = convert_to_jsonl(data)

output_test_file = 'data/corpora/lid/collapsed/test/turkish-english_test.jsonl'

with open(output_test_file, 'w') as f:
    for item in jsonl_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')