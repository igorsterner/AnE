import csv
from io import StringIO
from tqdm import tqdm
import json
import random

input_file = "/data/corpora/lid/indonesian-english/id-en-code-mixed/825_Indonesian_English_CodeMixed.csv"

with open(input_file, "r", encoding="utf-8") as file:
    csv_content = file.read()

csv_content_fixed = csv_content.replace("\n list([", ",list([")
csv_content_fixed = csv_content.replace(" list([", ",list([")

csv_stream = StringIO(csv_content_fixed)


csvreader = csv.DictReader(csv_stream)
data_list = []

label_map = {"en": "English", "id": "notEnglish", "un": "Other"}

# Process entries and create a list of dictionaries
for row in tqdm(csvreader):
    text, tokens = eval(row["tokens"])
    labels = eval(row["langs"])

    labels = [label_map[label] for label in labels]

    assert len(tokens) == len(labels)

    entry = {"text": text, "tokens": tokens, "labels": labels}
    data_list.append(entry)

output_test_file = "data/corpora/lid/collapsed/test/indonesian-english_test.jsonl"

with open(output_test_file, "w") as f:
    for item in data_list:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")
