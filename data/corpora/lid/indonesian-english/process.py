import csv
from io import StringIO
from tqdm import tqdm
import json

input_file = "id-en-code-mixed/825_Indonesian_English_CodeMixed.csv"

with open(input_file, "r", encoding="utf-8") as file:
    csv_content = file.read()

csv_content_fixed = csv_content.replace("\n list([", ",list([")

csv_content_fixed = csv_content.replace(" list([", ",list([")

csv_stream = StringIO(csv_content_fixed)


csvreader = csv.DictReader(csv_stream)
data_list = []

for row in tqdm(csvreader):
    text, tokens = eval(row["tokens"])
    labels = eval(row["langs"])

    assert len(tokens) == len(labels)

    entry = {"text": text, "tokens": tokens, "labels": labels}
    data_list.append(entry)

output_test_file = "data/corpora/lid/formatted/test/indonesian-english_test.jsonl"

with open(output_test_file, "w") as f:
    for item in data_list:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")
