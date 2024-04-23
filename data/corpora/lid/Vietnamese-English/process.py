import json

input_file = "CanVEC-mixed.json"
output_file = "data/corpora/lid/formatted/test/vietnamese-english_test.jsonl"

with open(input_file, "r") as f:
    data = json.load(f)

data_list = []

for entry in data["mixed"]:

    text = entry["clause"]
    tokens = entry["tokens"]
    labels = entry["token_langs"]

    example = {"text": text, "tokens": tokens, "labels": labels}
    data_list.append(example)

with open(output_file, "w") as f:
    for item in data_list:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")
