import os
import json

lang_mapping = {
    "hineng": "hindi-english",
    "nepeng": "nepali-english",
    "spaeng": "spanish-english",
    "msaea": "msarabic-egyptianarabic",
}

dir_path = "/data/corpora/lid/lince"

all_labels = []

for folder in os.listdir(dir_path):

    if not any([folder.endswith(key) for key in lang_mapping.keys()]):
        continue

    print(folder)

    lang_code = folder.split("_")[1]
    lang1, lang2 = lang_mapping[lang_code].split("-")

    # loop over train, dev and test files in each folder
    for filename in os.listdir(os.path.join(dir_path, folder)):

        print(filename)

        split = filename.split(".")[0]  # either 'train', 'dev' or 'test'
        new_folder = os.path.join("data/corpora/lid/formatted", split)
        output_filename = f"{lang1}-{lang2}_{split}.jsonl"
        output_filepath = os.path.join(new_folder, output_filename)

        with open(os.path.join(dir_path, folder, filename), "r") as f, open(
            output_filepath, "w"
        ) as outfile:

            print(os.path.join(dir_path, folder, filename))
            print(output_filepath)

            tokens = []
            labels = []

            for line in f:
                if line.startswith("# sent_enum ="):
                    continue

                if line.strip() == "":
                    record = {"tokens": tokens}
                    if split != "test":  # make sure it does not belong to a test set
                        record["labels"] = labels
                    json.dump(record, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    tokens = []
                    labels = []
                    continue

                data = line.strip().split("\t")

                if split != "test":
                    token, label = data
                    labels.append(label)
                    all_labels.append(label)
                else:
                    token = data[0]

                tokens.append(token)

print(set(all_labels))
