import os
import json

lang_mapping = {
    "hineng": "hindi-english",
    "nepeng": "nepali-english",
    "spaeng": "spanish-english",
}

# set the directory path
dir_path = "/data/corpora/ner/linc"
out_path = "/data/corpora/lid/collapsed"
all_labels = []

label_map = {
    "lang1": "English",
    "lang2": "notEnglish",
    "other": "Other",
    "ne": "NamedEntity",
    "mixed": "Mixed",
    "unk": "notEnglish",
    "ambiguous": "notEnglish",
    "fw": "notEnglish",
    "en": "English",
    "hi": "notEnglish",
    "rest": "Other",
}

# loop over all directories in the main directory
for folder in os.listdir(dir_path):
    if not any([folder.endswith(key) for key in lang_mapping.keys()]):
        print(folder)
        continue

    print(folder)

    lang_code = folder.split("_")[1]
    lang1, lang2 = lang_mapping[lang_code].split("-")

    # loop over train, dev and test files in each folder
    for filename in os.listdir(os.path.join(dir_path, folder)):
        print(filename)

        split = filename.split(".")[0]  # either 'train', 'dev' or 'test'
        new_folder = os.path.join(out_path, split)
        output_filename = f"{lang1}-{lang2}-from-ner_{split}.jsonl"
        output_filepath = os.path.join(new_folder, output_filename)

        with open(os.path.join(dir_path, folder, filename), "r") as f, open(
            output_filepath, "w"
        ) as outfile:
            print(os.path.join(dir_path, folder, filename))
            print(output_filepath)

            tokens = []
            labels = []
            bios = []

            for line in f:
                # ignore the line starting with '#'
                if line.startswith("# sent_enum ="):
                    continue

                # if there is a newline indicating the end of a sentence, feed the existing tokens and labels into JSON
                if line.strip() == "":
                    record = {"tokens": tokens, "labels": labels}

                    assert len(tokens) == len(
                        labels
                    ), f"{len(tokens)}, {len(labels)}, {len(bios)}"

                    json.dump(record, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    tokens = []
                    labels = []
                    continue

                data = line.strip().split("\t")

                if split != "test":
                    token, label, bio = data
                    bios.append(bio)
                else:
                    token, label = data

                assert label in label_map, f"{label} not in {label_map}, {line}"
                label = label_map[label]
                labels.append(label)
                all_labels.append(label)

                tokens.append(token)

print(set(all_labels))
