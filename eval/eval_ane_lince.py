import json
import os
from pprint import pprint

import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lid_model_name = "data/models/AnE-LID"
lid_tokenizer = AutoTokenizer.from_pretrained(lid_model_name)
lid_model = AutoModelForTokenClassification.from_pretrained(lid_model_name).to(device)

ner_model_name = "/data/models/AnE-NER"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name).to(device)


def most_frequent(lst):
    freqs = {}
    most_seen = None
    max_freq = -1

    for item in lst:
        freq = freqs.get(item, 0)
        freq += 1
        freqs[item] = freq

        if freq > max_freq:
            most_seen = item
            max_freq = freq

    return most_seen


name_map = {
    "hindi-english": "hin_eng",
    "spanish-english": "spa_eng",
    "nepali-english": "nep_eng",
}

split = "test"

input_dir = f"data/corpora/lid/formatted/{split}/"
output_dir = "data/corpora/lid/linc/predictions"


def get_token_labels(word_tokens, model, tokenizer):
    subword_inputs = tokenizer(
        word_tokens, truncation=True, is_split_into_words=True, return_tensors="pt"
    ).to(device)

    subword2word = subword_inputs.word_ids(batch_index=0)

    logits = model(**subword_inputs).logits

    predictions = torch.argmax(logits, dim=2)

    predicted_subword_labels = [model.config.id2label[t.item()] for t in predictions[0]]

    predicted_word_labels = [[] for _ in range(len(word_tokens))]

    for idx, predicted_subword in enumerate(predicted_subword_labels):
        if subword2word[idx] is None:
            continue
        else:
            predicted_word_labels[subword2word[idx]].append(predicted_subword)

    predicted_word_labels = [
        most_frequent(sublist) for sublist in predicted_word_labels
    ]

    return predicted_word_labels


linc2mine = {
    "lang1": "English",
    "lang2": "notEnglish",
    "other": "Other",
    "ne": "NamedEntity",
    "mixed": "Mixed",
    "unk": "notEnglish",
    "ambiguous": "notEnglish",
    "fw": "notEnglish",
}

mine2linc = {
    "English": "lang1",
    "notEnglish": "lang2",
    "Other": "other",
    "NamedEntity": "ne",
    "Mixed": "mixed",
}

for full_name, short_name in name_map.items():
    true_data = []

    with open(os.path.join(input_dir, f"{full_name}_{split}.jsonl"), "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            true_data.append(data)

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for example in tqdm(true_data):

            word_tokens = example["tokens"]

            predicted_word_lid_labels = get_token_labels(
                word_tokens, lid_model, lid_tokenizer
            )

            predicted_word_ner_labels = get_token_labels(
                word_tokens, ner_model, ner_tokenizer
            )

            assert len(predicted_word_lid_labels) == len(predicted_word_ner_labels)

            predicted_word_labels = [
                (
                    "NamedEntity"
                    if predicted_word_ner_labels[idx] == "I"
                    else predicted_word_lid_labels[idx]
                )
                for idx in range(len(predicted_word_lid_labels))
            ]

            predicted_word_labels = [mine2linc[l] for l in predicted_word_labels]

            assert len(example["tokens"]) == len(predicted_word_labels)

            all_predicted_labels.append(predicted_word_labels)

            if "labels" in example:
                all_true_labels.append(example["labels"])

        if "labels" in true_data[0]:
            flat_true_labels = [l for sublist in all_true_labels for l in sublist]
            flat_predicted_labels = [
                l for sublist in all_predicted_labels for l in sublist
            ]

            results = {}

            precision, recall, f1, support = precision_recall_fscore_support(
                flat_true_labels, flat_predicted_labels, average="weighted"
            )

            results["Overall"] = {"P": precision, "R": recall, "F1": f1}

            labels_list = list(linc2mine.keys())

            precision, recall, f1, support = precision_recall_fscore_support(
                flat_true_labels, flat_predicted_labels, labels=labels_list
            )

            for l, p, r, f, s in zip(labels_list, precision, recall, f1, support):
                results[l] = {"P": p, "R": r, "F1": f, "support": s}

            print(short_name)
            pprint(results)

    with open(f"{output_dir}/lid_{short_name}.txt", "w") as f:
        for predicted_labels in all_predicted_labels:
            for label in predicted_labels:
                f.write(f"{label}\n")
            f.write("\n")
