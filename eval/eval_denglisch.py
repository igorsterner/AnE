import json
import os
from pprint import pprint

import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "data/models/baseline-german-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)


def most_frequent(lst):

    if len(lst) == 0:
        return "Other"

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


split = "test"

input_dir = f"data/corpora/lid/collapsed/{split}/"


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


labels_list = list(model.config.label2id.keys())

print(labels_list)

header_map = {
    "german-english-denglisch-baseline": "German--English",
}

names = list(header_map.keys())


results = {}

for name in names:

    true_data = []

    with open(os.path.join(input_dir, f"{name}_{split}.jsonl"), "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            true_data.append(data)

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for example in tqdm(true_data):
            word_tokens = example["tokens"]

            assert len(example["tokens"]) == len(example["labels"])

            predicted_word_labels = get_token_labels(word_tokens, model, tokenizer)

            assert len(example["tokens"]) == len(predicted_word_labels)

            # all_true_labels.extend(example["labels"])
            all_predicted_labels.append(predicted_word_labels)

            all_true_labels.append(example["labels"])

        flat_true_labels = [l for sublist in all_true_labels for l in sublist]
        flat_predicted_labels = [l for sublist in all_predicted_labels for l in sublist]

        print(set(flat_true_labels))
        print(set(flat_predicted_labels))

        results[name] = {}

        precision, recall, f1, support = precision_recall_fscore_support(
            flat_true_labels, flat_predicted_labels, average="weighted"
        )

        results[name]["Overall"] = {"P": precision, "R": recall, "F1": f1}

        precision, recall, f1, support = precision_recall_fscore_support(
            flat_true_labels, flat_predicted_labels, labels=labels_list
        )

        for l, p, r, f, s in zip(labels_list, precision, recall, f1, support):

            results[name][l] = {"P": p, "R": r, "F1": f, "support": s}

        print(f"{name} results")
        pprint(results[name])


def print_latex_table(
    all_results, header_map, labels=labels_list + ["Overall"], metrics=["P", "R", "F1"]
):

    print("\\begin{table*}[]")
    print("\\resizebox{\\textwidth}{!}{")
    print("".join(["\\begin{tabular}{l"] + ["rrr|"] * len(labels)) + "}")
    print("\\toprule")
    print(
        "".join([" & \multicolumn{3}{c}{\\textbf{" + label + "}} " for label in labels])
        + "\\\\"
    )

    print("".join([" & $P$    & $R$     & $F_t$   "] * len(labels)) + "\\\\ \\midrule")

    for key, result in all_results.items():
        print(f"\\textit{{{header_map[key]}}} ", end="")

        for label in labels:
            print(
                f"& {100*result[label][metrics[0]]:.2f}  & {100*result[label][metrics[1]]:.2f}  & {100*result[label][metrics[2]]:.2f}  ".replace(
                    " 0.00", " -"
                ),
                end="",
            )
            # line)
            # print(line)

        print("\\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\caption{Results for the LID task}")
    print("\\end{table*}")


print_latex_table(results, header_map)
