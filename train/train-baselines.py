import argparse
import os

import numpy as np
import transformers
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import wandb

args = argparse.ArgumentParser()
args.add_argument("--file_name", type=str, default="data/train.json")
args = args.parse_args()

model_checkpoint = "xlm-roberta-large"
batch_size = 32


formatted_corpora_dir = "data/corpora/lid/formatted"
collapsed_corpora_dir = "data/corpora/lid/collapsed"


upsample_ratio = {
    "german": 16.242865636147442,
    "hindi": 11.329255650010367,
    "nepali": 6.465625369778724,
    "spanish": 1.0,
}

language = args.file_name.split("-")[0]

if language == "german":
    corpus_dir = collapsed_corpora_dir
else:
    corpus_dir = formatted_corpora_dir


datasets = load_dataset(
    "json",
    data_files={
        "train": os.path.join(corpus_dir, "train", f"{args.file_name}_train.jsonl"),
        "validation": os.path.join(corpus_dir, "dev", f"{args.file_name}_dev.jsonl"),
    },
)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

label_all_tokens = True

if language == "german":
    label2id = {
        "English": 0,
        "notEnglish": 1,
        "Mixed": 2,
        "Other": 3,
        "NamedEntity": 4,
    }
else:
    label2id = {
        "lang1": 0,
        "lang2": 1,
        "mixed": 2,
        "other": 3,
        "ne": 4,
        "fw": 5,
        "unk": 6,
        "ambiguous": 7,
    }

id2label = {v: k for k, v in label2id.items()}


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        label = [label2id.get(l, -100) for l in label]
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def preprocess_dataset(dataset):

    dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Tokenize and align labels",
        load_from_cache_file=False,
    )

    return dataset


tokenized_datasets = preprocess_dataset(datasets)


model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label2id), id2label=id2label, label2id=label2id
)

run = wandb.init(project="AnE", entity="igorsterner")

model_name = model_checkpoint.split("/")[-1]

experiment_name = f"baseline-{language}"

wandb.run.name = experiment_name

args = TrainingArguments(
    output_dir="data/models/" + experiment_name,
    evaluation_strategy="epoch",
    report_to="wandb",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=int(3 * upsample_ratio[language]),
    weight_decay=0.01,
    push_to_hub=True,
    hub_private_repo=True,
    save_total_limit=1,
    save_strategy="epoch",
    load_best_model_at_end=False,
)

data_collator = DataCollatorForTokenClassification(tokenizer)


def compute_metrics(p):

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    flat_true_predictions = [item for sublist in true_predictions for item in sublist]

    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    results = {}

    precision, recall, f1, support = precision_recall_fscore_support(
        flat_true_labels, flat_true_predictions, average="weighted"
    )

    results["Overall"] = {"P": precision, "R": recall, "F1": f1}

    labels_list = list(label2id.keys())

    precision, recall, f1, support = precision_recall_fscore_support(
        flat_true_labels, flat_true_predictions, labels=labels_list
    )

    for l, p, r, f, s in zip(labels_list, precision, recall, f1, support):
        results[l] = {"P": p, "R": r, "F1": f, "support": s}

    return results


trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()
