import json
import os
import random

import numpy as np
import transformers
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import wandb

print(transformers.__version__)

model_checkpoint = "xlm-roberta-large"
batch_size = 32


corpora_dir = "data/corpora/ner/collapsed"

file_names = [
    "german-english-denglisch-from-lid",
    "hindi-english-from-lid",
    "hindi-english-from-ner",
    "nepali-english-from-lid",
    "spanish-english-from-lid",
    "spanish-english-from-ner",
]

language_pairs_data = {}
for file_name in file_names:
    train_file_path = os.path.join(corpora_dir, "train", f"{file_name}_train.jsonl")
    with open(train_file_path, "r") as jsonl_file:
        data_entries = [json.loads(line) for line in jsonl_file]
        # Using file_name without suffix as language pair key.
        language_pair_key = file_name.rsplit("-", 2)[0]
        if language_pair_key in language_pairs_data:
            language_pairs_data[language_pair_key] += data_entries
        else:
            language_pairs_data[language_pair_key] = data_entries

print({k: len(v) for k, v in language_pairs_data.items()})

max_entries = max(len(data) for data in language_pairs_data.values())

print("Max entries:", max_entries)
print("Max entriex x 4 = ", max_entries * 4)

upsampled_datasets = []
for language_pair, data_entries in language_pairs_data.items():
    entries_needed = max_entries - len(data_entries)
    print(f"Upsampling {language_pair} by {entries_needed} entries")
    # Upsample by random sampling with replacement
    additional_entries = random.choices(data_entries, k=entries_needed)
    upsampled_datasets.extend(data_entries + additional_entries)

print("Now have num sample = ", len(upsampled_datasets))

random.shuffle(upsampled_datasets)

temp_file_path = "data/corpora/ner/collapsed/upsampled_train.jsonl"
with open(temp_file_path, "w") as outfile:
    for entry in upsampled_datasets:
        json.dump(entry, outfile)
        outfile.write("\n")  # Add newline for each JSON entry

train_datasets = load_dataset(
    "json",
    data_files={"train": temp_file_path},
    cache_dir="data/cache/upsampled_binary_ner",
)

label2id = {
    "I": 0,
    "O": 1,
}

id2label = {v: k for k, v in label2id.items()}

dev_file_names = [
    os.path.join(corpora_dir, "dev", f"{file_name}_dev.jsonl")
    for file_name in file_names
]


dev_datasets = {
    file_name: load_dataset(
        "json", data_files={"validation": dev_file_name}, cache_dir=None
    )["validation"]
    for file_name, dev_file_name in zip(file_names, dev_file_names)
}

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
label_all_tokens = True


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        label = [label2id[l] for l in label]
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


def preprocess_dataset(dataset, tokenizer=tokenizer):

    dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Tokenize and align labels",
        load_from_cache_file=False,
    )

    return dataset


tokenized_train_datasets = preprocess_dataset(train_datasets)

tokenized_dev_datasets = {
    file_name: preprocess_dataset(dataset)
    for file_name, dataset in dev_datasets.items()
}

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label2id), id2label=id2label, label2id=label2id
)

run = wandb.init(project="AnE", entity="igorsterner")

model_name = model_checkpoint.split("/")[-1]

experiment_name = "AnE-NER"

wandb.run.name = experiment_name


args = TrainingArguments(
    output_dir="data/models/" + experiment_name,
    evaluation_strategy="steps",
    eval_steps=250,
    report_to="wandb",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
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

    # Remove ignored index (special tokens)
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

    precision, recall, f1, support = precision_recall_fscore_support(
        flat_true_predictions, flat_true_labels, pos_label="I", average="binary"
    )

    return {"precision": precision, "recall": recall, "f1": f1}


class MultiDatasetEvalCallback(TrainerCallback):
    def __init__(self, eval_datasets):
        self.eval_datasets = eval_datasets

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.eval_steps == 0:
            for file_name, eval_dataset in self.eval_datasets.items():
                metrics = trainer.evaluate(eval_dataset)
                print(f"Evaluation on {file_name} - Step: {state.global_step}")
                wandb.log(
                    {
                        f"{file_name}/eval-precision": metrics["eval_precision"],
                        f"{file_name}/eval-recall": metrics["eval_recall"],
                        f"{file_name}/eval-f1": metrics["eval_f1"],
                        "train/global_step": state.global_step,
                    }
                )


multi_dataset_eval_callback = MultiDatasetEvalCallback(tokenized_dev_datasets)


trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train_datasets["train"],
    eval_dataset=None,
    callbacks=[multi_dataset_eval_callback],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()
