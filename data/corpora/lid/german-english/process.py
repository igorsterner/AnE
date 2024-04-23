from code.corpus import Corpus
import json
import random

input_file = "Denglisch/corpus/Manu_corpus.csv"

corpus = Corpus(input_file)
all_tokens, all_labels = corpus.get_sentences()

output_data = []

for tokens, labels in zip(all_tokens, all_labels):
    assert len(tokens) == len(labels)
    output_data.append({"tokens": tokens, "labels": labels})

total_samples = len(output_data)
train_ratio, dev_ratio, test_ratio = 0.8, 0.1, 0.1

train_size = int(total_samples * train_ratio)
dev_size = int(total_samples * dev_ratio)

train_indices = random.sample(range(total_samples), train_size)
dev_indices = random.sample(
    list(set(range(total_samples)) - set(train_indices)), dev_size
)
test_indices = list(set(range(total_samples)) - set(train_indices) - set(dev_indices))

train_set = [output_data[i] for i in train_indices]
dev_set = [output_data[i] for i in dev_indices]
test_set = [output_data[i] for i in test_indices]

output_train_file = (
    "/data/corpora/lid/formatted/train/german-english-denglisch_train.jsonl"
)
output_dev_file = "/data/corpora/lid/formatted/dev//german-english-denglisch_dev.jsonl"
output_test_file = (
    "/data/corpora/lid/formatted/test/german-english-denglisch_test.jsonl"
)

with open(output_train_file, "w") as f:
    for item in train_set:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

with open(output_dev_file, "w") as f:
    for item in dev_set:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

with open(output_test_file, "w") as f:
    for item in test_set:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")
