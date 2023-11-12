import glob
import os

from datasets import Dataset, DatasetDict

def read_data(path, label):
    row_list = []

    for filepath in glob.glob(os.path.join(path, "*.txt")):
        with open(filepath) as reader:
            row_list.append({"sentence": reader.read(), "labels": label})
    return row_list


train = Dataset.from_list(
    read_data("raw_data/train/pos", 0) + read_data("raw_data/train/neg", 1)
)
test = Dataset.from_list(
    read_data("raw_data/test/pos", 0) + read_data("raw_data/test/neg", 1)
)
test, validation = test.train_test_split(test_size=0.2, seed=42).values()

dataset = DatasetDict({"train": train, "validation": validation, "test": test})

with open("vietnamese-stopwords-dash.txt", "r") as f:
    stopwords = f.read().splitlines()


def remove_stopwords(batch):
    def remove_stopwords(sentence):
        return " ".join([word for word in sentence.split() if word not in stopwords])

    batch["processed_sentence"] = [
        remove_stopwords(sentence) for sentence in batch["sentence"]
    ]
    return batch


dataset = dataset.map(remove_stopwords, batched=True).remove_columns(["sentence"])
dataset = dataset.rename_column("processed_sentence", "sentence")
dataset.save_to_disk("data")
