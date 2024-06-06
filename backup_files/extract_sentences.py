import sys
from datasets import load_dataset, Dataset, IterableDataset, concatenate_datasets
import datatrove
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from datatrove.pipeline.readers import ParquetReader



## Load fineweb
data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb/sample/10BT", progress=True)

print("Loading dataset")
#dataset = load_dataset("meg/dolma-v1_6-sample", streaming=True, split="train", data_files="https://olmo-data.org/dolma-v1_6-8B-sample/v1_5r2_sample-0000.json.gz")
#corpus = map(lambda doc: doc.text, data_reader())
#corpus = map(lambda doc: doc['text'], dataset)


def make_dataset(subgroup, sentences):
    sentence_list = [{'text':sentence.text, 'metadata':sentence.metadata} for sentence in sentences]
    sentence_dataset = Dataset.from_list(sentence_list)
    sentence_dataset.push_to_hub("meg/fineweb-"+subgroup+"-sentences")

jewish_docs = filter(lambda doc: "jewish" in doc.text.lower().split(), data_reader())
#young_docs = filter(lambda doc: "young" in doc.text.lower().split(), data_reader())
#nonbinary_docs = filter(lambda doc: "non-binary" in doc.text.lower().split(), data_reader())

make_dataset("jewish", jewish_docs)
#make_dataset("young", young_docs)
#make_dataset("non-binary", nonbinary_docs)
sys.exit()
#woman_docs = map(lambda doc: doc.text, filter(lambda doc: "woman" in doc.text.lower().split(), data_reader()))
#man_docs = map(lambda doc: doc["text"], filter(lambda doc: "man" in doc.text.lower().split(), data_reader()))


print("Top man sentences")
top_man_sentences = filter(lambda doc: "god" in doc.text.lower().split() or "police" in doc.text.lower().split() or "world" in doc.text.lower().split(), man_docs)
man_list = [{'text':sentence.text, 'metadata':sentence.metadata} for sentence in top_man_sentences]
man_dataset = Dataset.from_list(man_list)
man_dataset.push_to_hub("meg/fineweb-bias-man-sentences")

print("Top woman sentences")
top_woman_sentences = filter(lambda doc: "dating" in doc.text.lower().split() or "sex" in doc.text.lower().split() or "love" in doc.text.lower().split(), woman_docs)
woman_list = [{'text':sentence.text, 'metadata':sentence.metadata} for sentence in top_woman_sentences]
woman_dataset = Dataset.from_list(woman_list)
woman_dataset.push_to_hub("meg/fineweb-bias-woman-sentences")
