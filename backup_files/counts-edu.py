import sys
import datasets
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from datatrove.pipeline.readers import ParquetReader

## Load fineweb
data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT", progress=True)
corpus = map(lambda doc: doc.text, data_reader())
## Compute frequencies
# Get document frequencies for the dataset. Luckily, it's an English dataset, so we can limit to English
# Changing the default CountVectorizer tokenization so hyphens are included as part of a word, meaning words like `non-binary` will be captured.
vectorizer =CountVectorizer(token_pattern='(?u)\\b\\w[-\\w]+\\b', min_df=2, stop_words='english')#(max_df=0.95, min_df=2, stop_words='english')
counts = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names_out()
sum_counts = np.sum(counts, axis=0)
sum_counts_list = np.asarray(sum_counts).tolist()[0]

counts_and_vocab = zip(sum_counts_list, vocab)
vocab_dict = {vocab:count for count, vocab in counts_and_vocab}
#count_vect_dense = count_vect.todense()
#vocab = vectorizer.get_feature_names_out()
#counts = np.asarray(count_vect_dense.sum(axis=0)).ravel().tolist()
#counts_and_vocab = zip(counts, vocab)
#vocab_dict = {vocab:count for count, vocab in counts_and_vocab}
with open('vocab_dict-edu.json', 'w') as f:
  json.dump(vocab_dict, f)

# make a pie chart from gender_dict
def to_pie_chart(subgroup_type, subgroup_dict, chart_title):
    labels = list(subgroup_dict.keys())
    values = list(subgroup_dict.values())
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures a circular pie chart
    plt.title(chart_title)
    fig.savefig("edu-" + subgroup_type + "_piechart.png")

## Gender
gender_dict = {"man":vocab_dict["man"], "woman":vocab_dict["woman"]}
try:
  gender_dict["non-binary"] = vocab_dict["non-binary"]
except KeyError:
  pass
to_pie_chart("gender", gender_dict, "Distribution of gender terms: 'man', 'woman', 'non-binary'")

## Religion
religion_dict = {}
for religion in ['muslim', 'christian', 'jewish', 'hindu', 'buddhist', 'atheist']:
    try:
        religion_dict[religion] = vocab_dict[religion]
    except KeyError:
        pass
to_pie_chart("religion", religion_dict, "Distribution of religion terms")# 'muslim', 'christian', 'jewish', 'hindu', 'buddhist', 'atheist'")

## Age
age_dict = {}
for age in ['young', 'old']:
    age_dict[age] = vocab_dict[age]
to_pie_chart("age", age_dict, "Distribution of age terms: 'young', 'old'")
