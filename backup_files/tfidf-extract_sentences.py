import sys
from datasets import load_dataset, Dataset
import datatrove
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from datatrove.pipeline.readers import ParquetReader
import dataframe_image as dfi

import uuid

class _DatasetGeneratorPickleHack:
    def __init__(self, generator, generator_id=None):
        self.generator = generator
        self.generator_id = (
            generator_id if generator_id is not None else str(uuid.uuid4())
        )

    def __call__(self, *args, **kwargs):
        return self.generator(*kwargs, **kwargs)

    def __reduce__(self):
        return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))


def _DatasetGeneratorPickleHack_raise(*args, **kwargs):
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickleHack!")

"""In order to measure bias in the dataset, we consider the following simple TF-IDF based approach. The idea is that the specificity of a term -- in our case, how biased it is -- can be quantified as an inverse function of the number of documents in which it occurs.

Given a dataset and terms for a subpopulation (gender) of interest:

Evaluate Inverse Document Frequencies on the full dataset
Compute the average TF-IDF vectors for the dataset for a given subpopulation (gender)
Sort the terms by variance to see words that are much more likely to appear specifically for a given subpopulation"""


## Load fineweb
#data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb/sample/10BT", progress=True)

print("Loading dataset")
dataset = load_dataset("meg/dolma-v1_6-sample", streaming=True, split="train", data_files="https://olmo-data.org/dolma-v1_6-8B-sample/v1_5r2_sample-0000.json.gz")

woman_docs = filter(lambda doc: "woman" in doc['text'].lower().split(), dataset)
man_docs = filter(lambda doc: "man" in doc['text'].lower().split(), dataset)


#woman_docs = map(lambda doc: doc.text, filter(lambda doc: "woman" in doc.text.lower().split(), data_reader()))
#man_docs = map(lambda doc: doc["text"], filter(lambda doc: "man" in doc.text.lower().split(), data_reader()))


print("Top man sentences")
top_man_sentences = filter(lambda doc: "god" in doc['text'].lower().split() or "police" in doc['text'].lower().split() or "world" in doc['text'].lower().split(), man_docs)
man_list = [{'text':sentence.text, 'metadata':sentence.metadata} for sentence in top_man_sentences]
man_dataset = Dataset.from_list(man_list)
man_dataset.push_to_hub("meg/dolma-bias-man-sentences")

print("Top woman sentences")
top_woman_sentences = filter(lambda doc: "dating" in doc.text.lower().split() or "sex" in doc.text.lower().split() or "love" in doc.text.lower().split(), woman_docs)
woman_list = [{'text':sentence.text, 'metadata':sentence.metadata} for sentence in top_woman_sentences]
woman_dataset = Dataset.from_list(woman_list)
woman_dataset.push_to_hub("meg/dolma-bias-woman-sentences")


#data_reader = ParquetReader("meg/dolma-v1_6-sample", progress=True, limit=1000)
#corpus = map(lambda doc: doc.text, data_reader())

#print(dataset)
#corpus = map(lambda doc: doc['text'], dataset)
#corpus = map(lambda doc: doc['text'].lower().split(), dataset)
#gender_docs = filter(lambda doc: "woman" in doc or "man" in doc, corpus)



woman_docs = map(lambda doc: doc["text"], filter(lambda doc: "woman" in doc["text"].lower().split(), dataset))
man_docs = map(lambda doc: doc["text"], filter(lambda doc: "man" in doc["text"].lower().split(), dataset))


#woman_docs = map(lambda doc: doc.text, filter(lambda doc: "woman" in doc.text.lower().split(), data_reader()))
#man_docs = map(lambda doc: doc["text"], filter(lambda doc: "man" in doc.text.lower().split(), data_reader()))


print("Top man sentences")
top_man_sentences = filter(lambda doc: "god" in doc or "police" in doc or "world" in doc, man_docs)
gendered_sentences = Dataset.from_dict({"man":top_man_sentences})
gendered_sentences.push_to_hub("meg/testingi-parquet")
#print(top_man_sentences)
#for sentence in top_man_sentences:
#    print(sentence)
print("Top woman sentences")
top_woman_sentences = filter(lambda doc: "dating" in doc or "sex" in doc or "love" in doc, woman_docs)
gendered_sentences = Dataset.from_dict({"man":top_man_sentences, "woman":top_woman_sentences})
gendered_sentences.push_to_hub("meg/testingi-parquet")
sys.exit()
# Just for debugging purposes
def ChunkIterator(corpus):
    i = 0
    for c in corpus:
        i += 1
        if i > 10000:
            break
        yield c

## Compute frequencies
# Step 1: get document frequencies for the dataset. Luckily, it's an English dataset, so we can limit to English
print("Computing frequencies")
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
full_tfidf = vectorizer.fit_transform(corpus)
tfidf_feature_names = np.array(vectorizer.get_feature_names_out())

# Step 2: get average TF-IDF vectors **for each gender**
print("Getting man/woman docs")

corpus = map(lambda doc: doc['text'].lower().split(), dataset)
gender_docs = list(filter(lambda doc: "woman" in doc or "man" in doc, corpus))
woman_docs = filter(lambda doc: "woman" in doc, gender_docs)
man_docs = filter(lambda doc: "man" in doc, gender_docs)

top_man_words = filter(lambda doc: "god" in doc or "police" in doc or "world" in doc, man_docs)
print(top_man_words)

top_woman_words = filter(lambda doc: "dating" in doc or "sex" in doc or "love" in doc, woman_docs)
print(top_woman_words)
#corpus = map(lambda doc: doc['text'], dataset)
#man_docs = filter(lambda doc: "man" in doc.split(), corpus)

tfidf_by_gender = {}
tfidf_by_gender["man"] = np.asarray(vectorizer.transform(man_docs).mean(axis=0))[0]
tfidf_by_gender["woman"] = np.asarray(vectorizer.transform(woman_docs).mean(axis=0))[0]

# Step 3: for each term, compute the variance across genders
print("computing variance")
all_tfidf = np.array(list(tfidf_by_gender.values()))
tf_idf_var = all_tfidf - all_tfidf.sum(axis=0, keepdims=True)
tf_idf_var = np.power((tf_idf_var * tf_idf_var).sum(axis=0), 0.5)
sort_by_variance = tf_idf_var.argsort()[::-1]

# Create the data structure for the visualization,
# showing the highest variance words for each gender,
# and how they deviate from the mean
print("Creating output")
pre_pandas_lines = [
    {
        "word": tfidf_feature_names[w],
        "man": all_tfidf[0, w],
        "woman": all_tfidf[1, w],
        "man+": all_tfidf[0, w] - all_tfidf[:, w].mean(),
        "woman+": all_tfidf[1, w] - all_tfidf[:, w].mean(),
        "variance": tf_idf_var[w],
        "total": all_tfidf[:, w].sum(),
    }
    for w in sort_by_variance[:50]
]

### Plot results

# Plot
df = pd.DataFrame.from_dict(pre_pandas_lines)
df.to_csv("tf_idf-man_woman-dolma-v1_6-sample-output-redo.csv")
#df.style.background_gradient(
#    axis=None,
#    vmin=0,
#    vmax=0.2,
#    cmap="YlGnBu"
#).format(precision=2)
#dfi.export(df, 'out.png')
#
