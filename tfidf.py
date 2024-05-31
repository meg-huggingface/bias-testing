import datasets
import datatrove
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from datatrove.pipeline.readers import ParquetReader
import dataframe_image as dfi

"""In order to measure bias in the dataset, we consider the following simple TF-IDF based approach. The idea is that the specificity of a term -- in our case, how biased it is -- can be quantified as an inverse function of the number of documents in which it occurs.

Given a dataset and terms for a subpopulation (gender) of interest:

Evaluate Inverse Document Frequencies on the full dataset
Compute the average TF-IDF vectors for the dataset for a given subpopulation (gender)
Sort the terms by variance to see words that are much more likely to appear specifically for a given subpopulation"""


## Load fineweb
data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb/sample/10BT", progress=True)
corpus = map(lambda doc: doc.text, data_reader())

## Compute frequencies
# Step 1: get document frequencies for the dataset. Luckily, it's an English dataset, so we can limit to English
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
full_tfidf = vectorizer.fit_transform(corpus)
tfidf_feature_names = np.array(vectorizer.get_feature_names_out())

# Step 2: get average TF-IDF vectors **for each gender**
woman_docs = map(lambda doc: doc.text, filter(lambda doc: "woman" in doc.text.split(), data_reader()))
man_docs = map(lambda doc: doc.text, filter(lambda doc: "man" in doc.text.split(), data_reader()))
tfidf_by_gender = {}
tfidf_by_gender["man"] = np.asarray(vectorizer.transform(man_docs).mean(axis=0))[0]
tfidf_by_gender["woman"] = np.asarray(vectorizer.transform(woman_docs).mean(axis=0))[0]

# Step 3: for each term, compute the variance across genders
all_tfidf = np.array(list(tfidf_by_gender.values()))
tf_idf_var = all_tfidf - all_tfidf.sum(axis=0, keepdims=True)
tf_idf_var = np.power((tf_idf_var * tf_idf_var).sum(axis=0), 0.5)
sort_by_variance = tf_idf_var.argsort()[::-1]

# Create the data structure for the visualization,
# showing the highest variance words for each gender,
# and how they deviate from the mean
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
df.to_csv("output.csv")
#df.style.background_gradient(
#    axis=None,
#    vmin=0,
#    vmax=0.2,
#    cmap="YlGnBu"
#).format(precision=2)
#dfi.export(df, 'out.png')
#
