import gzip
import ijson
import os
import pandas as pd
from sklearn.metrics import pairwise_distances
import csv
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import hdbscan
import re
import math
import numpy as np
import spacy
nlp = spacy.load('en_core_web_sm')
import re
from difflib import SequenceMatcher
import urllib
import json

def get_people():
    with urllib.request.urlopen("https://www.gov.uk/api/search.json?count=0&facet_people=1500,examples:0,order:value.title") as url:
        response = json.loads(url.read().decode())
        return[person['value']['title'].lower() for person in response['facets']['people']['options']]

def get_content_for_taxon(content, taxon_id):
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    content_ids_for_taxon = list(content_taxon_mapping[content_taxon_mapping['taxon_id'] == taxon_id]['content_id'])
    return content[content['content_id'].isin(content_ids_for_taxon)];

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        # Only include alphabetical terms (otherwise it splits content based on things like year)
        if re.match("^[A-Za-z ]", item):
            stemmed.append(stemmer.stem(item))
    return stemmed

# This tokenizer limits the number of words in each entry of the corpus to 100 words
def tokenize_limit_length(text):
    text = remove_stops(text)
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    tokens = tokens[:100]
    stems = stem_tokens(tokens, stemmer)
    return stems

def tokenize(text):
    text = remove_stops(text)
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def dummy_tokenize(text):
    return text

def remove_stops(text):
    doc = nlp(text)
    for entity in doc.ents:
        # Remove any dates
        if(entity.label_ == "DATE"):
            text = re.sub(str(entity), "", text)
    text = remove_names(text)
    return text

def remove_names(text):
    for person in PEOPLE:
        potential_match = substring_may_be_name(text, person)
        if(potential_match):
            text = re.sub(potential_match, "", text)
    return text

def substring_may_be_name(text,potential_substring):
    seqMatch = SequenceMatcher(None,text,potential_substring)
    match = seqMatch.find_longest_match(0, len(text), 0, len(potential_substring))
    if any(match):
        match = text[match.a: match.a + match.size].strip()
        match_split = list(filter(None, match.split(" ")))
        # We have a match, let's see if we think it's a good candidate for being an actual name
        # There must be at least two matches (ie firstname, last name) and those must be whole word matches (not chunks)
        if len(match_split) >= 2 and all(text_chunk in text.split() for text_chunk in match_split):
            print(match)
            print(len(list(filter(None, match.split(" ")))))
            return match.strip()
        else:
            return False
    else:
        return False

def find_best_clustering(corpus):
    # This finds the 'best' ways of clustering the items based on the 'min_cluster_size'.
    # It might be fun to extend this to also optimise `max_features`
    max_features = 300
    lowest_percentage_unclassified_items = float('infinity')
    cluster_size_lowest_percentage_unclassified_items = -1
    lowest_number_items_per_topic = float('infinity')
    cluster_size_lowest_number_items_per_topic = -1
    for min_cluster_size in list(range(2,10)):
        X, vectorizer, clusterer, percentage_unclassified, number_items_per_topic = cluster(corpus, max_features, min_cluster_size)
        if percentage_unclassified < lowest_percentage_unclassified_items:
            cluster_size_lowest_percentage_unclassified_items = min_cluster_size
            lowest_percentage_unclassified_items = percentage_unclassified
        if number_items_per_topic < lowest_number_items_per_topic:
            cluster_size_lowest_number_items_per_topic = min_cluster_size
            lowest_number_items_per_topic = number_items_per_topic
    # We've collected two metrics, the min_cluster_size that leads to the lowest number of unclassified items
    # and the min_cluster_size that leads to the lowest number of items per topic (ie making each cluster as
    # specific as possible
    min_cluster_sizes = [cluster_size_lowest_percentage_unclassified_items, cluster_size_lowest_number_items_per_topic]
    # There may be a better way to do this, for the moment we just find the average of these min_cluster_sizes
    average_best_cluster_size = sum(min_cluster_sizes) / len(min_cluster_sizes)
    best_min_cluster_size = math.floor(average_best_cluster_size)
    # Finally, use this average to get the clustering and return
    X, vectorizer, clustering, percentage_unclassified, number_items_per_topic = cluster(corpus, max_features, best_min_cluster_size)
    return X, vectorizer, clustering

def cluster(corpus, max_features, min_cluster_size):
    # Use TF-IDF. Using ngrams made it a lot better, tried a little bit of grid search to see if having a higher count improved things.
    # This didn't seem to make a difference but there is probably more we could do to optimise this part of the system
    vectorizer = TfidfVectorizer(tokenizer=dummy_tokenize, analyzer='word', stop_words='english', preprocessor=dummy_tokenize, max_features=max_features, ngram_range=(1,4)  )
    X = vectorizer.fit_transform(corpus).toarray()
    # I did a grid search and having min_samples of 1 and a leaf cluster_selection_method both improved the specificity of the clusterings
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=1, cluster_selection_method='leaf')
    clusterer.fit(X)
    # Find the percentage of items that didn't get classified (a label of -1)
    total_number_items = len(corpus)
    number_unclassified_items = list(clusterer.labels_).count(-1)
    number_classified_items = total_number_items - number_unclassified_items
    percentage_unclassified = number_unclassified_items / total_number_items
    # Ratio of items to number of clusters
    number_items_per_topic = number_classified_items / (max(clusterer.labels_) + 1)
    return (X, vectorizer, clusterer, percentage_unclassified, number_items_per_topic);

def get_important_words_for_labels(output, X, vectorizer):
    # Finds the 5 most important words for each label to help understand why things are there and name taxons
    totalled_embeddings_for_label = {}
    for index, row in output.iterrows():
        bottom_label = row['bottom_label']
        if not bottom_label in totalled_embeddings_for_label:
            totalled_embeddings_for_label[bottom_label] = np.zeros((1,len(X[index])))
        totalled_embeddings_for_label[bottom_label] += X[index][:]
    num_words = 5
    feature_names = vectorizer.get_feature_names()
    for bottom_label, total_embeddings in totalled_embeddings_for_label.items():
        word_indicies = np.argpartition(total_embeddings, -num_words, axis=1)[:, -num_words:][0]
        words_for_label = []
        for index in word_indicies:
            words_for_label.append(feature_names[index])
        output.loc[output['bottom_label'] == bottom_label, 'words'] = ", ".join(words_for_label)
    return output

DATADIR = os.getenv("DATADIR")
PEOPLE = get_people()
clean_content_path = os.path.join(DATADIR, 'embedded_clean_content.pkl')
content = pd.read_pickle(clean_content_path)
content['combined_text_embedding'] = ''
content['title_embedding'] = ''
# Get content
content_for_taxon = get_content_for_taxon(content, "62fcbba5-3a75-4d15-85a6-d8a80b03d57c")
limited_length_corpus = [tokenize_limit_length(text) for text in content_for_taxon['combined_text'].to_list()]
X, vectorizer, clusterer = find_best_clustering(limited_length_corpus)
# Make an output dataframe
output = pd.DataFrame()
output['title'] = content_for_taxon['title'].to_list()
output['combined_text'] = [tokenize(text) for text in content_for_taxon['combined_text'].to_list()]
# Add the lowest level taxons we'll be making
output['bottom_label'] = clusterer.labels_
output = get_important_words_for_labels(output, X, vectorizer)
# Now we'll get the combined corpus of each of lowest level taxons we've just generated
top_label_corpus = {}
top_label_corpus_mappings = {}
for index, row in output.iterrows():
    if(row['bottom_label'] == -1):
        continue
    if row['bottom_label'] not in top_label_corpus:
        top_label_corpus[row['bottom_label']] = []
    top_label_corpus[row['bottom_label']] += row['combined_text']
corpus_mappings = list(top_label_corpus.keys())
corpus = list(top_label_corpus.values())
# Find clusterings for the combined corpus of each of these lowest level taxons
# NB: We use 'tokenize' so we DON'T limit the word count of this corpus
# (as they'll be quite long and we want to retain that information)
X, vectorizer, clusterer = find_best_clustering(corpus)
top_labels = clusterer.labels_
output['top_label'] = -2
for (index, bottom_level_label) in enumerate(corpus_mappings):
    top_level_label = top_labels[index]
    for row_index, row in output.iterrows():
        if row['bottom_label'] == bottom_level_label:
            output.at[row_index, 'top_label'] = top_level_label
# Spit it out!
output = output.sort_values(['top_label', 'bottom_label'])
output = output.drop(['combined_text'], axis=1)
output.to_csv(f"nhs_hdbscan.csv")
