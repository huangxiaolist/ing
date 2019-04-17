import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth',200)
from sklearn.datasets import fetch_20newsgroups
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import umap

# document'shape is a list of [sentences]
# return a dict key=sentence , value=topic
def latent_semantic_analysis(document, labels, vectorizer=TfidfVectorizer(max_features=50, max_df=0.5, smooth_idf=True),
                             svd_model=TruncatedSVD(n_components=20, n_iter=100, random_state=41), view_method='umap'):

    if document is None:
        raise ValueError("the input data's shape is a list of [[n_samples]]")
    new_df = pd.DataFrame({'document': document})
    # clean the data:transfer punctuation\lower\len(w)>3
    new_df['clean_doc'] = new_df['document'].str.replace("^a-zA-Z#", " ")
    new_df['clean_doc'] = new_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split(r'\s+') if len(w) > 3]))
    new_df['clean_doc'] = new_df['clean_doc'].apply(lambda x: x.lower())
    # stop_words
    stop_words = stopwords.words('english')
    tokenized_doc = new_df['clean_doc'].apply(lambda x: x.split())
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    detokenized_dox = []
    for i in range(len(new_df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_dox.append(t)
    new_df['clean_doc'] = detokenized_dox
    X = vectorizer.fit_transform(new_df['clean_doc'])
    svd_model.fit(X)
    x_topics = svd_model.fit_transform(X)
    doucment_topic_list = []
    for i in range(len(x_topics)):
        d_t_dict = {}
        topic_prob = x_topics[i]
        topic_prob = topic_prob.tolist()
        topic_index = topic_prob.index(max(topic_prob))
        d_t_dict[new_df['document'][i]] = labels[topic_index]
        doucment_topic_list.append(d_t_dict)
    print(doucment_topic_list[0])
    if view_method != '':
        if view_method == 'umap':
            target = [i for i in range(len(labels))]
            embedding = umap.UMAP(n_neighbors=150, min_dist=0.5).fit_transform(x_topics)
            plt.figure(figsize=(7, 5))
            plt.scatter(embedding[:, 0], embedding[:, 1], c=target, s=10, edgecolors='none')
            plt.title("dot is document,the umap of the document")
            plt.show()


# import the data
dataset = fetch_20newsgroups(shuffle=False, random_state=41, remove=('headers', 'footers', 'quotes'))
document = dataset.data
labels = dataset.target_names
latent_semantic_analysis(document, labels, view_method='')
