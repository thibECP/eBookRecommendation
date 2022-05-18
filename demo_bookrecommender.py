import streamlit as st
from streamlit_folium import folium_static
import folium
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import pairwise_distances
from scipy.sparse import load_npz

#Useful function for breaking a long string into substrings
def split(N, text):
    chunks = [text[i:i+N] for i in range(0, len(text), N-1)]
    return chunks

#Page configuration and title
st.set_page_config(page_title="Demo eBook recommendation", layout="centered")
st.title('eBook recommender: demo')

#Import useful data
X_train_tf = load_npz("X_train_tf.npz")
X_train_tf_tsne = np.load("tsne.npy")
data = pd.read_csv("../data/kindlebookdataset/Kindle_Book_Dataset.csv")
count_vect = load("count_vect.joblib")
tf_transformer = load("tf_transformer.joblib")

xmin = np.min(X_train_tf_tsne, axis=0)
xmax = np.max(X_train_tf_tsne, axis=0)
X_train_tf_tsne = (X_train_tf_tsne - xmin) / (xmax - xmin)
titles = data["title"]
labels_words = data["url"]

#Choosing the article from which to recommend
st.subheader("Article read")
text = st.text_area("Paste an article to get a book recommendation:")

article = text.replace("\n", " ")
article_tfidf = tf_transformer.transform(count_vect.transform([article]))
dists = pairwise_distances(X_train_tf, article_tfidf, metric="cosine")[:,0]
idx_torecommend = np.argmin(dists)

if text != "":
    #Make a book recommendation, with link added
    st.subheader("Recommended book:")
    st.write("We believe that the following book would be interesting for you:")

    st.write(data["title"][idx_torecommend])
    st.write(data["url"][idx_torecommend])

    book_recommended = data["description"][idx_torecommend]
    st.write(book_recommended)
    # st.text("\n".join(split(80, book_recommended)))

    #Show map for exploration
    st.subheader("Explore")
    st.write("The 5 books that should be of most interest to you have been highlighted on the map")
    st.write("Feel free to explore around them to get other ideas")

    idx_toplot = np.argsort(dists)[5:1000]
    idx_tohighlight = np.argsort(dists)[:5]

    m = folium.Map(location=[0.5, 0.5], zoom_start=9, tiles=None)
    m.fit_bounds(bounds=[[0, 0], [1, 1]])

    for i in idx_toplot:
        folium.Marker([X_train_tf_tsne[i,0], X_train_tf_tsne[i,1]], popup=labels_words[i], tooltip=titles[i], icon=folium.DivIcon(html=f"""
                <div><svg>
                    <circle cx="5" cy="5" r="4" fill=gray opacity=".7"/>
                </svg></div>""")).add_to(m)

    for i in idx_tohighlight:
        folium.Marker([X_train_tf_tsne[i, 0], X_train_tf_tsne[i, 1]], popup=labels_words[i], tooltip=titles[i],
                      icon=folium.DivIcon(html=f"""
                <div><svg>
                    <circle cx="10" cy="10" r="8" fill=orange opacity="1."/>
                </svg></div>""")).add_to(m)

    folium.LayerControl().add_to(m)
    folium_static(m)

    #Insert hyperlink in article
    st.subheader("Inserting the hyperlink in the article")
    phrases = article.split(".")
    idx_phrase_insertion = np.argmin(
        pairwise_distances(count_vect.transform(phrases), X_train_tf[idx_torecommend], metric="cosine"))
    phrases[idx_phrase_insertion] += " ([{}]({}))".format(data["url"][idx_torecommend], data["url"][idx_torecommend])
    article_withhyperlink = ".".join(phrases)
    st.write(article_withhyperlink)