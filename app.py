import streamlit as st
import pickle
import numpy as np
import re
import nltk
import distance
import os
import gdown

from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

# ---------------- STREAMLIT PAGE ----------------
st.set_page_config(page_title="Duplicate Question Identifier")

st.title("Duplicate Question Identifier")

# ---------------- DOWNLOAD STOPWORDS SAFELY ----------------
try:
    STOP_WORDS = stopwords.words("english")
except:
    nltk.download("stopwords")
    STOP_WORDS = stopwords.words("english")

# ---------------- DOWNLOAD MODEL IF NOT PRESENT ----------------
MODEL_PATH = "model.pkl"

if not os.path.isfile(MODEL_PATH):
    file_id = "16ovjCaHiFu5WcMq-WNBHJ0hT_M1amF9I"
    url = f"https://drive.google.com/uc?id={file_id}"
    with st.spinner("Downloading ML model... (first run only)"):
        gdown.download(url, MODEL_PATH, quiet=False)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    cv = pickle.load(open("cv.pkl", "rb"))
    return model, cv

model, cv = load_models()

# ---------- PREPROCESS ----------
def preprocess(q):
    q = str(q).lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    return q


# ---------- FEATURE FUNCTIONS ----------
def test_common_words(q1,q2):
    return len(set(q1.split()) & set(q2.split()))

def test_total_words(q1,q2):
    return len(set(q1.split())) + len(set(q2.split()))


def test_fetch_token_features(q1,q2):

    SAFE_DIV = 0.0001
    token_features = [0.0]*8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens)==0 or len(q2_tokens)==0:
        return token_features

    q1_words = set([w for w in q1_tokens if w not in STOP_WORDS])
    q2_words = set([w for w in q2_tokens if w not in STOP_WORDS])

    q1_stops = set([w for w in q1_tokens if w in STOP_WORDS])
    q2_stops = set([w for w in q2_tokens if w in STOP_WORDS])

    token_features[0] = len(q1_words & q2_words)/(min(len(q1_words),len(q2_words))+SAFE_DIV)
    token_features[1] = len(q1_words & q2_words)/(max(len(q1_words),len(q2_words))+SAFE_DIV)
    token_features[2] = len(q1_stops & q2_stops)/(min(len(q1_stops),len(q2_stops))+SAFE_DIV)
    token_features[3] = len(q1_stops & q2_stops)/(max(len(q1_stops),len(q2_stops))+SAFE_DIV)
    token_features[4] = len(q1_words)/(len(q2_words)+SAFE_DIV)
    token_features[5] = len(q2_words)/(len(q1_words)+SAFE_DIV)
    token_features[6] = len(q1_stops)/(len(q2_stops)+SAFE_DIV)
    token_features[7] = len(q2_stops)/(len(q1_stops)+SAFE_DIV)

    return token_features


def test_fetch_length_features(q1,q2):

    length_features = [0.0]*3

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens)==0 or len(q2_tokens)==0:
        return length_features

    length_features[0] = abs(len(q1_tokens)-len(q2_tokens))
    length_features[1] = (len(q1_tokens)+len(q2_tokens))/2

    strs = list(distance.lcsubstrings(q1,q2))
    length_features[2] = len(strs[0])/(min(len(q1),len(q2))+1) if strs else 0

    return length_features


def test_fetch_fuzzy_features(q1,q2):

    return [
        fuzz.QRatio(q1,q2),
        fuzz.partial_ratio(q1,q2),
        fuzz.token_sort_ratio(q1,q2),
        fuzz.token_set_ratio(q1,q2)
    ]


# ---------- MAIN FEATURE BUILDER ----------
def query_point_creator(q1,q2):

    input_query = []

    q1 = preprocess(q1)
    q2 = preprocess(q2)

    input_query.extend([
        len(q1),
        len(q2),
        len(q1.split()),
        len(q2.split())
    ])

    cw = test_common_words(q1,q2)
    tw = test_total_words(q1,q2)

    input_query.extend([cw, tw, round(cw/(tw+0.0001),2)])

    input_query.extend(test_fetch_token_features(q1,q2))
    input_query.extend(test_fetch_length_features(q1,q2))
    input_query.extend(test_fetch_fuzzy_features(q1,q2))

    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1,22),q1_bow,q2_bow))


# ---------------- STREAMLIT UI ----------------
q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Check Duplicate"):

    if q1.strip() == "" or q2.strip() == "":
        st.warning("Please enter both questions.")
    else:
        query = query_point_creator(q1,q2)
        prediction = model.predict(query)[0]

        if prediction == 1:
            st.success("Result: Duplicate Questions ✅")
        else:
            st.error("Result: Not Duplicate ❌")
