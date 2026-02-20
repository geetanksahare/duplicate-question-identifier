from flask import Flask, request, render_template
import pickle
import numpy as np
import re
import nltk
import distance

from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords')

STOP_WORDS = stopwords.words("english")

# Load model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))

app = Flask(__name__)

# ---------- PREPROCESS ----------
def preprocess(q):
    q = str(q).lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    return q


# ---------- FEATURE FUNCTIONS ----------
def test_common_words(q1,q2):
    w1 = set(q1.split())
    w2 = set(q2.split())
    return len(w1 & w2)

def test_total_words(q1,q2):
    w1 = set(q1.split())
    w2 = set(q2.split())
    return len(w1) + len(w2)

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

    fuzzy_features=[0.0]*4
    fuzzy_features[0]=fuzz.QRatio(q1,q2)
    fuzzy_features[1]=fuzz.partial_ratio(q1,q2)
    fuzzy_features[2]=fuzz.token_sort_ratio(q1,q2)
    fuzzy_features[3]=fuzz.token_set_ratio(q1,q2)

    return fuzzy_features


# ---------- MAIN FEATURE BUILDER ----------
def query_point_creator(q1,q2):

    input_query = []

    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # basic features
    input_query.append(len(q1))
    input_query.append(len(q2))
    input_query.append(len(q1.split()))
    input_query.append(len(q2.split()))

    cw = test_common_words(q1,q2)
    tw = test_total_words(q1,q2)

    input_query.append(cw)
    input_query.append(tw)
    input_query.append(round(cw/(tw+0.0001),2))

    # token features
    input_query.extend(test_fetch_token_features(q1,q2))

    # length features
    input_query.extend(test_fetch_length_features(q1,q2))

    # fuzzy features
    input_query.extend(test_fetch_fuzzy_features(q1,q2))

    # BOW
    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1,22),q1_bow,q2_bow))


# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    q1 = request.form["q1"]
    q2 = request.form["q2"]

    query = query_point_creator(q1,q2)

    prediction = model.predict(query)[0]

    output = "Duplicate" if prediction==1 else "Not Duplicate"

    return render_template(
        "index.html",
        prediction_text=f"Result: {output}"
    )


# ---------- RUN ----------
if __name__=="__main__":
    app.run(debug=True)
