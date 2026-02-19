import pandas as pd

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1. LOAD DATA
# =========================
data = pd.read_csv("ds.csv")

# Remove missing values
data = data.dropna(subset=["lyrics", "genre"])

# Remove duplicates
data = data.drop_duplicates(subset=["lyrics"])

data = data.reset_index(drop=True)


# =========================
# 2. COMBINE FEATURES
# =========================
# Using multiple columns improves accuracy
data["text_features"] = (
    data["lyrics"] + " " +
    data["mood"].astype(str) + " " +
    data["artist"].astype(str)
)

X = data["text_features"]
y = data["genre"]


# =========================
# 3. TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),   # use single words + word pairs
    max_features=5000
)

X_vectorized = vectorizer.fit_transform(X)


# =========================
# 4. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.3,
    stratify=y,          # keeps genre balance
    random_state=42
)


# =========================
# 5. TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# =========================
# 6. PREDICTION
# =========================
y_pred = model.predict(X_test)


# =========================
# 7. EVALUATION
# =========================
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))


# =========================
# 8. FUNCTION FOR NEW SONG
# =========================
def predict_genre(lyrics, mood="", artist=""):
    text = lyrics + " " + mood + " " + artist
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    return prediction[0]


# =========================
# 9. TEST
# =========================
new_song = "I love you forever and my heart belongs to you"

print("Predicted Genre:", predict_genre(new_song, "romantic", "Unknown"))
