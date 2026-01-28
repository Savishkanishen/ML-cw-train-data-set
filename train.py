import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

def train_and_recommend():
    # Load dataset
    try:
        df = pd.read_csv("multilingual_song_dataset_220.csv")
    except FileNotFoundError:
        print("❌ Dataset file not found.")
        return

    # Fill missing values
    for col in ['song_name', 'artist', 'mood', 'genre', 'lyrics', 'language']:
        df[col] = df[col].fillna('')

    # Combine features (song name + artist + boosted mood + genre + lyrics)
    df['combined_features'] = (
        df['song_name'] + " " +
        df['artist'] + " " +
        (df['mood'] + " ") * 3 +
        df['genre'] + " " +
        df['lyrics']
    )

    # Vectorize
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    print("✅ Model trained successfully!")
    print("🔎 Search by song name, mood, or feeling (type 'quit' or 'exit' to close)")

    def recommend_songs(query, num_recommendations=5):
        query_lower = query.lower()

        # 1️⃣ Exact / partial song name matches (FIRST)
        name_matches = df[
            df['song_name'].str.lower().str.contains(query_lower)
        ][['song_name', 'artist']]

        # 2️⃣ ML similarity recommendations
        query_vec = tfidf.transform([query])
        similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        sorted_indices = similarity_scores.argsort()[::-1]
        ml_recommendations = df.iloc[sorted_indices][['song_name', 'artist']]

        # 3️⃣ Combine and remove duplicates
        final_results = pd.concat(
            [name_matches, ml_recommendations]
        ).drop_duplicates().head(num_recommendations)

        return final_results

    # Terminal interaction
    while True:
        try:
            if len(sys.argv) > 1:
                user_query = " ".join(sys.argv[1:])
                results = recommend_songs(user_query)
                print("\n🎵 Recommended Songs:")
                print(results.to_string(index=False))
                break

            user_query = input("\nEnter song name or mood: ")
            if user_query.lower() in ['quit', 'exit']:
                print("👋 Exiting the system. Goodbye!")
                break

            results = recommend_songs(user_query)

            print("\n🎵 Recommended Songs:")
            if results.empty:
                print("No recommendations found.")
            else:
                print(results.to_string(index=False))

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting the system. Goodbye!")
            break

if __name__ == "__main__":
    train_and_recommend()
