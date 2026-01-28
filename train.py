import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

def train_and_recommend():
    # Load the dataset
    try:
        df = pd.read_csv('multilingual_song_dataset_220.csv')
    except FileNotFoundError:
        print("Error: Dataset file 'multilingual_song_dataset_220.csv' not found.")
        return

    # Data Preprocessing
    # Handle missing values if any (replace NaN with empty string)
    df['mood'] = df['mood'].fillna('')
    df['genre'] = df['genre'].fillna('')
    df['lyrics'] = df['lyrics'].fillna('')

    # Create a combined feature column for content-based filtering
    # We give more weight to 'mood' (repeating 3 times) so it dominates recommendations
    df['combined_features'] = (df['mood'] + " ")*3 + df['genre'] + " " + df['lyrics']

    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    # Fit and transform the data
    try:
        tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    except ValueError as e:
        print(f"Error during vectorization: {e}")
        return

    print("Model trained successfully! (Type 'quit' to exit)")

    def recommend_songs(query, num_recommendations=5):
        # Vectorize the user query
        query_vec = tfidf.transform([query])

        # Calculate cosine similarity between query and all songs
        similarity_scores = cosine_similarity(query_vec, tfidf_matrix)

        # Get the indices of the most similar songs (flatten to 1D array)
        similarity_scores = similarity_scores.flatten()

        # Sort indices by score in descending order and get top N
        top_indices = similarity_scores.argsort()[::-1][:num_recommendations]

        # Fetch the details
        recommendations = df.iloc[top_indices][['song_name', 'artist', 'mood', 'genre', 'language']]
        
        return recommendations

    # Run one test immediately to verify it works without interaction if needed, 
    # but the user asked for a loop presumably. 
    # Let's stick to the interactive loop as requested.
    while True:
        try:
            if len(sys.argv) > 1: # If arguments provided, use them and exit
                user_query = " ".join(sys.argv[1:])
                print(f"\nQuery from args: {user_query}")
                results = recommend_songs(user_query)
                print(results.to_string(index=False))
                break
                
            user_query = input("\nEnter a mood or description (or 'quit' to exit): ")
            if user_query.lower() == 'quit':
                break
            
            results = recommend_songs(user_query)
            
            print(f"\nTop recommendations for '{user_query}':")
            if results.empty:
                print("No recommendations found.")
            else:
                print(results.to_string(index=False))
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    train_and_recommend()
