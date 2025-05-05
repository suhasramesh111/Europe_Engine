import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from collections import defaultdict
from joblib import dump

class WebPageClustering:
    def __init__(self, index_dir='Indexes_Clustering', n_clusters=10, max_features=10000, output_dir='clustering_results'):
        self.index_dir = index_dir
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.base_dir  = os.path.dirname(os.path.abspath(__file__))

        # Initialize NLP components
        self._init_nltk()
        
        # Initialize vectorizer with fixed params
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=2,
            max_df=0.85,
            stop_words='english',
            ngram_range=(1, 2),
            norm='l2'
        )
        
        # Attributes to be filled during clustering
        self.documents = []
        self.doc_ids = []
        self.urls = []
        self.matrix = None
        self.svd = None
        self.svd_matrix = None
        self.flat_clusters = None
        self.flat_centroids = None
        self.agglom_clusters = None

    def _init_nltk(self):
        """Initialize NLTK resources"""
        try:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            self.stop_words.update(['http', 'https', 'www', 'com', 'html'])
        except LookupError:
            # Handle NLTK resource download if necessary
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))
            self.stop_words.update(['http', 'https', 'www', 'com', 'html'])

    def clean_text(self, text):
        """Preprocess text using NLTK for tokenization and stemming"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def load_documents(self):
        """Load and process documents from index"""
        # Load document-term data
        doc_term_map = defaultdict(list)
        with open(os.path.join(self.base_dir, self.index_dir, 'b_index.json'), 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                for posting in entry['Postings']:
                    doc_id = posting['id']
                    doc_term_map[doc_id].extend([entry['term']] * posting['tf'])
        
        # Load URLs
        url_df = pd.read_csv(os.path.join(self.base_dir, self.index_dir, 'id_link.csv'))
        id_col = url_df.columns[0]  # Assume first column is ID
        url_col = next((c for c in url_df.columns if 'url' in c.lower()), url_df.columns[1])
        url_map = dict(zip(url_df[id_col], url_df[url_col]))
        
        # Prepare final documents
        doc_ids = sorted(doc_term_map.keys())
        documents = [self.clean_text(" ".join(doc_term_map[doc_id])) for doc_id in doc_ids]
        urls = [url_map.get(doc_id, "") for doc_id in doc_ids]
        
        return documents, doc_ids, urls

    def vectorize_documents(self):
        """Vectorize documents using TF-IDF"""
        print(f"Vectorizing {len(self.documents)} documents...")
        self.matrix = self.vectorizer.fit_transform(self.documents)
        return self.matrix

    def reduce_dimensions(self):
        """Perform dimensionality reduction with SVD"""
        print("Performing dimensionality reduction...")
        n_components = min(200, self.matrix.shape[1] - 1)  # Ensure n_components is valid
        self.svd = TruncatedSVD(n_components=n_components)
        self.svd_matrix = self.svd.fit_transform(self.matrix)
        return self.svd_matrix

    def perform_kmeans_clustering(self):
        """Perform K-means clustering"""
        print(f"Clustering into {self.n_clusters} clusters...")
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=1000)
        self.flat_clusters = kmeans.fit_predict(self.svd_matrix)
        self.flat_centroids = kmeans.cluster_centers_
        return self.flat_clusters, self.flat_centroids

    def perform_hierarchical_clustering(self):
        """Perform hierarchical sub-clustering"""
        print("Performing hierarchical sub-clustering...")
        self.agglom_clusters = np.zeros(len(self.flat_clusters), dtype=int)
        current_label = 0
        
        for cluster_id in range(self.n_clusters):
            indices = np.where(self.flat_clusters == cluster_id)[0]
            if len(indices) > 5:  # Only split large clusters
                n_subclusters = min(5, len(indices)//2)
                if n_subclusters >= 2:  # Ensure we have at least 2 subclusters
                    agglom = AgglomerativeClustering(n_clusters=n_subclusters)  
                    sub_labels = agglom.fit_predict(self.svd_matrix[indices])
                    self.agglom_clusters[indices] = sub_labels + current_label
                    current_label += sub_labels.max() + 1
                else:
                    self.agglom_clusters[indices] = current_label
                    current_label += 1
            else:
                self.agglom_clusters[indices] = current_label
                current_label += 1
        
        return self.agglom_clusters

    def extract_cluster_keywords(self, top_n=10):
        """Extract keywords that define each cluster"""
        print("Extracting cluster keywords...")
        cluster_keywords = {}
        
        # Get feature names from vectorizer
        feature_names = self.vectorizer.get_feature_names_out()
        
        for cluster_id in range(self.n_clusters):
            # Get documents in this cluster
            indices = np.where(self.flat_clusters == cluster_id)[0]
            if len(indices) == 0:
                continue
                
            # Average TF-IDF scores for this cluster
            cluster_tfidf = self.matrix[indices].mean(axis=0).A1
            
            # Get top terms for this cluster
            top_indices = cluster_tfidf.argsort()[-top_n:][::-1]
            top_terms = [(feature_names[i], float(cluster_tfidf[i])) for i in top_indices]
            
            # Store in dictionary
            cluster_keywords[str(cluster_id)] = top_terms
        
        # Save cluster keywords
        with open(os.path.join(self.base_dir, self.output_dir, "cluster_keywords.json"), 'w') as f:
            json.dump(cluster_keywords, f)
            
        print(f"Saved keywords for {len(cluster_keywords)} clusters")
        return cluster_keywords

    def save_results(self):
        """Save clustering outputs"""
        # Save cluster assignments
        cluster_df = pd.DataFrame({
            'doc_id': self.doc_ids,
            'url': self.urls,
            'flat_cluster': self.flat_clusters,
            'agglom_cluster': self.agglom_clusters
        })
        cluster_df.to_csv(os.path.join(self.base_dir, self.output_dir, "cluster_assignments.csv"), index=False)
        print(f"Saved cluster assignments for {len(self.doc_ids)} documents")
        
        # Save centroids and SVD matrix for quick search
        np.savez(os.path.join(self.base_dir, self.output_dir, "cluster_centroids.npz"),
                flat_centroids=self.flat_centroids,
                svd_matrix=self.svd_matrix)
        print("Saved cluster centroids and SVD matrix")

        # Save vectorizer and SVD model for future use
        dump(self.vectorizer, os.path.join(self.base_dir, self.output_dir, "tfidf_vectorizer.joblib"))
        dump(self.svd, os.path.join(self.base_dir, self.output_dir, "svd_model.joblib"))
        print("Saved TF-IDF vectorizer and SVD model")

    def perform_clustering(self):
        """Main clustering workflow with separated functions"""
        print("Loading documents...")
        self.documents, self.doc_ids, self.urls = self.load_documents()
        
        # Vectorize documents
        self.vectorize_documents()
        
        # Reduce dimensions
        self.reduce_dimensions()
        
        # Perform K-means clustering
        self.perform_kmeans_clustering()
        
        # Perform hierarchical clustering
        self.perform_hierarchical_clustering()
        
        # Extract cluster keywords
        self.extract_cluster_keywords()
        
        # Save all results
        self.save_results()
        
        print("Clustering complete!")
        return {
            "num_documents": len(self.documents),
            "num_clusters": self.n_clusters,
            "flat_clusters": self.flat_clusters,
            "agglom_clusters": self.agglom_clusters
        }


if __name__ == "__main__":
    print("Web Page Clustering Tool")
    try:
        n_clusters = int(input("Enter number of clusters (default: 10): ") or "10")
    except ValueError:
        n_clusters = 10
    
    print(f"Starting clustering with k={n_clusters}")
    clusterer = WebPageClustering(n_clusters=n_clusters, max_features=10000)
    result = clusterer.perform_clustering()
    print(f"Clustering complete with {result['num_documents']} documents into {result['num_clusters']} clusters!")