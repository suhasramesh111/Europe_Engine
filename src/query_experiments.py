import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from joblib import load

class QuickSearchEngine:
    """Lightweight search engine that loads pre-computed models once and can be reused for multiple queries"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, cluster_dir='clustering_results'):
        """Singleton pattern to ensure we only load the models once"""
        if cls._instance is None:
            cls._instance = cls(cluster_dir)
        return cls._instance
    
    def __init__(self, cluster_dir='clustering_results'):
        """Initialize the search engine using pre-computed clustering results"""
        print("Initializing search engine...")
        self.cluster_dir = cluster_dir
        self.base_dir  = os.path.dirname(os.path.abspath(__file__)) 

        # Initialize NLTK components
        self._init_nltk()
        
        # Load models and data
        self.load_models()
        self.load_cluster_data()
        
        # Create agglomerative clusters with different linkage methods
        self._create_additional_clusters()
        
        print(f"Search engine ready with {len(self.doc_ids)} documents")
    
    def _init_nltk(self):
        """Initialize NLTK resources"""
        try:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            self.stop_words.update(['http', 'https', 'www', 'com', 'html', 'php', 'asp', 'htm'])
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))
            self.stop_words.update(['http', 'https', 'www', 'com', 'html', 'php', 'asp', 'htm'])
    
    def load_models(self):
        """Load pre-computed models for query processing"""
        try:
            # Load TF-IDF vectorizer
            self.vectorizer = load(os.path.join(self.base_dir, self.cluster_dir, "tfidf_vectorizer.joblib"))
            
            # Load SVD model
            self.svd_model = load(os.path.join(self.base_dir, self.cluster_dir, "svd_model.joblib"))
            
            # Load centroids and SVD matrix
            centroids = np.load(os.path.join(self.base_dir, self.cluster_dir, "cluster_centroids.npz"))
            self.flat_centroids = centroids['flat_centroids']
            self.svd_matrix = centroids['svd_matrix']
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def load_cluster_data(self):
        """Load document-cluster mappings"""
        try:
            self.cluster_df = pd.read_csv(os.path.join(self.base_dir, self.cluster_dir, "cluster_assignments.csv"))
            self.doc_ids = self.cluster_df['doc_id'].values
            self.urls = self.cluster_df['url'].values
            self.flat_clusters = self.cluster_df['flat_cluster'].values
            self.agglom_clusters = self.cluster_df['agglom_cluster'].values
            
            # Create document maps for quick lookup
            self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
            self.url_to_idx = {url: idx for idx, url in enumerate(self.urls) if url}
        except Exception as e:
            print(f"Error loading cluster data: {e}")
            raise
    
    def _create_additional_clusters(self):
        """Create additional agglomerative clustering with different linkage methods"""
        print("Creating additional agglomerative clusters...")
        
        # Determine number of clusters - use same as flat clustering
        n_clusters = len(np.unique(self.flat_clusters))
        
        # Create agglomerative clustering with 'complete' linkage
        agglom_complete = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        self.agglom_complete_clusters = agglom_complete.fit_predict(self.svd_matrix)
        
        # Create agglomerative clustering with 'single' linkage
        agglom_single = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
        self.agglom_single_clusters = agglom_single.fit_predict(self.svd_matrix)
    
    def preprocess_query(self, query):
        """Clean and preprocess query using NLTK for consistent matching with documents"""
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Tokenize
        tokens = word_tokenize(query)
        
        # Remove stop words and stem
        processed_tokens = [self.stemmer.stem(token) for token in tokens 
                           if token not in self.stop_words and len(token) > 2]
        
        processed_query = ' '.join(processed_tokens)
        return processed_query
    
    def transform_query(self, query):
        """Transform query to same vector space as documents"""
        # Preprocess the query using NLTK
        processed_query = self.preprocess_query(query)
        
        # Transform query using same pipeline as documents
        query_vec = self.vectorizer.transform([processed_query])
        query_svd = self.svd_model.transform(query_vec)
        return query_vec, query_svd, processed_query
    
    def kmeans_search(self, query, top_n=10):
        """K-means basic search without cluster enhancement"""
        # Transform query
        query_vec, query_svd, processed_query = self.transform_query(query)
        
        # Get similarity scores - USE SVD TRANSFORMED VECTORS
        sims = cosine_similarity(query_svd, self.svd_matrix)[0]
        indices = sims.argsort()[::-1][:top_n]
        
        results = [{
            'doc_id': self.doc_ids[idx],
            'url': self.urls[idx],
            'score': float(sims[idx]),
            'cluster': int(self.flat_clusters[idx])
        } for idx in indices]
        
        return {
            'query': query,
            'processed_query': processed_query,
            'results': results
        } 
    
    def flat_cluster_search(self, query, top_n=10, alpha=0.3):
        """Perform search with flat cluster enhancement"""
        # Transform query
        query_vec, query_svd, processed_query = self.transform_query(query)
        
        # Get basic similarity scores
        basic_sims = cosine_similarity(query_svd, self.svd_matrix)[0]
        basic_indices = basic_sims.argsort()[::-1][:top_n*2]  # Get more than needed for reranking
        basic_results = [(idx, basic_sims[idx]) for idx in basic_indices]
        
        # Calculate similarity to cluster centroids
        cluster_sims = cosine_similarity(query_svd, self.flat_centroids)[0]
        
        # Find most similar cluster
        best_cluster = np.argmax(cluster_sims)
        
        # Boost scores based on cluster similarity
        enhanced = []
        for idx, score in basic_results:
            cluster_id = self.flat_clusters[idx]
            
            # Use direct similarity to centroids for flat clusters
            cluster_score = cluster_sims[cluster_id] 
                
            # Apply boosting
            boost = 1.0 + (alpha * cluster_score)
            # Extra boost for documents in the best cluster
            if cluster_id == best_cluster:
                boost *= 1.2
                
            enhanced.append((idx, score * boost))
        
        # Return top results after reranking
        enhanced.sort(key=lambda x: x[1], reverse=True)
        enhanced = enhanced[:top_n]
        
        results = [{
            'doc_id': self.doc_ids[idx],
            'url': self.urls[idx],
            'score': round(float(score), 4),
            'cluster': int(self.flat_clusters[idx])
        } for idx, score in enhanced]
        
        return {
            'query': query,
            'processed_query': processed_query,
            'results': results,
            'best_cluster': int(best_cluster)
        }
    
    def agglom_complete_search(self, query, top_n=10, alpha=0.3):
        """Perform search with hierarchical cluster enhancement using complete linkage"""
        # Transform query
        query_vec, query_svd, processed_query = self.transform_query(query)
        
        # Get basic similarity scores
        basic_sims = cosine_similarity(query_svd, self.svd_matrix)[0]
        basic_indices = basic_sims.argsort()[::-1][:top_n*2]  # Get more than needed for reranking
        basic_results = [(idx, basic_sims[idx]) for idx in basic_indices]
        
        # For agglomerative clusters, calculate similarity to all documents
        # and use as proxy for cluster similarity
        clusters = self.agglom_complete_clusters
        unique_clusters = np.unique(clusters)
        cluster_sims = np.zeros(len(unique_clusters))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_docs = np.where(clusters == cluster_id)[0]
            cluster_sims[i] = np.mean(basic_sims[cluster_docs])
            
        best_cluster = unique_clusters[np.argmax(cluster_sims)]
        
        # Boost scores based on cluster similarity
        enhanced = []
        for idx, score in basic_results:
            cluster_id = clusters[idx]
            
            # For agglom, find the cluster similarity from our computed values
            cluster_idx = np.where(unique_clusters == cluster_id)[0][0]
            cluster_score = cluster_sims[cluster_idx]
                
            # Apply boosting
            boost = 1.0 + (alpha * cluster_score)
            # Extra boost for documents in the best cluster
            if cluster_id == best_cluster:
                boost *= 1.2
                
            enhanced.append((idx, score * boost))
        
        # Return top results after reranking
        enhanced.sort(key=lambda x: x[1], reverse=True)
        enhanced = enhanced[:top_n]
        
        results = [{
            'doc_id': self.doc_ids[idx],
            'url': self.urls[idx],
            'score': round(float(score), 4),
            'cluster': int(clusters[idx])
        } for idx, score in enhanced]
        
        return {
            'query': query,
            'processed_query': processed_query,
            'results': results,
            'best_cluster': int(best_cluster)
        }
    
    def agglom_single_search(self, query, top_n=10, alpha=0.3):
        """Perform search with hierarchical cluster enhancement using single linkage"""
        # Transform query
        query_vec, query_svd, processed_query = self.transform_query(query)
        
        # Get basic similarity scores
        basic_sims = cosine_similarity(query_svd, self.svd_matrix)[0]
        basic_indices = basic_sims.argsort()[::-1][:top_n*2]  # Get more than needed for reranking
        basic_results = [(idx, basic_sims[idx]) for idx in basic_indices]
        
        # For agglomerative clusters, calculate similarity to all documents
        # and use as proxy for cluster similarity
        clusters = self.agglom_single_clusters
        unique_clusters = np.unique(clusters)
        cluster_sims = np.zeros(len(unique_clusters))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_docs = np.where(clusters == cluster_id)[0]
            cluster_sims[i] = np.mean(basic_sims[cluster_docs])
            
        best_cluster = unique_clusters[np.argmax(cluster_sims)]
        
        # Boost scores based on cluster similarity
        enhanced = []
        for idx, score in basic_results:
            cluster_id = clusters[idx]
            
            # For agglom, find the cluster similarity from our computed values
            cluster_idx = np.where(unique_clusters == cluster_id)[0][0]
            cluster_score = cluster_sims[cluster_idx]
                
            # Apply boosting
            boost = 1.0 + (alpha * cluster_score)
            # Extra boost for documents in the best cluster
            if cluster_id == best_cluster:
                boost *= 1.2
                
            enhanced.append((idx, score * boost))
        
        # Return top results after reranking
        enhanced.sort(key=lambda x: x[1], reverse=True)
        enhanced = enhanced[:top_n]
        
        results = [{
            'doc_id': self.doc_ids[idx],
            'url': self.urls[idx],
            'score': round(float(score), 4),
            'cluster': int(clusters[idx])
        } for idx, score in enhanced]
        
        return {
            'query': query,
            'processed_query': processed_query,
            'results': results,
            'best_cluster': int(best_cluster)
        }

    def display_results(self, results, title):
        """Helper function to display search results in a readable format"""
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"\n{'-' * 20} {title} {'-' * 20}")
            
        if 'query' in results:
            print(f"Query: '{results['query']}'")
            print(f"Processed query: '{results['processed_query']}'")
            
        if 'best_cluster' in results:
            print(f"Best matching cluster: {results['best_cluster']}")
            
        print("\nResults:")
        for i, res in enumerate(results['results']):
            print(f"{i+1}. Doc ID: {res['doc_id']}, Score: {res['score']}, Cluster: {res['cluster']}")
            print(f"   URL: {res['url']}")

def search_with_all_methods(query, top_n=5):
    """Run all clustering methods and display results for comparison"""
    # Get singleton instance of search engine
    engine = QuickSearchEngine.get_instance()
    
    # Run all four clustering methods
    kmeans_results = engine.kmeans_search(query, top_n)
    flat_results = engine.flat_cluster_search(query, top_n)
    agglom_complete_results = engine.agglom_complete_search(query, top_n)
    agglom_single_results = engine.agglom_single_search(query, top_n)
    
    # Display results for each method
    engine.display_results(kmeans_results, "K-means Basic Search")
    engine.display_results(flat_results, "Flat Cluster-Enhanced Search")
    engine.display_results(agglom_complete_results, "Agglomerative (Complete) Cluster-Enhanced Search")
    engine.display_results(agglom_single_results, "Agglomerative (Single) Cluster-Enhanced Search")

def main():
    """Simple command line interface for quick searching"""
    # Handle command line arguments if provided
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your search query: ")
    
    # Run search with all methods
    search_with_all_methods(query)
    
    # Interactive mode
    while True:
        print("\nEnter a new query or press Ctrl+C to exit")
        try:
            query = input("> ")
            if query.strip():
                search_with_all_methods(query)
        except KeyboardInterrupt:
            print("\nExiting...")
            break

# if __name__ == "__main__":
#     main()