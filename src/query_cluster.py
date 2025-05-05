import os
import json
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from clustering import WebPageClustering

class QueryClusteringExperiment:
    def __init__(self, index_dir='Indexes_Clustering', cluster_results_dir='clustering_results', max_features=50000):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_dir = index_dir
        self.cluster_results_dir = cluster_results_dir
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.documents, self.doc_ids = self._load_documents()

    def _load_documents(self):
        cache_file = os.path.join(self.base_dir, self.index_dir, 'doc_term_cache.pkl')
        print(cache_file)

        # ✅ Try to load from cache
        if os.path.exists(cache_file):
            print("Loading documents from cache...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                return data['documents'], data['doc_ids']

        # ❌ Cache not found, compute from scratch
        print("Computing documents from b_index.json...")
        b_index_path = os.path.join(self.base_dir, self.index_dir, 'b_index.json')
        with open(b_index_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        term_to_postings = {
            (entry := json.loads(line))['term']: entry
            for line in lines
        }

        doc_term_map = {}
        for term, data in term_to_postings.items():
            for posting in data['Postings']:
                doc_id = posting['id']
                if doc_id not in doc_term_map:
                    doc_term_map[doc_id] = []
                doc_term_map[doc_id].extend([term] * posting['tf'])

        doc_ids = sorted(doc_term_map.keys())
        documents = [" ".join(doc_term_map[doc_id]) for doc_id in doc_ids]

        # ✅ Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump({'documents': documents, 'doc_ids': doc_ids}, f)

        print("Cached document-term map.")
        return documents, doc_ids

    def get_top_n(self, query_vec, tfidf_matrix, top_n=10):
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        ranked_indices = similarities.argsort()[::-1][:top_n]
        return [(i, similarities[i]) for i in ranked_indices]

    def compute_flat_centroids(self, tfidf_matrix, flat_clusters):
        flat_centroids = []
        for i in range(flat_clusters.max() + 1):
            indices = np.where(flat_clusters == i)[0]
            if indices.shape[0] > 0:
                sub = tfidf_matrix[indices].mean(axis=0)
                flat_centroids.append(np.asarray(sub).ravel() if hasattr(tfidf_matrix, 'toarray') else sub)
            else:
                flat_centroids.append(np.zeros(tfidf_matrix.shape[1]))
        return np.array(flat_centroids)

    def run(self, queries, top_n=10, cluster_type='flat', alpha=0.3):
        print(f"\nRunning experiments with cluster type: {cluster_type}")
        cluster_df = pd.read_csv(os.path.join(self.base_dir, self.cluster_results_dir, 'cluster_assignments.csv'))
        cluster_col = f"{cluster_type}_cluster" if cluster_type == "flat" else cluster_type

        tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}

        n_clusters = len(cluster_df[cluster_col].unique())
        print("Initializing WebPageClustering..")
        clusterer = WebPageClustering(index_dir=self.index_dir, n_clusters=n_clusters)
        clusterer.matrix = tfidf_matrix.toarray() if hasattr(tfidf_matrix, 'toarray') else tfidf_matrix
        clusterer.doc_ids = self.doc_ids
        clusterer.flat_clusters = cluster_df['flat_cluster'].values
        clusterer.agglom_clusters_combined = cluster_df['agglom_on_kmeans'].values

        print("Computing Dense Centroids for Kmeans..")
        clusterer.flat_centroids = self.compute_flat_centroids(tfidf_matrix, clusterer.flat_clusters)

        results = []
        for qid, query in enumerate(queries):
            query_vec = self.vectorizer.transform([query])
            original = self.get_top_n(query_vec, tfidf_matrix, top_n=top_n)

            enhanced = clusterer.enhance_relevance_with_clustering(
                query_vec, original, cluster_type=cluster_type, alpha=alpha
            )

            results.append({
                'query_id': qid,
                'query': query,
                'original_doc_ids': [self.doc_ids[i] for i, _ in original],
                'clustered_doc_ids': [self.doc_ids[i] for i, _ in enhanced]
            })

        for row in results:
            print(row['clustered_doc_ids'])

        print("Clustering Done.")
        return results


# Example usage:
if __name__ == '__main__':
    experiment = QueryClusteringExperiment()
    queries = ["Oktoberfest"]
    experiment.run(queries, cluster_type='flat')
