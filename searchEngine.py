import pandas as pd
import json
import string
from sortedcontainers import SortedDict
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from itertools import combinations
import os


class engine:
    """
    Search engine class that supports BM25 ranking with proximity-based boosting.
    """
    
    def __init__(self):
        """Initialize the search engine by loading indices and configuration parameters."""
        base_dir  = os.path.dirname(os.path.abspath(__file__))
        b_index_path   = os.path.join(base_dir, "Indexes", "b_index.json")
        h_index_path   = os.path.join(base_dir, "Indexes", "h_index.json")
        t_index_path   = os.path.join(base_dir, "Indexes", "t_index.json")
        id_link_path   = os.path.join(base_dir, "Indexes", "id_link.csv")
        doc_norms_path = os.path.join(base_dir, "Indexes", "doc_norms.csv")
        pagerank_path  = os.path.join(base_dir, "Indexes", "pagerank.csv")

        self.b_index = self._load_index(b_index_path)
        self.h_index = self._load_index(h_index_path)
        self.t_index = self._load_index(t_index_path)
        
        self.id_link   = self._load_index(id_link_path)
        self.doc_norms = self._load_index(doc_norms_path)
        try:
            self.pagerank = self._load_index(pagerank_path)
        except Exception as e:
            print(f"Warning: Could not load PageRank data: {e}")
            self.pagerank = {}
        
        self.stop_words = set(stopwords.words('english'))
        self.wnl = WordNetLemmatizer()
        self.punctuation = string.punctuation
        
        self.b_contri = 1/6 # contribution weight for body index
        self.h_contri = 1/3 # contribution weight for headings index
        self.t_contri = 1/2 # contribution weight for title index
        
        self.k1 = 1.5
        self.b = 0.75
        
        self.results = SortedDict() # Use SortedDict instead of heap
        self.max_links = 1000
        
        self.proximity_weight_b = 0.5
        self.proximity_weight_h = 0.3
        self.proximity_weight_t = 0.5 
        self.proximity_scale = 10.0
        
        self.bm25_weight = 0.7      # Weight for BM25 score
        self.pagerank_weight = 0.3  # Weight for PageRank score
        
        # Keep track of which results have been returned
        self.current_result_index = 0
    
    def _load_index(self, index):
        """Helper to load different index files (CSV or JSON)."""
        
        name, ext = index.split('.')
        
        if ext=='json':
            index_df = pd.read_json(index, lines = True)
            index_dict = {}
            for _, row in index_df.iterrows():
                term = row['term']
                postings = row['Postings']
                if isinstance(postings, str):
                    postings = json.loads(postings.replace("'", "\""))
                    
                index_dict[term] = {
                    'postings': postings,
                    'df': row['df'],
                    'idf': row['idf']
                    }
            return index_dict
        
        else:
            base_name = name.split('\\')[-1]
            
            if base_name == 'doc_norms':
                index_df = pd.read_csv(index)
                index_dict = index_df.set_index('id')[['doc_norm_b', 'doc_norm_t', 'doc_norm_h']].to_dict(orient='index')
                return index_dict
            
            elif base_name == 'id_link':
                index_df = pd.read_csv(index)
                index_dict = dict(zip(index_df.iloc[:, 0].astype(int), index_df.iloc[:, 1]))
                return index_dict
            
            elif base_name == 'pagerank':
                index_df = pd.read_csv(index)
                index_dict = dict(zip(index_df.iloc[:, 0].astype(int), index_df.iloc[:, 1]))
                return index_dict
            
            else:
                index_df = pd.read_csv(index)
                index_dict = dict(zip(index_df.iloc[:, 0], index_df.iloc[:, 1]))
                return index_dict
            
        
    def search(self, query, use_pagerank=False):
        """Search for the given query and rank documents."""
        print("Search Function : ",query)
        self.results = SortedDict()  # Reset results as SortedDict
        self.current_result_index = 0  # Reset result index
        terms = self._process_query(query)
        if not terms:
            return
        self._rank_docs(terms, use_pagerank)
    
    
    def retrieve_new_res(self, k=10):
        """Retrieve top k ranked results with their scores."""
        
        final_docs = []
        results_items = list(self.results.items())    
        start_idx = self.current_result_index
        end_idx = min(start_idx + k, len(results_items))
        
        for i in range(start_idx, end_idx):
            neg_score, doc_id = results_items[i]
            link = self.id_link.get(doc_id)
            if link:
                final_docs.append((link, -neg_score))
        
        self.current_result_index = end_idx
        print(self.current_result_index)
        print("Ret : ",final_docs)
        
        return final_docs


    def _process_query(self, q):
        """Tokenize, clean, and lemmatize the input query."""
        
        tokens = word_tokenize(q.lower())
        return [self.wnl.lemmatize(t) for t in tokens if t not in self.stop_words and t not in self.punctuation]
    
    
    def _rank_docs(self, terms, use_pagerank=False):
        """Rank documents using BM25 and proximity boosting."""
        
        all_docs = self._score_docs(terms, use_pagerank)
        for doc_id, score in all_docs.items():
            self.results[-score] = doc_id
            if len(self.results) > self.max_links:
                self.results.popitem(index=-1)
    
    
    def _score_docs(self, terms, use_pagerank=False):
        """Compute document scores as a weighted average of BM25 and PageRank when enabled."""
        
        res = {}
        term_positions_b = {}
        term_positions_h = {}
        term_positions_t = {}
        
        for term in terms:
            if term in self.b_index:
                for doc in self.b_index[term]['postings']:
                    doc_id = doc['id']
                    idf_ = self.b_index[term]['idf']
                    doc_norm = self.doc_norms[doc['id']]['doc_norm_b']
                    tf_ = doc['tf']
                    positions = doc['positions']
                    
                    bm = self._calc_bm(idf_, doc_norm, tf_)
                    res[doc_id] = res.get(doc_id, 0) + self.b_contri * bm
                    
                    if doc_id not in term_positions_b:
                        term_positions_b[doc_id] = {}
                    if term not in term_positions_b[doc_id]:
                        term_positions_b[doc_id][term] = []
                    term_positions_b[doc_id][term].extend(positions)
            
            if term in self.h_index:
                for doc in self.h_index[term]['postings']:
                    doc_id = doc['id']
                    idf_ = self.h_index[term]['idf']
                    doc_norm = self.doc_norms[doc['id']]['doc_norm_h']
                    tf_ = doc['tf']
                    positions = doc['positions']
                    
                    bm = self._calc_bm(idf_, doc_norm, tf_)
                    res[doc_id] = res.get(doc_id, 0) + self.h_contri * bm
                    
                    if doc_id not in term_positions_h:
                        term_positions_h[doc_id] = {}
                    if term not in term_positions_h[doc_id]:
                        term_positions_h[doc_id][term] = []
                    term_positions_h[doc_id][term].extend(positions)
                    
            if term in self.t_index:
                for doc in self.t_index[term]['postings']:
                    doc_id = doc['id']
                    idf_ = self.t_index[term]['idf']
                    doc_norm = self.doc_norms[doc['id']]['doc_norm_t']
                    tf_ = doc['tf']
                    positions = doc['positions']
                    
                    bm = self._calc_bm(idf_, doc_norm, tf_)
                    res[doc_id] = res.get(doc_id, 0) + self.t_contri * bm
                    
                    if doc_id not in term_positions_t:
                        term_positions_t[doc_id] = {}
                    if term not in term_positions_t[doc_id]:
                        term_positions_t[doc_id][term] = []
                    term_positions_t[doc_id][term].extend(positions)
        
        doc_ids = list(res.keys())
        bm25_scores = np.array(list(res.values()))
        boosts_b = np.ones(len(doc_ids))
        boosts_h = np.ones(len(doc_ids))
        boosts_t = np.ones(len(doc_ids))
        
        for i, doc_id in enumerate(doc_ids):
            if doc_id in term_positions_b:
                boosts_b[i] = self._calc_proximity_boost(terms, term_positions_b[doc_id])
            if doc_id in term_positions_h:
                boosts_h[i] = self._calc_proximity_boost(terms, term_positions_h[doc_id])
            if doc_id in term_positions_t:
                boosts_t[i] = self._calc_proximity_boost(terms, term_positions_t[doc_id])
        
        bm25_scores = bm25_scores * (
            1 + 
            self.proximity_weight_b * (boosts_b - 1) +
            self.proximity_weight_h * (boosts_h - 1) +
            self.proximity_weight_t * (boosts_t - 1)
        )
        
        if use_pagerank and self.pagerank:
            final_scores = self.bm25_weight * bm25_scores
            pagerank_scores = np.zeros(len(doc_ids))
            for i, doc_id in enumerate(doc_ids):
                if doc_id in self.pagerank:
                    pagerank_scores[i] = self.pagerank[doc_id]
            final_scores = final_scores + self.pagerank_weight * pagerank_scores
        else:
            final_scores = bm25_scores
        
        return {doc_id: score for doc_id, score in zip(doc_ids, final_scores)}
    
    
    def _calc_proximity_boost(self, terms, doc_term_positions):
        """Compute a proximity boost factor based on distances between query terms - optimized."""
        
        if len(terms) < 2 or not doc_term_positions:
            return 1.0
        
        total_boost = 0.0
        pair_count = 0
        term_pairs = list(combinations(terms, 2))
        
        available_terms = set(doc_term_positions.keys())
        valid_pairs = [(t1, t2) for t1, t2 in term_pairs if t1 in available_terms and t2 in available_terms]
        
        if not valid_pairs:
            return 1.0
            
        for term1, term2 in valid_pairs:
            pos1 = doc_term_positions[term1]
            pos2 = doc_term_positions[term2]
            
            if not all(pos1[i] <= pos1[i+1] for i in range(len(pos1)-1)):
                pos1.sort()
            if not all(pos2[i] <= pos2[i+1] for i in range(len(pos2)-1)):
                pos2.sort()
            
            i, j = 0, 0
            min_dist = float('inf')
            while i < len(pos1) and j < len(pos2):
                dist = abs(pos1[i] - pos2[j])
                min_dist = min(min_dist, dist)
                
                if pos1[i] < pos2[j]:
                    i += 1
                else:
                    j += 1
                    
            if min_dist < float('inf'):
                boost = 1 / (1 + min_dist / self.proximity_scale)
                total_boost += boost
                pair_count += 1
        
        return (total_boost / pair_count) if pair_count > 0 else 1.0
    
    def _calc_bm(self, idf, doc_norm, tf):
        """Compute BM25 score component."""
        
        return idf * ((self.k1 + 1) * tf)/(self.k1*((1-self.b)+self.b*doc_norm)+tf)
        
    def check(self):
        """Print samples of loaded indices for verification."""
        
        print("b_index:")
        for i, (key, value) in enumerate(self.b_index.items()):
            if i < 5:
                print(f"{key}: {value}")
            else:
                break
        print()
        
        print("h_index:")
        for i, (key, value) in enumerate(self.h_index.items()):
            if i < 5:
                print(f"{key}: {value}")
            else:
                break
        print()
        
        print("t_index:")
        for i, (key, value) in enumerate(self.t_index.items()):
            if i < 5:
                print(f"{key}: {value}")
            else:
                break
        print()
        
        print("id_link:")
        for i, (key, value) in enumerate(self.id_link.items()):
            if i < 5:
                print(f"{key}: {value}")
            else:
                break
        print()
        
        print("doc_norms:")
        for i, (key, value) in enumerate(self.doc_norms.items()):
            if i < 5:
                print(f"ID {key}:")
                print(f"  doc_norm_b: {value['doc_norm_b']}")
                print(f"  doc_norm_t: {value['doc_norm_t']}")
                print(f"  doc_norm_h: {value['doc_norm_h']}")
            else:
                break
        print()
        
        if self.pagerank:
            print("PageRank:")
            ranks = list(self.pagerank.items())
            ranks.sort(key=lambda x: x[1], reverse=True)
            for i, (doc_id, rank) in enumerate(ranks[:5]):
                print(f"Doc ID {doc_id}: {rank}")
            print()