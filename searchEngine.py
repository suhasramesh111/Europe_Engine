import pandas as pd
import heapq
import string
import json

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from itertools import combinations


class engine:
    """
    Search engine class that supports BM25 ranking with proximity-based boosting.
    """
    
    def __init__(self):
        """Initialize the search engine by loading indices and configuration parameters."""
        
        self.b_index = self._load_index("Indexes/b_index.json")
        # self.h_index = self._load_index("Indexes/h_index.json")
        # self.t_index = self._load_index("Indexes/t_index.csv")
        
        self.id_doc_len = self._load_index("Indexes/id_doc_len.csv")
        self.id_link = self._load_index("Indexes/id_link.csv")
        self.doc_norms = self._load_index("Indexes/doc_norms.csv")
        
        
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.wnl = WordNetLemmatizer()
        
        self.b_contri = 1/6 # contribution weight for body index
        # self.h_contri = 1/3 # contribution weight for headings index
        # self.t_contri = 1/2 # contribution weight for title index
        
        self.k1 = 1.5
        self.b = 0.75
        
        self.results = []
        self.max_links = 1000
        
        self.proximity_weight = 0.5  
        self.proximity_scale = 10.0
        
    
    def _load_index(self, index):
        """Helper to load different index files (CSV or JSON)."""
        
        _, ext = index.split('.')
        
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
            index_df = pd.read_csv(index)
            index_dict = dict(zip(index_df.iloc[:, 0], index_df.iloc[:, 1]))
            return index_dict
        
        
    def search(self, query):
        """Search for the given query and rank documents."""
        
        self.results = []
        terms = self._process_query(query)
        if not terms:
            return
        self._rank_docs(terms)
        
    
    def retrieve_new_res(self, k = 10):
        """Retrieve top k ranked results with their scores."""
        
        final_docs = []
        
        for i in range(k):
            score, doc_id = heapq.heappop(self.results)
            link = self.id_link.get(doc_id)
            final_docs.append((link, -score))
        
        return final_docs
    
    
    def _process_query(self,q):
        """Tokenize, clean, and lemmatize the input query."""
        
        tokens = word_tokenize(q)
        return [self.wnl.lemmatize(token.lower()) for token in tokens if token.lower() not in self.stop_words and token not in self.punctuation]
    
    
    def _rank_docs(self, terms):
        """Rank documents using BM25 and proximity boosting."""
        
        all_docs = self._score_docs(terms)
        
        for doc_id, score in all_docs.items():
            if len(self.results) < self.max_links:
                heapq.heappush(self.results, (-score, doc_id))
            else:
                heapq.heappushpop(self.results, (-score, doc_id))
    
    
    def _score_docs(self, terms):
        """Compute document scores based on BM25 and proximity of terms."""
        
        res = {}
        term_positions = {}
        
        for term in terms:
            
            if term in self.b_index:
                for doc in self.b_index[term]['postings']:
                    
                    doc_id = doc['id']
                    idf_ = self.b_index[term]['idf']
                    doc_norm = self.doc_norms[doc['id']]
                    tf_ = doc['tf']
                    positions = doc['positions']
                    
                    bm = self._calc_bm(idf_, doc_norm, tf_)
                    res[doc_id] = res.get(doc_id, 0) + self.b_contri * bm
                    
                    if doc_id not in term_positions:
                        term_positions[doc_id] = {}
                    term_positions[doc_id][term] = positions
            
            # if term in self.h_index:
            #     for doc in self.h_index[term]['postings']:
                    
            #         doc_id = doc['id']
            #         idf_ = self.h_index[term]['idf']
            #         doc_norm = self.doc_norms[doc['id']]
            #         tf_ = doc['tf']
            #         positions = doc['positions']
                    
            #         bm = self._calc_bm(idf_, doc_norm, tf_)
            #         res[doc_id] = res.get(doc_id, 0) + self.h_contri * bm
                    
            #         if doc_id not in term_positions:
            #             term_positions[doc_id] = {}
            #         term_positions[doc_id][term] = positions
                    
            # if term in self.t_index:
            #     for doc in self.t_index[term]['postings']:
                    
            #         doc_id = doc['id']
            #         idf_ = self.t_index[term]['idf']
            #         doc_norm = self.doc_norms[doc['id']]
            #         tf_ = doc['tf']
            #         positions = doc['positions']
                    
            #         bm = self._calc_bm(idf_, doc_norm, tf_)
            #         res[doc_id] = res.get(doc_id, 0) + self.t_contri * bm
                    
            #         if doc_id not in term_positions:
            #             term_positions[doc_id] = {}
            #         term_positions[doc_id][term] = positions
        
        #Adding proximity boost
        for doc_id in res:
            
            bm25_score = res[doc_id]
            proximity_boost = self._calc_proximity_boost(terms, term_positions.get(doc_id, {}))
            
            final_score = bm25_score * (1 + self.proximity_weight * (proximity_boost - 1))
            res[doc_id] = final_score
            # print(f"Doc {doc_id}: BM25={bm25_score:.3f}, Proximity={proximity_boost:.3f}, Final={final_score:.3f}")
            
        return res
    
    
    def _calc_proximity_boost(self, terms, doc_term_positions):
        """Compute a proximity boost factor based on distances between query terms."""
        
        if len(terms) < 2 or not doc_term_positions:
            return 1.0
        
        total_boost = 0.0
        pair_count = 0
        for term1, term2 in combinations(terms, 2):
            if term1 in doc_term_positions and term2 in doc_term_positions:
                pos1 = doc_term_positions[term1]
                pos2 = doc_term_positions[term2]
                min_dist = float('inf')
                for p1 in pos1:
                    for p2 in pos2:
                        dist = abs(p1 - p2)
                        min_dist = min(min_dist, dist)
                if min_dist < float('inf'):
                    
                    # Use inverse distance boost
                    boost = 1 / (1 + min_dist / self.proximity_scale)
                    total_boost += boost
                    pair_count += 1
        
        return (total_boost / pair_count) if pair_count > 0 else 1.0
    
    
    def _calc_bm(self, idf, doc_norm, tf):
        """Compute BM25 score component."""
        
        return idf * ((self.k1 + 1) * tf)/(self.k1*((1-self.b)+self.b*doc_norm)+tf)
        
        
    def check(self):
        """Print samples of loaded indices for verification."""
        
        i = 0
        print("b_index:")
        for key, value in self.b_index.items():
            if i < 5:
                print(key, value)
                i += 1
            else:
                break
        print()
        
        i = 0
        print("id_doc_len:")
        for key, value in self.id_doc_len.items():
            if i < 5:
                print(key, value)
                i += 1
            else:
                break
        print()
        
        i = 0
        print("id_link:")
        for key, value in self.id_link.items():
            if i < 5:
                print(key, value)
                i += 1
            else:
                break
        print()
        
        i = 0
        print("doc_norms:")
        for key, value in self.doc_norms.items():
            if i < 5:
                print(key, value)
                i += 1
            else:
                break
        print()
        