import os
from   searchEngine import engine
import json
import sys
import string
from   tqdm import tqdm
from   nltk.corpus import stopwords
from   nltk.stem import PorterStemmer
from   nltk.tokenize import word_tokenize
import nltk
import numpy as np
from   sklearn.feature_extraction.text import TfidfVectorizer

import pickle
from   collections import defaultdict
from   sklearn.feature_extraction.text import CountVectorizer
from   scipy.sparse import save_npz, load_npz
from   sklearn.cluster import KMeans
from   sklearn.metrics.pairwise import cosine_similarity
from   sklearn.decomposition import PCA
from   nltk.corpus import wordnet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from   configuration import *

nltk.download('punkt')
nltk.download('stopwords')

class Rocchio_:

	def __init__(self, model):

		# print("Loading Relvance Model..")
		# self.engine_handle = engine()
		# print("Model Successfully loaded.")

		self.base_dir  = os.path.dirname(os.path.abspath(__file__))      
		self.url_data_path   = os.path.join(self.base_dir, "WebData", URL_DATA_FILE)
		self.doc_vec_path    = os.path.join(self.base_dir, "WebData", DOC_VEC_FILE)
		self.doc_vec_pkl_path   = os.path.join(self.base_dir, "WebData", DOC_VEC_PKL_FILE)
		self.coomat_path        = os.path.join(self.base_dir, "WebData", COOMAT_FILE)

		self.engine_handle = model
		self.stop_words    = set(stopwords.words('english'))
		self.stemmer       = PorterStemmer()

		self.url_data_dict = self.get_json(self.url_data_path)
		# self.doc_vec_dict  = self.get_json(self.doc_vec_path)

		if self.doc_vec_pkl_path.split("\\")[-1] in os.listdir(os.path.join(self.base_dir, "WebData")):
			with open(self.doc_vec_pkl_path, 'rb') as f:
				self.vectorizer = pickle.load(f)
				print(f"{self.doc_vec_pkl_path} present.")
		else:
			print(f"{self.doc_vec_pkl_path} absent.")
			self.vectorizer    = TfidfVectorizer(
												    max_features = 10000,        # Only keep top 10,000 words
												    min_df       = 10,            # Word must appear in at least 5 documents
												    max_df       = 0.8           # Ignore words in more than 80% of documents
												)

	def create_document_vectors(self):

		urls      = list(self.url_data_dict.keys())
		docs_data = list(self.url_data_dict.values())

		print("Creating Document Vectors..")
		docs_vectors = self.vectorizer.fit_transform(docs_data).toarray()
		print("Done")
		new_docs_vector_dict = {}

		for i, url in tqdm(enumerate(urls)):
			new_docs_vector_dict[url] = docs_vectors[i].tolist()

		self.store_json(new_docs_vector_dict, 'WebData/document_vector.json')
		self.store_pickle(DOC_VEC_PKL_FILE)

	def transform_vectors_(self, data):

		print("Transforming Document Vector..")
		_vector = self.vectorizer.transform([data]).toarray()[0]
		print("Done")

		return _vector

	def clean_data(self, data):

		data         = data.lower()
		data         = data.translate(str.maketrans('','', string.punctuation))
		tokens       = word_tokenize(data)
		clean_tokens = [word for word in tokens if word not in self.stop_words]

		return " ".join(clean_tokens)

	def preprocess_raw_data(self,file_name):

		data = self.get_json(file_name)

		new_data = {}

		for row in tqdm(data):

			cleaned_data = self.clean_data(row['DATA'])
			new_data[row['Current URL']] = cleaned_data

		print("No. of Rows : ", len(new_data))

		self.store_json(new_data, 'WebData/url_data.json')

	def store_pickle(self, file_name):

		with open(file_name, 'wb') as f:
			pickle.dump(self.vectorizer, f)
			print(f"Successfully stored {file_name}")

	def store_json(self,sample_json, file_name):
		# file_name   = "sample_json_VR_0.json"
		file_handle = open(file_name,'w')
		json.dump(sample_json, file_handle, indent=5)
		file_handle.close()

		print("Sucessfully stored ",file_name)

	def get_json(self,filename):
		file_name   = filename
		file_handle = open(file_name,'r')
		token_data  = json.load(file_handle)
		file_handle.close()

		print("Sucessfully fetched ",file_name)

		return token_data

	def get_synonyms(self, word):
	    synonyms = set()
	    for syn in wordnet.synsets(word):
	        for lemma in syn.lemmas():
	            synonyms.add(lemma.name().replace('_', ' '))
	    print(synonyms)
	    return list(synonyms)

	def get_relevance_response(self, query, page_rank=False, k=20):

		# self.engine_handle.search(query, page_rank)

		# response = self.engine_handle.retrieve_new_res(k)

		# for row in response:
		# 	print("=====")
		# 	print(row[0])
		# 	print(self.url_data_dict[row[0]])
		# 	print("=====")

		# response = [('https://theculturetrip.com/europe/france/paris/articles', 7.145702096829678), ('https://www.thetrainline.com/en/destinations/trains-to-paris', 5.813547127339542), ('https://theculturetrip.com/europe/france/paris', 5.773837295028225), ('https://www.pinterest.es/ideas/paris-photo/893450598768', 5.76317810966183), ('https://www.timeshighereducation.com/news/uk-university-explores-options-south-koreas-forgotten-hub', 5.7323231892759035), ('https://op.europa.eu/en/more-search-options', 8.692267958742365), ('https://www.thoughtco.com/english-test-1212215', 7.73257173970749), ('https://www.thoughtco.com/paris-in-ancient-world-trojan-tribal-112870', 5.644280123300901), ('https://kids.britannica.com/students/article/Paris/276285', 5.628671977809937), ('https://u-paris.fr/en/culture-at-universite-de-paris', 5.5951620476115345)]
		response = [('https://www.britannica.com/video/Overview-Bonn-decision-Berlin-Germany/-197060', 6.916810113785645), ('https://www.worldatlas.com/articles/what-is-the-capital-of-russia.html', 6.940733446751313), ('https://www.worldatlas.com/articles/what-is-the-capital-of-france.html', 6.938024370567117), ('https://www.worldatlas.com/articles/what-is-the-capital-of-england.html', 6.934423284267179), ('https://www.worldatlas.com/articles/what-is-the-capital-of-spain.html', 6.930974379075249), ('https://www.worldatlas.com/articles/what-is-the-capital-of-greece.html', 6.913361743343211), ('https://visiteurope.com/en/experience/the-wonderful-capital-of-hungary', 6.291711512075911), ('https://www.britannica.com/quiz/european-capitals-quiz', 6.263608251703955), ('https://visiteurope.com/en/experience/welcome-to-slovakias-capital', 5.930034925465279), ('https://visiteurope.com/en/trip/exploring-the-capitals-of-northern-europe', 5.603632966759685)]

		expanded_terms = self.pseudo_relevance_(query, response)

		expanded_query = query + ' ' + ' '.join(expanded_terms)
		print("Expanded Query : ", expanded_query)

		return expanded_query

	def pseudo_relevance_(self, query, response):

		response_doc_vec_list = []

		query_vector = self.transform_vectors_(self.clean_data(query))

		for row in tqdm(response):
			print("ROw : ", row)
			cleaned_data = self.clean_data(self.url_data_dict[row['url']])
			doc_v        = self.transform_vectors_(cleaned_data)

			response_doc_vec_list.append(doc_v)

		relevant_docs     = response_doc_vec_list[:PSEUDO_RELEVANCE_K]
		non_relevant_docs = response_doc_vec_list[PSEUDO_RELEVANCE_K:]

		expanded_terms = self.compute_rocchio_expansion(query_vector, relevant_docs, non_relevant_docs)

		return expanded_terms

	def compute_rocchio_expansion(self, query_vector, relevant_docs, non_relevant_docs):

	    relevant_vecs     = np.mean(relevant_docs, axis=0)
	    non_relevant_vecs = np.mean(non_relevant_docs, axis=0)

	    expanded_query_vector = ALPHA_ * query_vector + BETA_ * relevant_vecs - GAMMA_ * non_relevant_vecs

	    feature_names = self.vectorizer.get_feature_names_out()
	    idf_values    = self.vectorizer.idf_

	    # Find terms appearing in at least 2 relevant docs
	    relevant_terms_counter = np.sum(np.array(relevant_docs) > 0, axis=0)
	    frequent_term_indices  = np.where(relevant_terms_counter >= PSEUDO_RELEVANCE_K-1)[0]

	    # Only keep frequent terms with positive weights
	    sorted_indices = expanded_query_vector.argsort()[::-1]
	    candidate_indices = [i for i in sorted_indices if i in frequent_term_indices and expanded_query_vector[i] > 0]

	    # Re-rank by IDF * query weight
	    scored_terms = [(i, expanded_query_vector[i] * idf_values[i]) for i in candidate_indices]

	    # Sort by this boosted score
	    scored_terms = sorted(scored_terms, key=lambda x: x[1], reverse=True)

	    # Pick top 20
	    top_indices = [i for i, score in scored_terms[:WORD_LIMIT]]

	    expanded_terms = [feature_names[i] for i in top_indices]

	    print(expanded_terms)

	    return expanded_terms

	def build_cooccurrence_matrix(self):
		url_data_dict = self.url_data_dict
		vocab         = self.vectorizer.get_feature_names_out()
		cooccurrence  = defaultdict(lambda: defaultdict(int))

		for doc_text in tqdm(url_data_dict.values()):
			tokens = set(doc_text.split())
			tokens = [t for t in tokens if t in set(vocab)]
			for token1 in tokens:
				for token2 in tokens:
					if token1 != token2:
						cooccurrence[token1][token2] += 1

		# Convert to matrix form
		vocab_list = list(vocab)
		vocab_idx  = {word: idx for idx, word in enumerate(vocab_list)}
		matrix     = np.zeros((len(vocab_list), len(vocab_list)))

		for word1 in cooccurrence:
			for word2 in cooccurrence[word1]:
				i, j = vocab_idx[word1], vocab_idx[word2]
				matrix[i][j] = cooccurrence[word1][word2]

		print("Saving cooccurence matrix..")
		np.save('WebData/cooccurrence_matrix.npy', matrix)
		print("Done")

		self.store_json(vocab_list, 'WebData/vocab_list.json')

		return matrix, vocab_list

	def fast_build_cooccurrence(self):
	    """
	    Builds and saves the co-occurrence matrix using sparse matrix math.
	    Much faster for large datasets.
	    """
	    documents, vocab = self.url_data_dict.values(), self.vectorizer.get_feature_names_out()
	    vectorizer = CountVectorizer(vocabulary=vocab, binary=True, max_df=0.7, min_df=5, stop_words='english')  # 1 if word exists, not count
	    print("Fit Transforming documents..")
	    X          = vectorizer.fit_transform(documents)  # document-term matrix (sparse)
	    print("Done")
	    
	    cooccurrence_matrix = (X.T @ X)  # term-term matrix
	    
	    # Remove self-cooccurrence (diagonal)
	    cooccurrence_matrix.setdiag(0)
	    
	    print("Saving cooccurence matrix..")
	    save_npz(self.coomat_path, cooccurrence_matrix)
	    print("Done")

	    return cooccurrence_matrix

	def preprocess_metric_QE(self, response):

		word_list = []

		for row in response:
			print(f"Fetching data for {row['url']}")
			data = self.url_data_dict[row['url']]
			for word in word_tokenize(data):
				if word not in word_list and len(word) > 3:
					word_list.append(word)

		print(f"Got {len(word_list)} words.")

		return word_list

	def get_association_QE(self, query):

		cleaned_query       = self.clean_data(query)
		cooccurrence_matrix = load_npz(self.coomat_path)
		vocab_list          = self.vectorizer.get_feature_names_out()
		vocab_index         = {word: idx for idx, word in enumerate(vocab_list)}

		tokens         = cleaned_query.lower().split()
		expanded_terms = []

		for token in tokens:
			if token in vocab_index:
				idx         = vocab_index[token]
				cooc_vector = cooccurrence_matrix.getrow(idx).toarray().flatten()

				top_indices = np.argsort(cooc_vector)[::-1]
				
				count = 0
				for i in top_indices:
					if cooc_vector[i] >= MIN_COOC :
						if vocab_list[i] not in bad_terms and vocab_list[i] not in expanded_terms:
							expanded_terms.append(vocab_list[i])
							count += 1
					if count >= 5:
						break
					# print(len(expanded_terms), count)


		expanded_query = " ".join(expanded_terms[:WORD_LIMIT])
		query          = word_tokenize(query)
		for word in expanded_terms:
			query.append(word)

		expanded_query = " ".join(query)
		print("Assoc. Query Terms : ", expanded_query)

		return expanded_query

	def get_metric_QE(self, query, response):

		# response = [('https://www.britannica.com/video/Overview-Bonn-decision-Berlin-Germany/-197060', 6.916810113785645), ('https://www.worldatlas.com/articles/what-is-the-capital-of-russia.html', 6.940733446751313), ('https://www.worldatlas.com/articles/what-is-the-capital-of-france.html', 6.938024370567117), ('https://www.worldatlas.com/articles/what-is-the-capital-of-england.html', 6.934423284267179), ('https://www.worldatlas.com/articles/what-is-the-capital-of-spain.html', 6.930974379075249), ('https://www.worldatlas.com/articles/what-is-the-capital-of-greece.html', 6.913361743343211), ('https://visiteurope.com/en/experience/the-wonderful-capital-of-hungary', 6.291711512075911), ('https://www.britannica.com/quiz/european-capitals-quiz', 6.263608251703955), ('https://visiteurope.com/en/experience/welcome-to-slovakias-capital', 5.930034925465279), ('https://visiteurope.com/en/trip/exploring-the-capitals-of-northern-europe', 5.603632966759685)]

		vocab_list = self.preprocess_metric_QE(response)

		term_vectors = self.vectorizer.transform(vocab_list).toarray()

		kmeans = KMeans(QE_CLUSTERS, random_state=42)
		labels = kmeans.fit_predict(term_vectors)

		word_to_cluster = {word:label for word, label in zip(vocab_list, labels)}

		query_tokens   = query.lower().split()
		expanded_terms = list(query_tokens)

		for token in query_tokens:

			if token in word_to_cluster:
				cluster_id    = word_to_cluster[token]

				cluster_terms = [word for word, label in word_to_cluster.items() if label == cluster_id and word not in expanded_terms]

				expanded_terms.extend(cluster_terms)

		cluster_terms = list(set(expanded_terms))#[:len(query_tokens) + WORD_LIMIT]
		# expanded_query = " ".join(expanded_terms)

		cluster_vectors = self.vectorizer.transform(cluster_terms).toarray()

		query_vector = self.vectorizer.transform([' '.join(query_tokens)]).toarray()

		similarities = cosine_similarity(query_vector, cluster_vectors)[0]

		top_indices = np.argsort(similarities)[::-1][:WORD_LIMIT]
		# print(query.lower().split())
		top_words = [cluster_terms[i] for i in top_indices if cluster_terms[i] not in query.lower().split()]

		# Step 8: Final expanded query
		final_terms = list(query_tokens) + top_words
		expanded_query = " ".join(final_terms)

		print("Expanded Query : ",expanded_query)

		return expanded_query

	def get_scalar_QE(self, query, response):

		# response = [('https://www.britannica.com/video/Overview-Bonn-decision-Berlin-Germany/-197060', 6.916810113785645), ('https://www.worldatlas.com/articles/what-is-the-capital-of-russia.html', 6.940733446751313), ('https://www.worldatlas.com/articles/what-is-the-capital-of-france.html', 6.938024370567117), ('https://www.worldatlas.com/articles/what-is-the-capital-of-england.html', 6.934423284267179), ('https://www.worldatlas.com/articles/what-is-the-capital-of-spain.html', 6.930974379075249), ('https://www.worldatlas.com/articles/what-is-the-capital-of-greece.html', 6.913361743343211), ('https://visiteurope.com/en/experience/the-wonderful-capital-of-hungary', 6.291711512075911), ('https://www.britannica.com/quiz/european-capitals-quiz', 6.263608251703955), ('https://visiteurope.com/en/experience/welcome-to-slovakias-capital', 5.930034925465279), ('https://visiteurope.com/en/trip/exploring-the-capitals-of-northern-europe', 5.603632966759685)]

		vocab_list   = self.preprocess_metric_QE(response)
		term_vectors = self.vectorizer.transform(vocab_list).toarray()

		pca = PCA(n_components=min(QE_CLUSTERS, term_vectors.shape[1]))
		reduced_vectors = pca.fit_transform(term_vectors)

		kmeans = KMeans(n_clusters=QE_CLUSTERS, random_state=42, n_init='auto')
		labels = kmeans.fit_predict(reduced_vectors)

		word_to_cluster = {word: label for word, label in zip(vocab_list, labels)}

		query_tokens   = word_tokenize(query)
		expanded_terms = list(query_tokens)

		for token in query_tokens:
			if token in word_to_cluster:
			    cluster_id = word_to_cluster[token]
			    cluster_terms = [word for word, label in word_to_cluster.items() if label == cluster_id and word not in expanded_terms]
			    expanded_terms.extend(cluster_terms[:WORD_LIMIT])

		# Limit final expansion
		expanded_terms = expanded_terms[:len(query_tokens) + WORD_LIMIT]

		expanded_query = " ".join(expanded_terms)

		print("Expanded Query:", expanded_query)
		
		return expanded_query


# if __name__ == "__main__":
# 	print("Hi")


# 	rocchio_handle = Rocchio_()

# 	# rocchio_handle.get_relevance_response("What is the capital of Germany?")

# 	# rocchio_handle.preprocess_raw_data("D:\\MS_ComputerScience\\UTD\\Spring2025\\Information Retreival_CS6322\\Project\\Code\\search_engine_ui\\src\\ClusterData\\europe_data 1.json")

# 	# rocchio_handle.create_document_vectors()

# 	# print(rocchio_handle.clean_data("Pari good travel option"))

# 	# rocchio_handle.fast_build_cooccurrence()

# 	# rocchio_handle.get_association_QE("What is the capital of Germany?")
# 	# rocchio_handle.get_metric_QE("What is the capital of Germany?")
# 	# rocchio_handle.get_scalar_QE("What is the capital of Germany?")

# 	rocchio_handle.get_synonyms('Capital')
