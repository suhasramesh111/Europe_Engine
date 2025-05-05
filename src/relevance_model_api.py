import os
from   searchEngine import *
import time
import sys
from   rocchio import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# print("Loading Modal..")
# engine_handle = engine()
# print("Modal loaded Successfully.")
import csv
from   configuration import *
from   query_cluster import *

# _engine_handle = None

# def get_engine():
#     global _engine_handle
#     if _engine_handle is None:
#         print("Loading Model (singleton)...")
#         _engine_handle = engine()
#         print("Model loaded successfully.")
#     else:
#         print("Using existing engine instance.")
#     return _engine_handle

def retreive_response(engine_handle, query, page_rank_enable, use_hits_enable, k):
	
		
	print("[PAGE RANK STATUS] : ", page_rank_enable)
	print("[HTIS STATUS]      : ", use_hits_enable)
	# engine_handle = get_engine()

	print("[QUERY] : ", query)
	start_time = time.time()
	
	engine_handle.search(query, page_rank_enable, use_hits_enable)
	
	end_time = time.time()
	print("[Search Time] : ", end_time - start_time)
	
	response = engine_handle.retrieve_new_res(k)
	# print(response)

	
	response = preprocess_response(response)

	# response = engine_handle.fetch_response(query, page_rank_enable, k)

	# response = preprocess_response(response)

	return response

def preprocess_response(response):

	results = []

	for index, row in enumerate(response):
		results.append({'url':row[0], "title": row[0]})
		print({'url':row[0], "title": row[0]})

	return results

def fetch_rocchio_response(query, response, rocchio_handle, engine_handle, k):

	print("[ROCCHIO API]")
	print("Received Query Rocchio : ", query)
	print("Received Responses : ", response)

	expanded_terms = rocchio_handle.pseudo_relevance_(query, response)
	expanded_query = query + ' ' + ' '.join(expanded_terms)
	print("Expanded Query : ", expanded_query)

	print("Fetching responses for Expanded Query..")
	response = retreive_response(engine_handle, expanded_query, False, False, k)
	

	# return {'expanded_query' : expanded_query, 'response' : response}
	return response, expanded_query

def fetch_associative_response(query, response, rocchio_handle, engine_handle, k):

	print("[ASSOCIATIVE API]")
	print("Received Query Rocchio : ", query)
	print("Received Responses : ", response)

	expanded_terms = rocchio_handle.get_association_QE(query)
	# expanded_query = query + ' ' + ' '.join(expanded_terms)
	print("Expanded Query : ", expanded_terms)

	print("Fetching responses for Expanded Query..")
	response = retreive_response(engine_handle, expanded_terms, False, False, k)
	

	# return {'expanded_query' : expanded_terms, 'response' : response}
	return response, expanded_terms


def fetch_metric_response(query, response, rocchio_handle, engine_handle, k):

	print("[METRIC API]")
	print("Received Query Rocchio : ", query)
	print("Received Responses : ", response)

	expanded_terms = rocchio_handle.get_metric_QE(query, response)
	# expanded_query = query + ' ' + ' '.join(expanded_terms)
	print("Expanded Query : ", expanded_terms)

	print("Fetching responses for Expanded Query..")
	response = retreive_response(engine_handle, expanded_terms, False, False, k)
	

	# return {'expanded_query' : expanded_terms, 'response' : response}
	return response, expanded_terms

def fetch_scalar_response(query, response, rocchio_handle, engine_handle, k):

	print("[SCALAR API]")
	print("Received Query Rocchio : ", query)
	print("Received Responses : ", response)

	expanded_terms = rocchio_handle.get_scalar_QE(query, response)
	# expanded_query = query + ' ' + ' '.join(expanded_terms)
	print("Expanded Query : ", expanded_terms)

	print("Fetching responses for Expanded Query..")
	response = retreive_response(engine_handle, expanded_terms, False, False, k)
	

	# return {'expanded_query' : expanded_terms, 'response' : response}
	return response, expanded_terms

def fetch_clustering_response_old(query, k, cluster_handle, cluster_type):

	print("[CLUSTERING QUERY] : ", query)
	print("[CLSUTER TYPE] : ", cluster_type)

	data_dict = {}
	response  = []

	print("Reading id_link document..")
	base_dir  = os.path.dirname(os.path.abspath(__file__))
	with open(os.path.join(base_dir, "Indexes_Clustering", ID_LINK_DOC__CLUSTERING), mode='r', newline='', encoding='utf-8') as file:
		reader = csv.reader(file)
		for index, row in enumerate(reader):
			# print(index)
			if index > 0:
				if len(row) >= 2:  # Ensure there are at least 2 columns
					key   = int(row[0])
					value = row[1]
					data_dict[key] = value
	print("Read Success.")
	# print(data_dict)
	# print(data_dict[28766])
	print("Starting Clustering..")
	# results = run_experiments(query, top_n=10, cluster_type=cluster_type)
	results = cluster_handle.run(query, cluster_type='flat')
	for row in results:
		for doc_id in row['clustered_doc_ids']:
			print(doc_id, " : ", data_dict[doc_id])
			response.append(data_dict[doc_id])
			response.append({'url':data_dict[doc_id], "title": data_dict[doc_id]})
	
	return response	

def fetch_clustering_response(query, k, cluster_handle, cluster_type):

	print("[CLUSTERING QUERY] : ", query)
	print("[CLSUTER TYPE] : ", cluster_type)

	data_dict = {}
	response  = []

	if cluster_type == 'kmeans':
		print("[K-means CLUSTERING]")
		results = cluster_handle.kmeans_search(query, k)
		print(results['query'])
		print(results['processed_query'])
		results = results['results']

	elif cluster_type == 'flat':
		print("[FLAT CLUSTERING]")
		results = cluster_handle.flat_cluster_search(query, k)
		print(results['query'])
		print(results['processed_query'])
		results = results['results']

	elif cluster_type == 'agglo_single':
		print("[AGGLOMERATIVE SINGLE]")
		results = cluster_handle.agglom_single_search(query, k)
		print(results['query'])
		print(results['processed_query'])
		results = results['results']

	elif cluster_type == 'agglo_complete':
		print("[AGGLOMERATIVE COMPLETE]")
		results = cluster_handle.agglom_complete_search(query, k)
		print(results['query'])
		print(results['processed_query'])
		results = results['results']

	else:
		results = []


	response = []

	for row in results:
		print(row)
		response.append({'url' : row['url'], 'title' : row['url']})

	return response

# fetch_clustering_response(['christmas market'], WORD_LIMIT, 'agglom_on_kmeans')

# def load_relevance_model():

# 	print("Loading Model..")
# 	engine_handle = engine()
# 	print("Model Load Success.") 

# 	return engine_handle

# if __name__ == '__main__':
# 	print("Hi")
	
# # 	engine_handle = load_relevance_model()

# 	# data = retreive_response("Restaurants in Germany", False, MODEL_RESPONSE_LIMIT)
# 	# print(data)
# 	# print(len(data))
# 	print("=====")
# 	data = retreive_response("Restaurants in Germany", True, MODEL_RESPONSE_LIMIT)
# 	print(data)
# 	print(len(data))

	
