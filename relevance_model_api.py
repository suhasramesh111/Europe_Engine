import os
from   searchEngine import *
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# print("Loading Modal..")
# engine_handle = engine()
# print("Modal loaded Successfully.")

from configuration import *


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

def retreive_response(engine_handle, query, page_rank_enable, k):
	
		
	print("[PAGE RANK STATUS] : ", page_rank_enable)
	# engine_handle = get_engine()

	print("[QUERY] : ", query)
	start_time = time.time()
	
	engine_handle.search(query, page_rank_enable)
	
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

	
