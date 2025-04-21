import os
import sys
from   googleapiclient.discovery import build

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configuration import *

api_key      = GOOGLE_SEARCH_API
cse_id       = GOOGLE_SEARCH_ENGINE_ID
search_limit = GOOGLE_SEARCH_LIMIT

def google_search(query, api_key, cse_id):
	num_results = search_limit
	service     = build("customsearch", "v1", developerKey=api_key)
	results     = []

	res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
	if 'items' in res:
	    for item in res['items']:
	        results.append({
	            'title': item['title'],
	            'link': item['link'],
	            'snippet': item.get('snippet', '')
	        })

	return results

def fetch_google_search(query):
	# query = "information retrieval"

	search_results = google_search(query, api_key, cse_id)

	# for idx, result in enumerate(search_results, start=1):
	#     print(f"{idx}. {result['title']}")
	#     print(result['link'])
	#     print(result['snippet'])
	#     print()

	return search_results