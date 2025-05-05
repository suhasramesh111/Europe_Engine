from   flask import Flask, render_template, request, redirect, url_for, make_response
import random
import sys
import os
from   datetime import datetime
from   configuration import *
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from google_api          import fetch_google_search
from relevance_model_api import *
from searchEngine        import *
from rocchio             import Rocchio_
from query_experiments   import *

start_time = time.time()
print("Starting Application..")
app           = Flask(__name__)

print("Loading Model..")
engine_handle = engine()
print("Model Successfully loaded.")

print("Initializing Rocchio..")
rocchio_handle = Rocchio_(engine_handle)
print("Initialized.")

print("Initializing Clustering..")
cluster_handle = QuickSearchEngine.get_instance()
print("Initialized.")

end_time = time.time()
print("Application Start Time : ", end_time - start_time, 's')

def read_file(file_name):

    try:
        file_handle = open(file_name, 'r')
        data        = file_handle.readlines()
        file_handle.close()

        print("Read Successful.")

        return data

    except Exception as e:
        print(e)

        return []

def write_file(file_name, data):

    file_handle = open(file_name, 'w')
    for row in data:
        if row != "\n":
            file_handle.write(row)
            file_handle.write("\n")
    file_handle.close()

    print("Write Successful.")

@app.route('/', methods=['GET', 'POST'])
def index():
    query = ''
    
    file_data  = read_file(PAST_QUERY_FILE)    
    query_list = [q.replace("\n","") for q in file_data][-5:]

    if request.method == 'POST':
        query = request.form['query']
        method = request.form['method']
        query_list.append(query.replace("\n", ""))
        write_file(PAST_QUERY_FILE, query_list)
        return redirect(url_for('results', query=query, method=method))

    return render_template('index.html', year=datetime.now().year, history=query_list)

@app.route('/results')
def results():
    query = request.args.get('query', '')
    method = request.args.get('method', 'relevance')
    print(f"Query: {query} | Method: {method}")

    results = {
        'relevance': [],
        'pagerank': [],
        'hits': [],
        'kmeans': [],
        'flat': [],
        'agglo_single': [],
        'agglo_complete': [],
        'rocchio': [],
        'association': [],
        'metric': [],
        'scalar': [],
        'google': [],
        'bing': []  # this is handled by iframe
    }
    algo_choice    = ""
    expanded_query = ""

    try:
        relevance_results = retreive_response(engine_handle, query, False, False, MODEL_RESPONSE_LIMIT)

        if method == 'relevance':
            results = relevance_results
            algo_choice = 'relevance'
        
        elif method == 'pagerank':
            results = retreive_response(engine_handle, query, True, False, MODEL_RESPONSE_LIMIT)
            algo_choice = 'pagerank'
        
        elif method == 'hits':
            results = retreive_response(engine_handle, query, False, True, MODEL_RESPONSE_LIMIT)
            algo_choice = 'hits'
        
        elif method == 'kmeans':
            results = fetch_clustering_response(query, MODEL_RESPONSE_LIMIT, cluster_handle, 'kmeans')
            algo_choice = 'kmeans'

        elif method == 'flat':
            results = fetch_clustering_response(query, MODEL_RESPONSE_LIMIT, cluster_handle, 'flat')
            algo_choice = 'flat'

        elif method == 'agglo_single':
            results = fetch_clustering_response(query, MODEL_RESPONSE_LIMIT, cluster_handle, 'agglo_single')
            algo_choice = 'agglo_single'

        elif method == 'agglo_complete':
            results = fetch_clustering_response(query, MODEL_RESPONSE_LIMIT, cluster_handle, 'agglo_complete')
            algo_choice = 'agglo_complete'
        
        elif method == 'rocchio':
            results, expanded_query = fetch_rocchio_response(query, relevance_results, rocchio_handle, engine_handle, MODEL_RESPONSE_LIMIT)
            algo_choice = 'rocchio'
        
        elif method == 'association':
            results, expanded_query = fetch_associative_response(query, relevance_results, rocchio_handle, engine_handle, MODEL_RESPONSE_LIMIT)
            algo_choice = 'association'
        
        elif method == 'metric':
            results, expanded_query = fetch_metric_response(query, relevance_results, rocchio_handle, engine_handle, MODEL_RESPONSE_LIMIT)
            algo_choice = 'metric'
       
        elif method == 'scalar':
            results, expanded_query = fetch_scalar_response(query, relevance_results, rocchio_handle, engine_handle, MODEL_RESPONSE_LIMIT)
            algo_choice = 'scalar'

        results_google = fetch_google_search(query)

        query_list = [q.replace("\n", "") for q in read_file(PAST_QUERY_FILE)][-5:]
        rendered = render_template('results.html', query=query, results=results, history=query_list, algo=algo_choice, expanded_query=expanded_query, results_google=results_google)
        response = make_response(rendered)
        response.headers['Cache-Control'] = 'no-store'
       
        return response

    except Exception as e:
        print("Error:", e)
        return make_response("Ahh, Something went wrong :)", 500)


def simulate_results(source, query):
    return [
                {'title': f'{source} Result {i}', 'url': f'https://example.com/{source.lower()}/{i}'}
                for i in range(1, 6)
            ]


# @app.route("/rocchio_results")
# def rocchio_results():
#     query = request.args.get('query', '')
#     relevance_results = request.args.get('results', '')

#     print("======")
#     print(query)
#     print(response)
#     print('======')
#     fetch_rocchio_response(query, response, rocchio_handle)


if __name__ == '__main__':
    app.run(debug=False)
