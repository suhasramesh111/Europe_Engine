from   flask import Flask, render_template, request, redirect, url_for, make_response
import random
import sys
import os
from   datetime import datetime
from   configuration import *


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from google_api          import fetch_google_search
from relevance_model_api import retreive_response
from searchEngine        import *

print("Starting Application..")
app           = Flask(__name__)

print("Loading Model..")
engine_handle = engine()
print("Model Successfully loaded.")

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
        query_list.append(query.replace("\n",""))
        write_file(PAST_QUERY_FILE, query_list)

        return redirect(url_for('results', query=query, query_data = query_list))

    return render_template('index.html', year=datetime.now().year, history=query_list)

@app.route('/results')
def results():
    query = request.args.get('query', '')
    print("Results Query : ", query)
    results = {
                'vsm'            : [],
                'pagerank'       : [],
                'hits'           : [],
                'clustering'     : [],
                'query_expansion': [],
                'google'         : [],
                'bing'           : [],
                'expanded_query' : ''
            }
   
    results['relevance']       = retreive_response(engine_handle, query, False, MODEL_RESPONSE_LIMIT)
    results['pagerank']        = retreive_response(engine_handle, query, True, MODEL_RESPONSE_LIMIT)
    results['hits']            = simulate_results("HITS", query)
    results['clustering']      = simulate_results("Clustering", query)
    results['query_expansion'] = simulate_results("Query Expansion", query)
    results['google']          = fetch_google_search(query)
    results['bing']            = simulate_results("Bing", query)
    results['expanded_query']  = query + " + extra keywords"
    
    file_data  = read_file(PAST_QUERY_FILE)    
    query_list = [q.replace("\n","") for q in file_data][-5:]
    print(query_list)
    
    rendered = render_template('results.html', query=query, results=results, history=query_list)
    response = make_response(rendered)
    response.headers['Cache-Control'] = 'no-store'

    return response

def simulate_results(source, query):
    return [
                {'title': f'{source} Result {i}', 'url': f'https://example.com/{source.lower()}/{i}'}
                for i in range(1, 6)
            ]


if __name__ == '__main__':
    app.run(debug=True)
