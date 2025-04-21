from   flask import Flask, render_template, request
import random
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from google_api import fetch_google_search

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    query = ''
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

    if request.method == 'POST':
        query = request.form['query']

        results['vsm']             = simulate_results("VSM", query)
        results['pagerank']        = simulate_results("PageRank", query)
        results['hits']            = simulate_results("HITS", query)
        results['clustering']      = simulate_results("Clustering", query)
        results['query_expansion'] = simulate_results("Query Expansion", query)
        results['google']          = fetch_google_search(query)
        results['bing']            = simulate_results("Bing", query)
        results['expanded_query']  = query + " + extra keywords"

    return render_template('index.html', query=query, results=results)

def simulate_results(source, query):
    return [
        {'title': f'{source} Result {i}', 'url': f'https://example.com/{source.lower()}/{i}'}
        for i in range(1, 6)
    ]

if __name__ == '__main__':
    app.run(debug=True)
