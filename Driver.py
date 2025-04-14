from flask import Flask, request, jsonify
from searchEngine import engine

app = Flask(__name__)
search_engine = engine()  # Initialize the search engine

@app.route('/search')
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    results = run_query(search_engine, query)
    return jsonify(results)

def run_query(search_engine, query, k=10):
    """Run a single query and return results."""
    search_engine.search(query)
    results = search_engine.retrieve_new_res(k=k)
    return results

if __name__ == "__main__":
    app.run(debug=True)