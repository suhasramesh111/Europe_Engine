import argparse
from searchEngine import engine
from flask import Flask, request, jsonify


def run_query(search_engine, query, k=10, debug=False):
    """Run a single query and display results."""
    print(f"Running query: '{query}'")
    search_engine.search(query)
    results = search_engine.retrieve_new_res(k=k)
    
    if debug:
        terms = search_engine._process_query(query)
        print(f"Processed terms: {terms}")
        print("Terms in b_index:", [t for t in terms if t in search_engine.b_index])
        print(f"Scored documents: {len(search_engine.results)}")
        print("Sample doc_ids:", [doc_id for _, doc_id in search_engine.results[:5]])
    
    if not results:
        print("No results found.")
        if debug:
            print("Check: Are query terms in b_index? Is id_link correctly mapped?")
    else:
        print(f"Top {k} results:")
        for i, (link, score) in enumerate(results, 1):
            print(f"{i}. {link} (Score: {score:.3f})")

def check_indexes(search_engine):
    """Run engine.check() to inspect indexes."""
    print("Checking indexes...")
    search_engine.check()

def get_next_results(search_engine, k=10):
    """Retrieve the next k results from the current results heap."""
    results = search_engine.retrieve_new_res(k=k)
    if not results:
        print("No more results available.")
    else:
        print(f"Next {k} results:")
        for i, (link, score) in enumerate(results, 1):
            print(f"{i}. {link} (Score: {score:.3f})")
    return results

def tune_parameters(search_engine, query, k=10):
    """Interactively tune parameters for a given query."""
    param_options = {
        '1': ('k1', 'BM25 term frequency saturation'),
        '2': ('b', 'BM25 length normalization'),
        '3': ('proximity_weight', 'Weight of proximity boost'),
        '4': ('proximity_scale', 'Proximity distance sensitivity'),
        '5': ('b_contri', 'Body zone weight'),
        '6': ('h_contri', 'Header zone weight'),
        '7': ('t_contri', 'Title zone weight'),
    }
    
    while True:
        # Display current parameters
        print(f"\nCurrent parameters:")
        print(f"  k1: {search_engine.k1}")
        print(f"  b: {search_engine.b}")
        print(f"  proximity_weight: {search_engine.proximity_weight}")
        print(f"  proximity_scale: {search_engine.proximity_scale}")
        print(f"  b_contri: {search_engine.b_contri}")
        print(f"  h_contri: {search_engine.h_contri}")
        print(f"  t_contri: {search_engine.t_contri}")
        
        # Show tuning options
        print("\nSelect parameter to tune (or 'exit' to return):")
        for key, (param, desc) in param_options.items():
            print(f"  {key}. {param} - {desc}")
        
        choice = input("Enter choice: ").strip().lower()
        
        if choice == 'exit':
            print("Returning to main menu...")
            break
        elif choice not in param_options:
            print("Invalid choice, try again.")
            continue
        
        param_name, param_desc = param_options[choice]
        current_value = getattr(search_engine, param_name)
        print(f"\nTuning {param_name} (current value: {current_value})")
        
        # Get new value
        try:
            new_value = float(input(f"Enter new value for {param_name}: "))
            if new_value < 0 and param_name not in ['proximity_weight']:  # Allow negative proximity_weight if needed
                print("Value must be non-negative.")
                continue
            setattr(search_engine, param_name, new_value)
            print(f"Set {param_name} to {new_value}")
        except ValueError:
            print("Invalid number, try again.")
            continue
        
        # Run query with updated parameter
        run_query(search_engine, query, k=k, debug=True)
'''
def interactive_mode(search_engine, k=10, debug=False):
    """Run queries one after another interactively with pagination."""
    
    print("Interactive Search Engine (type 'exit' to quit, 'check' to inspect indexes, 'tune' to tune parameters, 'next' for more results)")
    last_query = None
    
    while True:
        query = input("\nEnter query: ").strip().lower()
        if query == 'exit':
            print("Exiting...")
            break
        elif query == 'check':
            check_indexes(search_engine)
        elif query == 'tune':
            tune_query = input("Enter query to tune: ").strip()
            if tune_query:
                tune_parameters(search_engine, tune_query, k=k)
                last_query = tune_query  # Update last query after tuning
        elif query == 'next':
            if not last_query:
                print("No previous query to get next results for. Run a query first.")
            elif not search_engine.results:
                print("No more results available for the last query.")
            else:
                get_next_results(search_engine, k=k)
        elif not query:
            print("Empty query, please try again.")
        else:
            run_query(search_engine, query, k=k, debug=debug)
            last_query = query  # Store the last query for 'next'
'''
if __name__ == '__main__':
    # Initialize engine once
    search_engine = engine()
    # Run interactive mode
    interactive_mode(search_engine)
'''
app = Flask(__name__)

# Initialize engine (outside the request context)
search_engine = engine()

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Missing query parameter'}), 400
    
    # You can adjust 'k' as needed or pass it as a parameter
    search_engine.search(query)
    results = search_engine.retrieve_new_res(k=10)  

    # Format the results as a list of dictionaries for better JSON serialization
    formatted_results = [{'link': link, 'score': score} for link, score in results]
    
    return jsonify(formatted_results)

if __name__ == '__main__':
    app.run(debug=True)