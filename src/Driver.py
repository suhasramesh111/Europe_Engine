import argparse
from searchEngine import engine

def run_query(search_engine, query, use_pagerank=False, use_hits=False, k=10, debug=False):
    """Run a single query and display results."""
    print(f"Running query: '{query}'")
    search_engine.search(query, use_pagerank=use_pagerank, use_hits=use_hits)
    results = search_engine.retrieve_new_res(k=k)
    
    if debug:
        terms = search_engine._process_query(query)
        print(f"Processed terms: {terms}")
        print("Terms in b_index:", [t for t in terms if t in search_engine.b_index])
        print("Terms in h_index:", [t for t in terms if t in search_engine.h_index])
        print("Terms in t_index:", [t for t in terms if t in search_engine.t_index])
        print(f"Scored documents: {len(search_engine.results)}")
        print("Sample doc_ids:", [doc_id for _, doc_id in list(search_engine.results.items())[:5]])
    
    if not results:
        print("No results found.")
        if debug:
            print("Check: Are query terms in indexes? Is id_link correctly mapped?")
    else:
        print(f"Top {k} results:")
        for i, (link, score) in enumerate(results, 1):
            print(f"{i}. {link} (Score: {score:.3f})")
    return results

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

def check_indexes(search_engine):
    """Run engine.check() to inspect indexes."""
    print("Checking indexes...")
    search_engine.check()

def tune_parameters(search_engine, query, k=10, use_pagerank=False, use_hits=False):
    """Interactively tune parameters for a given query."""
    param_options = {
        '1': ('k1', 'BM25 term frequency saturation'),
        '2': ('b', 'BM25 length normalization'),
        '3': ('proximity_weight_b', 'Body proximity boost weight'),
        '4': ('proximity_weight_h', 'Header proximity boost weight'),
        '5': ('proximity_weight_t', 'Title proximity boost weight'),
        '6': ('proximity_scale', 'Proximity distance sensitivity'),
        '7': ('b_contri', 'Body zone weight'),
        '8': ('h_contri', 'Header zone weight'),
        '9': ('t_contri', 'Title zone weight'),
        '10': ('bm25_weight', 'BM25 score weight'),
        '11': ('pagerank_weight', 'PageRank score weight'),
        '12': ('hits_weight', 'HITS authority score weight'),
    }
    
    while True:
        print(f"\nCurrent parameters:")
        print(f"  k1: {search_engine.k1}")
        print(f"  b: {search_engine.b}")
        print(f"  proximity_weight_b: {search_engine.proximity_weight_b}")
        print(f"  proximity_weight_h: {search_engine.proximity_weight_h}")
        print(f"  proximity_weight_t: {search_engine.proximity_weight_t}")
        print(f"  proximity_scale: {search_engine.proximity_scale}")
        print(f"  b_contri: {search_engine.b_contri}")
        print(f"  h_contri: {search_engine.h_contri}")
        print(f"  t_contri: {search_engine.t_contri}")
        print(f"  bm25_weight: {search_engine.bm25_weight}")
        print(f"  pagerank_weight: {search_engine.pagerank_weight}")
        print(f"  hits_weight: {search_engine.hits_weight}")
        print(f"  Using PageRank: {use_pagerank}")
        print(f"  Using HITS: {use_hits}")
        
        print("\nSelect parameter to tune (or 'exit' to return):")
        for key, (param, desc) in param_options.items():
            print(f"  {key}. {param} - {desc}")
        print("  13. Toggle PageRank usage")
        print("  14. Toggle HITS usage")
        
        choice = input("Enter choice: ").strip().lower()
        
        if choice == 'exit':
            print("Returning to main menu...")
            break
        elif choice == '13':
            use_pagerank = not use_pagerank
            print(f"PageRank usage {'enabled' if use_pagerank else 'disabled'}")
            run_query(search_engine, query, use_pagerank=use_pagerank, use_hits=use_hits, k=k, debug=True)
            continue
        elif choice == '14':
            use_hits = not use_hits
            print(f"HITS usage {'enabled' if use_hits else 'disabled'}")
            run_query(search_engine, query, use_pagerank=use_pagerank, use_hits=use_hits, k=k, debug=True)
            continue
        elif choice not in param_options:
            print("Invalid choice, try again.")
            continue
        
        param_name, param_desc = param_options[choice]
        current_value = getattr(search_engine, param_name)
        print(f"\nTuning {param_name} (current value: {current_value})")
        
        try:
            new_value = float(input(f"Enter new value for {param_name}: "))
            if new_value < 0 and param_name not in ['proximity_weight_b', 'proximity_weight_h', 'proximity_weight_t']:
                print("Value must be non-negative.")
                continue
            setattr(search_engine, param_name, new_value)
            print(f"Set {param_name} to {new_value}")
        except ValueError:
            print("Invalid number, try again.")
            continue
        
        run_query(search_engine, query, use_pagerank=use_pagerank, use_hits=use_hits, k=k, debug=True)

def interactive_mode(search_engine, k=10, debug=False, use_pagerank=False, use_hits=False):
    """Run queries one after another interactively with pagination."""
    print("Interactive Search Engine (type 'exit' to quit, 'check' to inspect indexes, 'tune' to tune parameters, 'next' for next results)")
    print("Type 'pagerank:on/off' or 'hits:on/off' to enable/disable PageRank/HITS for queries")
    
    last_query = None
    
    while True:
        query = input("\nEnter query: ").strip()
        
        if not query:
            print("Empty query, please try again.")
            continue
            
        if query.lower() == 'exit':
            print("Exiting...")
            break
        elif query.lower() == 'check':
            check_indexes(search_engine)
            continue
        elif query.lower().startswith('tune'):
            tune_query = last_query
            if query.lower() != 'tune' and len(query) > 5:
                tune_query = query[5:].strip()
            
            if tune_query:
                tune_parameters(search_engine, tune_query, k=k, use_pagerank=use_pagerank, use_hits=use_hits)
            else:
                print("No query to tune. Run a query first or specify one with 'tune [query]'.")
            continue
        elif query.lower() == 'next':
            if not last_query:
                print("No previous query to get next results for. Run a query first.")
            elif not search_engine.results:
                print("No more results available for the last query.")
            else:
                get_next_results(search_engine, k=k)
            continue
        elif query.lower() == 'pagerank:on':
            use_pagerank = True
            print("PageRank enabled for future queries.")
            continue
        elif query.lower() == 'pagerank:off':
            use_pagerank = False
            print("PageRank disabled for future queries.")
            continue
        elif query.lower() == 'hits:on':
            use_hits = True
            print("HITS enabled for future queries.")
            continue
        elif query.lower() == 'hits:off':
            use_hits = False
            print("HITS disabled for future queries.")
            continue
        
        run_query(search_engine, query, use_pagerank=use_pagerank, use_hits=use_hits, k=k, debug=debug)
        last_query = query

def main():
    parser = argparse.ArgumentParser(description="Search Engine Driver (Single Engine Instance)")
    parser.add_argument('--k', type=int, default=10,
                        help="Number of results to return (default: 10)")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug output (terms, index hits, doc_ids)")
    parser.add_argument('--pagerank', action='store_true',
                        help="Enable PageRank for scoring by default")
    parser.add_argument('--hits', action='store_true',
                        help="Enable HITS for scoring by default")
    args = parser.parse_args()
    
    try:
        search_engine = engine()
        print("Engine initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return
    
    interactive_mode(search_engine, k=args.k, debug=args.debug, use_pagerank=args.pagerank, use_hits=args.hits)

if __name__ == "__main__":
    main()