# BM25 Search Engine with Proximity Boosting

This project implements a search engine using the BM25 ranking algorithm, enhanced with proximity-based boosting to improve result relevance. It currently supports a body index (`b_index`) and is structured to accommodate future header (`h_index`) and title (`t_index`) indexes for weighted zone scoring. The engine is initialized once and allows interactive querying and parameter tuning via a driver script.

## Features

- **BM25 Ranking**: Scores documents using term frequency (TF), inverse document frequency (IDF), and document length normalization.
- **Proximity Boosting**: Adjusts scores based on the closeness of query terms in documents.
- **Interactive Interface**: Run queries one by one, check indexes, tune parameters, or paginate through results without reinitializing the engine.
- **Debug Mode**: Provides detailed output to diagnose issues like missing results or `None` outputs.
- **Pagination**: Retrieve the next set of results after the initial top results are displayed.
- **Extensibility**: Ready for weighted zone scoring with header and title indexes (currently commented out).

## Prerequisites

- **Python**: 3.x
- **Libraries**:
  - `pandas`: For loading index files.
  - `nltk`: For text processing (tokenization, stopwords, lemmatization).

Install via:

```bash
pip install pandas nltk
```

### NLTK Data
Download required NLTK resources (uncomment and run once in `searchEngine.py` if needed):

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

## File Structure

```
search-engine/
├── searchEngine.py    # Core engine class with BM25 and proximity logic
├── driver.py          # Interactive driver script for querying and tuning
├── Indexes
    ├── b_index.json       # Body index (JSON format)
    ├── id_doc_len.csv     # Document lengths (CSV: id,length)
    ├── id_link.csv        # Document IDs to URLs (CSV: id,link)
    └── doc_norms.csv      # Normalized document lengths (CSV: id,norm)
```

## Index File Formats

**b_index.json:**
```json
{"term": "example", "Postings": [{"id": 1, "positions": [10], "tf": 1}], "df": 50, "idf": 1.2}
```

**id_doc_len.csv:**
```
id,length
1,100
2,80
```

**id_link.csv:**
```
id,link
1,https://example.com
2,https://example2.com
```

**doc_norms.csv:**
```
id,norm
1,1.2
2,0.8
```

## Setup

- **Clone or Download**: Place all files in a single directory (e.g., `search-engine/`).
- **Prepare Indexes**: Ensure `b_index.json`, `id_doc_len.csv`, `id_link.csv`, and `doc_norms.csv` are present and correctly formatted.
- **Install Dependencies**: Run the pip command above if libraries are missing.

## Usage

### Running the Search Engine

**Basic Start:**
```bash
python driver.py
```
Initializes the engine and enters interactive mode:
```
Engine initialized successfully.
Interactive Search Engine (type 'exit' to quit, 'check' to inspect indexes, 'tune' to tune parameters, 'next' for next results)
Enter query:
```

**Debug Mode:**
```bash
python driver.py --debug
```
Adds debug output (processed terms, index hits, document IDs) for all queries and subsequent result sets.

**Custom Result Count:**
```bash
python driver.py --k 5
```
Returns top 5 results per query and 5 results per `next` command (default is 10).

### Interactive Commands

**Run a Query:**
```
Enter query: best restaurants in Rome
Running query: 'best restaurants in Rome'
Top 10 results:
1. https://rome-eats.com (Score: 15.200)
2. https://best-restaurants.com (Score: 14.800)
...
Enter query:
```

**Get Next Results:**
```
Enter query: next
Next 10 results:
11. https://italy-guide.com (Score: 13.200)
12. https://rome-food.com (Score: 13.100)
...
Enter query:
```

**Check Indexes:**
```
Enter query: check
Checking indexes...
b_index:
restaurant {'postings': [{'id': 1, 'positions': [10], 'tf': 1}, ...], 'df': 50, 'idf': 1.2}
...
id_link:
1 https://rome-eats.com
...
```

**Tune Parameters:**
```
Enter query: tune
Enter query to tune: best restaurants in Rome
Current parameters:
  k1: 1.5
  b: 0.75
  proximity_weight: 0.5
  ...
Select parameter to tune (or 'exit' to return):
  1. k1 - BM25 term frequency saturation
  3. proximity_weight - Weight of proximity boost
Enter choice: 3
Tuning proximity_weight (current value: 0.5)
Enter new value for proximity_weight: 0.0
Set proximity_weight to 0.0
Running query: 'best restaurants in Rome'
Top 10 results:
1. https://rome-eats.com (Score: 14.500)
...
```

**Exit:**
```
Enter query: exit
Exiting...
```

## Parameters

**BM25:**
- `k1` (1.5): Controls term frequency saturation.
- `b` (0.75): Controls length normalization.

**Proximity:**
- `proximity_weight` (0.5): Balances BM25 and proximity boost.
- `proximity_scale` (10.0): Adjusts distance sensitivity in proximity boost.

**Zone Weights:**
- `b_contri` (1/6): Body index contribution (currently active).
- `h_contri` (1/3): Header index contribution (commented out).
- `t_contri` (1/2): Title index contribution (commented out).

## Extending the Engine

**Add Header/Title Indexes:**
- Uncomment `h_index` and `t_index` loading in `__init__`.
- Provide `h_index.json` and `t_index.csv` (ensure CSV format matches JSON structure if used).
- Uncomment scoring logic in `_score_docs`.

