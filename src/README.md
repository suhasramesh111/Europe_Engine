# BM25 Search Engine with Proximity Boosting and PageRank

This project implements a search engine using the BM25 ranking algorithm, enhanced with proximity-based boosting and optional PageRank integration. It supports body (`b_index`), header (`h_index`), and title (`t_index`) indexes for weighted zone scoring. The engine is initialized once and allows interactive querying, parameter tuning, and pagination via a driver script.

## Features

- **BM25 Ranking**: Scores documents using term frequency (TF), inverse document frequency (IDF), and document length normalization.
- **Proximity Boosting**: Boosts scores based on the closeness of query terms in documents.
- **PageRank Integration**: Optionally combines BM25 scores with PageRank scores using a weighted average.
- **Interactive Interface**: Run queries, toggle PageRank, check indexes, tune parameters, or paginate results without reinitializing the engine.
- **Debug Mode**: Provides detailed output (processed terms, index hits, document IDs) for troubleshooting.
- **Pagination**: Retrieve subsequent result sets using the `next` command.
- **Zone Scoring**: Weights contributions from body, header, and title indexes differently.

## Prerequisites

- **Python**: 3.x
- **Libraries**:
  - `pandas`: For loading index files.
  - `nltk`: For text processing (tokenization, stopwords, lemmatization).
  - `sortedcontainers`: For efficient result ranking.
  - `numpy`: For vectorized score calculations.

Install via:

```bash
pip install pandas nltk sortedcontainers numpy
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
├── searchEngine.py    # Core engine class with BM25, proximity, and PageRank logic
├── driver.py          # Interactive driver script for querying and tuning
├── Indexes/
│   ├── b_index.json       # Body index (JSON)
│   ├── h_index.json       # Header index (JSON)
│   ├── t_index.json       # Title index (JSON)
│   ├── id_link.csv        # Document IDs to URLs (CSV: id,link)
│   ├── doc_norms.csv      # Normalized document lengths (CSV: id,doc_norm_b,doc_norm_t,doc_norm_h)
│   └── pagerank.csv       # PageRank scores (CSV: id,score)
```

## Index File Formats

**b_index.json, h_index.json, t_index.json**:

```json
{"term": "example", "Postings": [{"id": 1, "positions": [10], "tf": 1}], "df": 50, "idf": 1.2}
```

**id_link.csv**:

```
id,link
1,https://example.com
2,https://example2.com
```

**doc_norms.csv**:

```
id,doc_norm_b,doc_norm_t,doc_norm_h
1,1.2,0.8,0.9
2,0.8,0.7,0.85
```

**pagerank.csv**:

```
id,score
1,0.05
2,0.03
```

## Setup

- **Clone or Download**: Place all files in a single directory (e.g., `search-engine/`).
- **Prepare Indexes**: Ensure `b_index.json`, `h_index.json`, `t_index.json`, `id_link.csv`, `doc_norms.csv`, and `pagerank.csv` are in the `Indexes/` directory and correctly formatted.
- **Install Dependencies**: Run the pip command above if libraries are missing.

## Usage

### Running the Search Engine

**Basic Start**:

```bash
python driver.py
```

Initializes the engine and enters interactive mode:

```
Engine initialized successfully.
Interactive Search Engine (type 'exit' to quit, 'check' to inspect indexes, 'tune' to tune parameters, 'next' for next results)
Type 'pagerank:on' or 'pagerank:off' to enable/disable PageRank for queries
Enter query:
```

**Debug Mode**:

```bash
python driver.py --debug
```

Adds debug output for queries and result sets, showing processed terms, index hits, and document IDs.

**Custom Result Count**:

```bash
python driver.py --k 5
```

Returns top 5 results per query and 5 results per `next` command (default is 10).

**Enable PageRank by Default**:

```bash
python driver.py --pagerank
```

Enables PageRank scoring for all queries unless disabled with `pagerank:off`.

### Interactive Commands

**Run a Query**:

```
Enter query: best restaurants in Rome
Running query: 'best restaurants in Rome'
Top 10 results:
1. https://rome-eats.com (Score: 15.200)
2. https://best-restaurants.com (Score: 14.800)
...
Enter query:
```

**Get Next Results**:

```
Enter query: next
Next 10 results:
11. https://italy-guide.com (Score: 13.200)
12. https://rome-food.com (Score: 13.100)
...
Enter query:
```

**Toggle PageRank**:

```
Enter query: pagerank:on
PageRank enabled for future queries.
Enter query: best restaurants in Rome
Running query: 'best restaurants in Rome'
Top 10 results:
1. https://rome-eats.com (Score: 10.500)
...
Enter query: pagerank:off
PageRank disabled for future queries.
Enter query:
```

**Check Indexes**:

```
Enter query: check
Checking indexes...
b_index:
restaurant {'postings': [{'id': 1, 'positions': [10], 'tf': 1}, ...], 'df': 50, 'idf': 1.2}
...
h_index:
restaurant {'postings': [{'id': 1, 'positions': [5], 'tf': 1}, ...], 'df': 20, 'idf': 1.5}
...
id_link:
1 https://rome-eats.com
...
```

**Tune Parameters**:

```
Enter query: tune best restaurants in Rome
Current parameters:
  k1: 1.5
  b: 0.75
  proximity_weight_b: 0.5
  proximity_weight_h: 0.3
  proximity_weight_t: 0.5
  proximity_scale: 10.0
  b_contri: 0.16666666666666666
  h_contri: 0.3333333333333333
  t_contri: 0.5
  bm25_weight: 0.7
  pagerank_weight: 0.3
  Using PageRank: False
Select parameter to tune (or 'exit' to return):
  1. k1 - BM25 term frequency saturation
  10. bm25_weight - BM25 score weight
  11. pagerank_weight - PageRank score weight
  12. Toggle PageRank usage
Enter choice: 10
Tuning bm25_weight (current value: 0.7)
Enter new value for bm25_weight: 0.8
Set bm25_weight to 0.8
Running query: 'best restaurants in Rome'
Top 10 results:
1. https://rome-eats.com (Score: 16.200)
...
```

**Exit**:

```
Enter query: exit
Exiting...
```

## Parameters

**BM25**:

- `k1` (1.5): Controls term frequency saturation.
- `b` (0.75): Controls length normalization.

**Proximity**:

- `proximity_weight_b` (0.5): Weight of proximity boost for body index.
- `proximity_weight_h` (0.3): Weight of proximity boost for header index.
- `proximity_weight_t` (0.5): Weight of proximity boost for title index.
- `proximity_scale` (10.0): Adjusts distance sensitivity in proximity boost.

**Zone Weights**:

- `b_contri` (1/6): Body index contribution.
- `h_contri` (1/3): Header index contribution.
- `t_contri` (1/2): Title index contribution.

**Scoring Weights**:

- `bm25_weight` (0.7): Weight for BM25 score in weighted average (when PageRank is enabled).
- `pagerank_weight` (0.3): Weight for PageRank score in weighted average (when PageRank is enabled).

## Scoring Details

- **When PageRank is Disabled**: Documents are scored using raw BM25 scores, incorporating zone contributions (body, header, title) and proximity boosts.
- **When PageRank is Enabled**: Final score is a weighted average: \[ \\text{final_score} = \\text{bm25_weight} \\cdot \\text{bm25_score} + \\text{pagerank_weight} \\cdot \\text{pagerank_score} \]
- **Proximity Boost**: Increases scores for documents where query terms appear closer together, controlled by `proximity_weight_*` and `proximity_scale`.
