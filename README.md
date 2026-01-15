Wikipedia Search Engine

This repository contains a high-performance search engine optimized for Wikipedia's 6.3 million documents. The system is designed specifically to operate within a strict 4GB RAM limit while utilizing a two-stage retrieval process that balances speed and accuracy.

📂 Project Organization

The implementation is structured into several key components to ensure both efficiency and maintainability.
Initialization and Pre-loading

Handled by the initialize_engine_parallel() function, this section performs high-speed loading of document titles and metadata into memory.

It uses 8 parallel threads to fully saturate network bandwidth during startup.
Query Processing

This module tokenizes input strings, applies Porter Stemming to handle word variations, and filters out common English stopwords to focus on meaningful terms.
Retrieval Engine

We implement the BM25 scoring model to quickly identify the most relevant document candidates from the body index.
Re-ranking Logic

To finalize results, the engine performs a title-based Cosine Similarity check on a candidate subset.
On-Demand Data Access

To save memory, specialized functions like get_pageview_values and get_pagerank_values fetch data from the cloud only when required via selective shard-loading.

⚙️ Key Functionalities
1. Required Configuration

The engine requires a local configuration file named ProjAndBucket.txt to connect to Google Cloud Storage. If this file is missing or formatted incorrectly, the system will throw a fatal exception and stop the initialization process.

The file must follow this structure exactly:

"

PROJECT_ID=your-gcp-project-id

BUCKET_NAME=your-bucket-name

"

2. Memory Management (The 4GB Constraint)

Staying under the 4GB RAM limit is achieved through a "Lazy-Loading" strategy:

    Resident in RAM: Only essential metadata, such as the full Title Dictionary (for O(1) lookups) and the DocLen Dictionary, are stored in memory at all times.

    Fetched On-Demand: Heavy datasets like PageRank and PageViews are not loaded at startup. Instead, the engine groups requested IDs by shard and fetches only the specific files needed from the cloud, minimizing memory spikes.

3. Hyper-Parameter Tuning

The /search route supports real-time experimentation through 7 adjustable hyperparameters passed as URL parameters:

    k1: BM25 term frequency saturation constant. Default: 1.5.

    b: BM25 document length normalization constant. Default: 0.75.

    w_title: Weight of the Title Cosine Similarity in the final score. Default: 0.8.

    threshold: The maximum document frequency for a term before it is pruned. Default: 600,000.

    pool: Number of top candidates passed from the body index to the re-ranker. Default: 100.

    stem_w: The relative weight given to stemmed query tokens. Default: 0.5.

    list_penalty: A multiplier used to lower the rank of "List of" or "Timeline of" articles. Default: 0.2.

4. Search Logic and Fallback

To maintain high search speeds, terms appearing in more than 600,000 documents are automatically pruned.

If a query consists entirely of common terms that exceed the threshold, the engine uses fallback logic to process the term with the lowest document frequency among them, ensuring the user always receives results.
