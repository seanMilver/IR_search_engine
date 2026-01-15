Wikipedia Search Engine

This repository contains a high-performance search engine optimized for Wikipedia's 6.3 million documents. The system is designed to run on a machine with a strict 4GB RAM limit, utilizing a two-stage retrieval process that balances speed and accuracy.
📂 Project Organization

The implementation is organized into several key components to ensure maintainability and efficiency:

    Initialization & Pre-loading: Located in initialize_engine_parallel(), this section handles the high-speed loading of document titles and metadata into memory using 8 parallel threads to saturate network bandwidth.

    Query Processing: Tokenizes input strings, applies Porter Stemming, and filters out English stopwords.

    Retrieval Engine: Implements the BM25 scoring model to identify the most relevant document candidates from the body index.

    Re-ranking Logic: Uses a title-based Cosine Similarity check on a subset of candidates to finalize the results.

    On-Demand Data Access: Specialized functions like get_pageview_values and get_pagerank_values fetch data from the cloud only when needed via selective shard-loading.

⚙️ Key Functionalities
1. Required Configuration

The engine requires a local configuration file named ProjAndBucket.txt to establish a connection with Google Cloud Storage. If this file is missing or incomplete, the system will throw a fatal exception and stop initialization.

The file must follow this exact structure:
"
PROJECT_ID=your-gcp-project-id
BUCKET_NAME=your-bucket-name
"
2. Memory Management (The 4GB Constraint)

To stay under the 4GB limit, the engine uses a "Lazy-Loading" strategy:

    Resident in RAM: The full Title Dictionary (for O(1) lookups) and the DocLen Dictionary are loaded at startup.

    Fetched On-Demand: PageRank and PageViews are not stored in memory. Instead, the engine groups requested IDs by shard and fetches only the specific files required from the cloud to minimize RAM spikes.

3. Hyper-Parameter Tuning

The /search route supports real-time experimentation through 7 adjustable hyperparameters passed as URL parameters:

Parameter	   Description	                                                      Default
k1	         BM25 term frequency saturation constant.	                          1.5
b	           BM25 document length normalization constant.	                      0.75
w_title	     Weight of the Title Cosine Similarity in the final score.	        0.8
threshold	   Max document frequency for a query term before pruning.	          600,000
pool	       Number of top candidates passed to the re-ranker.	                100
stem_w	     The relative weight given to stemmed query tokens.	                0.5
list_penalty multiplier applied to "List of" or "Timeline of" articles.	        0.2

4. Search Logic & Fallback

    Dynamic Thresholding: To maintain speed, terms appearing in over 600,000 documents are pruned. If all query terms are common, the engine uses fallback logic to process the term with the lowest document frequency among them.
