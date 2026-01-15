from flask import Flask, request, jsonify
import re, pickle, math, pandas as pd
from collections import Counter
from operator import itemgetter
from google.cloud import storage
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
import os

# --- GCP CONFIGURATION ---
PROJECT_ID = ''
BUCKET_NAME = ''

# --- BUCKET PATHS ---
BODY_INDEX_BLOB = 'postings_gcp/index.pkl'
BODY_POSTINGS_PREFIX = 'postings_gcp'
TITLE_SHARDS_PREFIX = 'titles_shards'
DOCLEN_SHARDS_PREFIX = 'doclen_shards'
ANCHOR_INDEX_BLOB = 'postings_anchor_gcp/anchor_index.pkl'
ANCHOR_POSTINGS_PREFIX = 'postings_anchor_gcp'
TITLE_POSTINGS_PREFIX = 'postings_title_gcp'
TITLE_INDEX_BLOB = 'postings_title_gcp/title_index.pkl' 
PAGEVIEWS_SHARDS_PREFIX = 'pageviews_shards/2021-08'

app = Flask(__name__)

def load_essential_config(file_path='ProjAndBucket.txt'):
    # Throw exception if the file is missing
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical Error: The configuration file '{file_path}' was not found. System cannot start.")

    config = {}
    with open(file_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key] = value

    # Check for specific required keys
    if 'PROJECT_ID' not in config or 'BUCKET_NAME' not in config:
        raise KeyError("Config Error: 'PROJECT_ID' and 'BUCKET_NAME' must be defined in config.txt")
        
    return config

# Load values at the global level
try:
    config_data = load_essential_config()
    PROJECT_ID = config_data['PROJECT_ID']
    BUCKET_NAME = config_data['BUCKET_NAME']
    print(f"Config loaded for Project: {PROJECT_ID}")
except Exception as e:
    # This will stop the server from even initializing
    print(f"FATAL STARTUP ERROR: {e}")
    raise

# --- INITIALIZE TOOLS ---
stemmer = PorterStemmer()
nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
RE_WORD = re.compile(r"[\#\@\w](['\-]?\w){2,24}", re.UNICODE)

# --- GLOBALS ---
_STORAGE_CLIENT = storage.Client(project=PROJECT_ID)
_BUCKET = _STORAGE_CLIENT.bucket(BUCKET_NAME)
_INDEX_LOCK = threading.Lock()

DOCLEN_DICT = {} 
LIST_IDS = set()
TITLE_DICT = {}
AVG_DOC_LEN = 0
N_DOCS = 0
PV_SHARD_SIZE = 100000

_BODY_INDEX = None
_ANCHOR_INDEX = None
_TITLE_INDEX = None

# --- PARALLEL INITIALIZATION (STARTUP) ---

def load_doclen_shard(blob):
    """Worker to load doclen shard."""
    if blob.name.endswith('.pkl'):
        return pickle.loads(blob.download_as_bytes())
    return {}

def load_title_shard_to_memory(blob):
    """Worker to load titles into the global dict."""
    if blob.name.endswith('.pkl'):
        return pickle.loads(blob.download_as_bytes())
    return {}

def load_pageview_shard(blob):
    """Worker to load pageview shards."""
    if blob.name.endswith('.pkl'):
        return pickle.loads(blob.download_as_bytes())
    return {}

def initialize_engine_parallel():
    """Startup with BM25 and Full Title pre-loading. PageRank is excluded for RAM."""
    global DOCLEN_DICT, LIST_IDS, AVG_DOC_LEN, N_DOCS, TITLE_DICT
    print("--- STARTING PARALLEL INITIALIZATION (RAM OPTIMIZED) ---")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 1. Parallel Load DocLens & Calculate AVG_LEN
        d_blobs = list(_BUCKET.list_blobs(prefix=f'{DOCLEN_SHARDS_PREFIX}/'))
        total_len = 0
        for shard in executor.map(load_doclen_shard, d_blobs):
            for k, v in shard.items():
                d_id, d_len = int(k), int(v)
                DOCLEN_DICT[d_id] = d_len
                total_len += d_len
        
        N_DOCS = len(DOCLEN_DICT)
        AVG_DOC_LEN = total_len / N_DOCS if N_DOCS > 0 else 0
            
        # 2. Parallel Load Titles and Mark List IDs
        t_blobs = list(_BUCKET.list_blobs(prefix=f'{TITLE_SHARDS_PREFIX}/'))
        for shard in executor.map(load_title_shard_to_memory, t_blobs):
            for k, v in shard.items():
                tid = int(k)
                title_str = str(v)
                TITLE_DICT[tid] = title_str
                if title_str.lower().startswith(("list of", "timeline of")):
                    LIST_IDS.add(tid)

    print(f"COMPLETED: Titles: {len(TITLE_DICT)}")
    gc.collect()

# --- LAZY LOADERS ---

def get_body_index():
    global _BODY_INDEX
    if _BODY_INDEX is None:
        with _INDEX_LOCK:
            if _BODY_INDEX is None:
                _BODY_INDEX = pickle.loads(_BUCKET.blob(BODY_INDEX_BLOB).download_as_bytes())
    return _BODY_INDEX

def get_title_index():
    global _TITLE_INDEX
    if _TITLE_INDEX is None:
        with _INDEX_LOCK:
            if _TITLE_INDEX is None:
                _TITLE_INDEX = pickle.loads(_BUCKET.blob(TITLE_INDEX_BLOB).download_as_bytes())
    return _TITLE_INDEX

def get_anchor_index():
    global _ANCHOR_INDEX
    if _ANCHOR_INDEX is None:
        with _INDEX_LOCK:
            if _ANCHOR_INDEX is None:
                _ANCHOR_INDEX = pickle.loads(_BUCKET.blob(ANCHOR_INDEX_BLOB).download_as_bytes())
    return _ANCHOR_INDEX

def get_real_title(doc_id: int) -> str:
    # O(1) Memory lookup - no network delay
    return TITLE_DICT.get(int(doc_id), f"ID {doc_id}")

# --- CORE FUNCTIONS ---

def tokenize(text, stem=False):
    tokens = [t.group() for t in RE_WORD.finditer(text.lower()) if t.group() not in english_stopwords]
    return [stemmer.stem(t) for t in tokens] if stem else tokens

def get_posting_list(index, token, postings_prefix):
    """Parallel worker to fetch a single term's postings."""
    if token not in index.posting_locs:
        return token, []
    
    postings = []
    TUPLE_SIZE, TF_MASK = 6, 2**16 - 1
    file_name, offset = index.posting_locs[token][0]
    blob = _BUCKET.blob(f'{postings_prefix}/{file_name}')
    df = index.df.get(token, 0)
    
    b = blob.download_as_bytes(start=offset, end=offset + (df * TUPLE_SIZE) - 1)
    for i in range(0, len(b), TUPLE_SIZE):
        packed = int.from_bytes(b[i:i+TUPLE_SIZE], 'big')
        postings.append((packed >> 16, packed & TF_MASK))
    return token, postings

# --- OPTIMIZED SEARCH ROUTE ---

@app.route("/search")
def search():
    query = request.args.get('query', '')
    if not query: return jsonify({"results": []})

    K1 = float(request.args.get('k1', 1.5))
    B = float(request.args.get('b', 0.75))
    W_TITLE = float(request.args.get('w_title', 0.8))
    W_BODY = 1.0 - W_TITLE
    LIST_PENALTY = float(request.args.get('list_penalty', 0.2))
    MAX_DF_THRESHOLD = int(request.args.get('threshold', 600000))
    DOC_AMOUNT_CHECK = int(request.args.get('pool', 100))
    STEM_WEIGHT = float(request.args.get('stem_w', 0.5))

    try:
        body_idx = get_body_index()
        raw_tokens = tokenize(query, stem=False)
        stemmed_tokens = tokenize(query, stem=True)
        expanded_query = list(set(raw_tokens + stemmed_tokens))
        
        q_mag = math.sqrt(len(set(stemmed_tokens)))
        body_scores = Counter()
        
        valid_query_terms = [t for t in expanded_query if body_idx.df.get(t, 0) < MAX_DF_THRESHOLD]
        # If all terms are above threshold, pick the one with the smallest DF among heavy terms
        if not valid_query_terms and expanded_query:
            best_heavy_term = min(expanded_query, key=lambda t: body_idx.df.get(t, 10000000))
            valid_query_terms = [best_heavy_term]
            print(f"Fallback: All terms heavy. Using {best_heavy_term}")

        # Parallel Retrieval
        with ThreadPoolExecutor(max_workers=len(expanded_query) or 1) as executor:
            futures = [executor.submit(get_posting_list, body_idx, t, BODY_POSTINGS_PREFIX) for t in valid_query_terms]
            
            for future in futures:
                token, postings = future.result()
                df = body_idx.df.get(token, 1)
                
                # BM25 IDF
                idf = math.log10((N_DOCS - df + 0.5) / (df + 0.5) + 1)
                term_weight = 1.0 if token in raw_tokens else STEM_WEIGHT
                
                for doc_id, tf in postings:
                    did_int = int(doc_id)
                    d_len = DOCLEN_DICT.get(did_int, AVG_DOC_LEN)
                    
                    # BM25 Term Frequency Saturation
                    numerator = tf * (K1 + 1)
                    denominator = tf + K1 * (1 - B + B * d_len / AVG_DOC_LEN)
                    bm25_score = idf * (numerator / denominator) * term_weight
                    
                    if did_int in LIST_IDS: bm25_score *= LIST_PENALTY
                    body_scores[did_int] += bm25_score

        if not body_scores: return jsonify({"results": []})

        top_candidates = body_scores.most_common(DOC_AMOUNT_CHECK)
        max_body = top_candidates[0][1]
        candidate_ids = [cid for cid, score in top_candidates]
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Map the doc_ids to the get_real_title function
            # This returns titles in the same order as candidate_ids
            raw_titles = list(executor.map(get_real_title, candidate_ids))

        final_list = []
        for i, (doc_id, b_score) in enumerate(top_candidates):
            raw_title = raw_titles[i] # Access the pre-fetched title
            title_tokens = tokenize(raw_title, stem=True)
            dot = len(set(stemmed_tokens).intersection(set(title_tokens)))
            t_mag = math.sqrt(len(set(title_tokens)) or 1)
            title_cos = dot / (q_mag * t_mag) if (q_mag * t_mag) > 0 else 0
            
            total_score = (W_TITLE * title_cos) + (W_BODY * (b_score / max_body))
            final_list.append((str(doc_id), raw_title, total_score))

        sorted_res = sorted(final_list, key=itemgetter(2), reverse=True)[:30]
        return jsonify({"results": [[d, t] for d, t, s in sorted_res]})

    except Exception as e: return jsonify({"error": str(e)})

# --- ADDITIONAL FEATURES ---

def get_pagerank_values(ids):
    """Fetches PageRank from GCS on-demand to save RAM."""
    try:
        blobs = list(_BUCKET.list_blobs(prefix='pr/'))
        # Load and filter only the requested IDs to minimize memory spike
        dfs = [pd.read_csv(b.open("rb"), compression='gzip', names=['id', 'pr'])
               for b in blobs if b.name.endswith('.csv.gz')]
        if not dfs: return [0.0] * len(ids)
        
        full_pr_map = dict(zip(pd.concat(dfs)['id'], pd.concat(dfs)['pr']))
        return [float(full_pr_map.get(int(wid), 0)) for wid in ids]
    except:
        return [0.0] * len(ids)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ids = request.get_json()
    if not ids: return jsonify([])
    # Called only when needed
    return jsonify(get_pagerank_values(ids))

def _fetch_shard(prefix, sid):
    """Parallel worker to fetch a specific data shard from GCS."""
    try:
        blob = _BUCKET.blob(f"{prefix}/{sid:06d}.pkl")
        if blob.exists():
            return pickle.loads(blob.download_as_bytes())
    except: pass
    return {}

def get_pageview_values(ids):
    """
    Optimized: Splits the requested IDs into threads to fetch only required shards.
    This avoids downloading the entire 60+ file database for small requests.
    """
    if not ids: return []
    # 1. Group IDs by Shard to minimize redundant downloads
    required_sids = set(int(wid) // PV_SHARD_SIZE for wid in ids)
    
    pv_map = {}
    # 2. Parallel Fetching: Only fire requests for the shards we need
    with ThreadPoolExecutor(max_workers=min(len(required_sids), 10)) as executor:
        for shard_data in executor.map(lambda sid: _fetch_shard(PAGEVIEWS_SHARDS_PREFIX, sid), required_sids):
            pv_map.update(shard_data)
            
    return [int(pv_map.get(int(wid), 0)) for wid in ids]
    
@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ids = request.get_json(silent=True) or []
    return jsonify(get_pageview_values(ids))

@app.route("/search_body")
def search_body():
    query = request.args.get('query', '')
    if not query: return jsonify([])
    try:
        index = get_body_index()
        tokens = tokenize(query, stem=False)
        scores = Counter()
        for token in tokens:
            if token in index.posting_locs:
                idf = math.log10(6348910 / index.df.get(token, 1))
                for doc_id, tf in get_posting_list(index, token, BODY_POSTINGS_PREFIX)[1]:
                    scores[doc_id] += (tf * idf)
        top30 = scores.most_common(30)
        return jsonify([(str(doc_id), get_real_title(doc_id)) for doc_id, _ in top30])
    except: return jsonify([])

@app.route("/search_title")
def search_title():
    query = request.args.get('query', '')
    if not query: return jsonify([])
    try:
        tindex = get_title_index()
        counts = Counter()
        tokens = set(tokenize(query, stem=False))
        for token in tokens:
            for doc_id, _ in get_posting_list(tindex, token, TITLE_POSTINGS_PREFIX)[1]:
                counts[doc_id] += 1
        top30 = counts.most_common(30)
        return jsonify([(str(doc_id), get_real_title(doc_id)) for doc_id, _ in top30])
    except: return jsonify([])

@app.route("/search_anchor")
def search_anchor():
    query = request.args.get('query', '')
    if not query: return jsonify([])
    try:
        aindex = get_anchor_index()
        counts = Counter()
        tokens = set(tokenize(query, stem=False))
        for token in tokens:
            for doc_id, _ in get_posting_list(aindex, token, ANCHOR_POSTINGS_PREFIX)[1]:
                counts[doc_id] += 1
        top30 = counts.most_common(30)
        return jsonify([(str(doc_id), get_real_title(doc_id)) for doc_id, _ in top30])
    except: return jsonify([])

def run(**kwargs):
    """Allows se.run() call from your launcher code."""
    app.run(**kwargs)

initialize_engine_parallel()

if __name__ == '__main__':
    #load_all_doc_lengths() # Pre-load to avoid network latency during search
    app.run(host='0.0.0.0', port=8080)