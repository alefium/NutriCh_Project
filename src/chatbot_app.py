# Import required libraries for Flask chatbot application
import chromadb  # Vector database for retrieving stored embeddings
import litellm  # For LLM integration with OpenRouter
import logging  # For debug logging
import numpy as np  # For numerical operations on similarity scores
import os  # For environment variables
import re  # For text normalization (whitespace collapsing)
import time  # For brief retries on Windows file locks
import unicodedata  # For Unicode normalization (NFKC)

from dotenv import load_dotenv  # For loading .env file
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer  # For generating query embeddings
from utils import normalize_text  # Shared normalization

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application for the chatbot
app = Flask(__name__)
# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)
# Set secret key for session management
app.secret_key = os.environ["FLASK_SECRET_KEY"]

# Configure LiteLLM for OpenRouter
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
LLM_MODEL = os.getenv('LLM_MODEL', 'openrouter/meta-llama/llama-3.3-70b-instruct')
APP_NAME = os.getenv('APP_NAME', 'NutRich-Chatbot')

# Configure litellm
if OPENROUTER_API_KEY:
    litellm.api_key = OPENROUTER_API_KEY
    litellm.api_base = "https://openrouter.ai/api/v1"
    litellm.headers = {
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://yourdomain.com"),
        "X-Title": APP_NAME
    }
    logger.info(f"LiteLLM configured with OpenRouter using model: {LLM_MODEL}")
else:
    logger.warning("OPENROUTER_API_KEY not found in environment variables. LLM features will be disabled.")

# Initialize sentence transformer model for generating query embeddings
# Using the same model as the main app for consistency
model = SentenceTransformer('all-MiniLM-L6-v2')

# Removed local normalize_text; imported from utils

# --- Similarity/Distance helpers (single source of truth) ---
def cosine_sim_and_distance01_from_vectors(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    s = float(np.dot(a, b) / ((np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)))
    return s, (1.0 - s) / 2.0  # (similarity_-1_1, distance_0_1)

def cosine_sim_from_chroma_distance(d):
    # Chroma cosine distance is 0..2 where d = 1 - cosine_similarity
    return 1.0 - float(d)

def distance01_from_chroma_distance(d):
    # Map Chroma distance 0..2 to 0..1 by dividing by 2
    return float(d) / 2.0

# Retrieves similarity threshold for relevant matches
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

# Number of top matches to retrieve and display
TOP_K_MATCHES = 3

# Global variables for current dataset
current_dataset = None  # No default dataset initially

# Initialize with first available dataset (if any)
def initialize_current_dataset():
    global current_dataset
    available_datasets = list_available_datasets()
    if not available_datasets:
        # No datasets available, ensure current_dataset is None
        current_dataset = None
        logger.info("Chatbot initialized with no active dataset")
    elif current_dataset is None:
        # No current dataset but datasets exist, pick first one
        current_dataset = available_datasets[0]
        logger.info(f"Chatbot initialized with existing dataset: {current_dataset}")
    elif current_dataset not in available_datasets:
        # Current dataset no longer exists, pick first available or set to None
        if available_datasets:
            current_dataset = available_datasets[0]
            logger.info(f"Previous dataset no longer exists, switched to: {current_dataset}")
        else:
            current_dataset = None
            logger.info("Previous dataset no longer exists and no datasets available")

def perform_manual_similarity_search(query_lower, collection, query_embedding):
    """
    Perform manual similarity search when ChromaDB indexing fails
    This bypasses the corrupted index and manually computes similarities
    """
    try:
        logger.info("Performing manual similarity search as fallback")
        
        # Get all entries directly from storage
        all_data = collection.get(include=['documents', 'metadatas'])
        all_docs = all_data.get('documents', [])
        all_metas = all_data.get('metadatas', [])
        # Flatten if Chroma returns nested lists
        if all_docs and isinstance(all_docs[0], list):
            all_docs = all_docs[0]
        if all_metas and isinstance(all_metas[0], list):
            all_metas = all_metas[0]
        
        if not all_docs:
            logger.info("No documents found in manual fallback")
            return None, []
        
        
        
        # Manually compute similarities for all entries (batch for speed)
        doc_embs = model.encode(all_docs, batch_size=64, convert_to_numpy=True, normalize_embeddings=False)
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = np.linalg.norm(q) + 1e-8
        dn = np.linalg.norm(doc_embs, axis=1) + 1e-8
        sims = (doc_embs @ q) / (dn * qn)  # cosine in [-1,1]
        d01 = (1.0 - sims) / 2.0          # distance in [0,1]

        manual_results = []
        for i, doc in enumerate(all_docs):
            try:
                if i < len(all_metas) and isinstance(all_metas[i], dict) and 'answer' in all_metas[i] and isinstance(doc, str):
                    manual_results.append({
                        'question': doc,
                        'answer': all_metas[i]['answer'],
                        'similarity_-1_1': float(sims[i]),
                        'distance_0_1': float(d01[i])
                    })
            except Exception:
                pass
        
        # Sort by similarity (highest first)
        manual_results.sort(key=lambda x: x['similarity_-1_1'], reverse=True)
        
        # Filter by threshold and get top matches, but always fill up to TOP_K_MATCHES
        relevant_matches = [r for r in manual_results if r['similarity_-1_1'] >= SIMILARITY_THRESHOLD]
        best_match = relevant_matches[0] if relevant_matches else None
        top_matches = list(relevant_matches[:TOP_K_MATCHES])
        # If not enough above threshold, fill with remaining highest-similarity results
        if len(top_matches) < TOP_K_MATCHES:
            used_questions = {m['question'] for m in top_matches}
            for r in manual_results:
                if r['question'] in used_questions:
                    continue
                top_matches.append(r)
                if len(top_matches) >= TOP_K_MATCHES:
                    break
        # If no matches above threshold at all, choose best overall
        if not best_match and manual_results:
            best_match = manual_results[0]
        
        logger.info(f"Manual fallback completed: {len(relevant_matches)} relevant matches found")
        return best_match, top_matches
        
    except Exception as e:
        logger.error(f"Manual similarity search also failed: {e}")
        return None, []

def get_collection(dataset_name=None):
    """
    Get the ChromaDB collection containing question-answer pairs for a specific dataset
    Refreshes the client connection each time to pick up new entries
    
    Args:
        dataset_name (str): Name of the dataset to use. If None, uses current_dataset
    
    Returns:
        Collection: ChromaDB collection object or None if not found
    """
    if dataset_name is None:
        dataset_name = current_dataset
    
    collection_name = f"qa_collection_{dataset_name}"
    
    # Retry a few times to avoid transient access issues on Windows after writes
    last_error = None
    for attempt in range(3):
        try:
            fresh_chroma_client = chromadb.PersistentClient(path="./vector_db")
            collection = fresh_chroma_client.get_or_create_collection(
                name=collection_name, 
                metadata={"hnsw:space": "cosine", "dataset": dataset_name}
            )
            
            # Note: Enhanced persistence removed - using automatic fallback instead
            
            # Log current count if possible
            try:
                _ = collection.count() if hasattr(collection, 'count') else len((collection.get() or {}).get('ids', []))
            except Exception:
                pass
            return collection
        except Exception as e:
            last_error = e
            time.sleep(0.3)
    # Return None if collection doesn't exist or there's an error
    logger.error(f"Error accessing collection '{collection_name}' after retries: {last_error}")
    return None

def call_llm_for_answer(user_question, db_answer=None, has_relevant_answer=False):
    """
    Call LLM through LiteLLM to generate an answer based on user question and database result
    
    Args:
        user_question (str): The user's original question
        db_answer (str): Answer from the database (if any)
        has_relevant_answer (bool): Whether database has a relevant answer above threshold
    
    Returns:
        str: LLM-generated response
    """
    try:
        if not OPENROUTER_API_KEY:
            return "LLM service is not configured. Please set up your OpenRouter API key."
        
        # Construct the prompt based on whether we have a relevant database answer
        if has_relevant_answer and db_answer:
            prompt = f"""You are a helpful nutrition assistant. A user has asked a question, and we found relevant information in our FAQ database.

User Question: {user_question}

From our FAQ database, the answer is: {db_answer}

Please provide a polite, concise, helpful response to the user's question based on this database information. You can rephrase, expand, or clarify the database answer to make it more conversational and helpful, but stay true to the core information provided.
Please don't mention to the user that you are using a database to answer their question."""
        else:
            prompt = f"""You are a helpful nutrition assistant. A user has asked a question, but we don't have specific information about this topic in our FAQ database.

User Question: {user_question}

We don't have information on that in our database. Please politely and concisely let the user know that we don't have information on their specific question and suggest they try a different question. Be helpful and encouraging."""

        # Call LiteLLM
        response = litellm.completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful nutrition assistant providing polite, concise and accurate responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.4
        )
        
        # Extract the response text
        llm_answer = response.choices[0].message.content.strip()
        return llm_answer
        
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        # Fallback response
        if has_relevant_answer and db_answer:
            return f"Based on our database: {db_answer}"
        else:
            return "I'm sorry, I don't have information on that topic. Please try a different question."

def list_available_datasets():
    """
    List all available datasets by scanning ChromaDB collections
    
    Returns:
        list: List of dataset names (empty list if none exist)
    """
    try:
        fresh_chroma_client = chromadb.PersistentClient(path="./vector_db")
        collections = fresh_chroma_client.list_collections()
        datasets = []
        
        for coll in collections:
            # Extract dataset name from collection name (qa_collection_<dataset>)
            if coll.name.startswith("qa_collection_"):
                dataset_name = coll.name.replace("qa_collection_", "")
                datasets.append(dataset_name)
        
        return sorted(datasets)
    except Exception as e:
        logger.warning(f"Error listing datasets: {e}")
        return []  # Return empty list on error

def search_similar_questions(query, collection):
    """
    Perform semantic similarity search on the vector database
    
    Args:
        query (str): User's question to search for
        collection: ChromaDB collection containing stored questions
    
    Returns:
        tuple: (best_match, top_matches) where:
            - best_match: dict with highest similarity match or None
            - top_matches: list of dicts with top matches above threshold
    """
    try:
        # Normalize query for consistency with stored data
        query_lower = normalize_text(query)
        
        # Generate embedding for the normalized user's query
        query_embedding = model.encode(query_lower).tolist()
        
        # Perform similarity search in the vector database
        # n_results=10 ensures we get enough results to filter by threshold
        # Determine how many results to fetch to improve recall for newly added items
        # Force a fresh count check to detect recently uploaded entries
        try:
            # Try multiple methods to get accurate count
            if hasattr(collection, 'count'):
                total_count = collection.count()
            else:
                fresh_data = collection.get()
                total_count = len(fresh_data.get('ids', []))
            
            # Double-check with get() if count seems suspicious
            if total_count == 0:
                fresh_data = collection.get()
                actual_count = len(fresh_data.get('documents', []))
                if actual_count > 0:
                    logger.warning(f"Count mismatch detected: collection.count()={total_count} but get() shows {actual_count} documents")
                    total_count = actual_count
                    
        except Exception:
            total_count = 10
        n_results = max(10, min(total_count, 100))
        
        # Omit verbose pre-query debug verification
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['metadatas', 'documents', 'distances']
            )
        except Exception as query_error:
            logger.warning(f"ChromaDB query failed with error: {query_error}")
            logger.info("ChromaDB index appears corrupted - immediately triggering manual fallback")
            fallback_result = perform_manual_similarity_search(query_lower, collection, query_embedding)
            if fallback_result[0] or fallback_result[1]:  # If fallback found something
                logger.info("Manual fallback successfully recovered from ChromaDB query failure")
                return fallback_result
            else:
                logger.info("Manual fallback also found no results after ChromaDB query failure")
                return None, []
        # Omit verbose query results summary
        
        # Process results if any were found
        if not results['ids'] or not results['ids'][0]:
            logger.info("No results returned from ChromaDB query - attempting fallback recovery")
            # Fallback: try getting all entries directly and do manual similarity search
            fallback_result = perform_manual_similarity_search(query_lower, collection, query_embedding)
            if fallback_result[0] or fallback_result[1]:  # If fallback found something
                logger.info("Manual fallback found entries after ChromaDB returned empty results")
                return fallback_result
            else:
                logger.info("Manual fallback confirmed no entries exist or no relevant matches")
                return None, []
        
        # Extract and process similarity scores
        # ChromaDB returns distances (lower = more similar), convert to similarity
        distances = results.get('distances', [[]])[0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        ids = []  # Chroma 0.4.x does not support returning ids via include
        
        # Convert Chroma distances (0..2) to cosine similarity (-1..1) and distance (0..1)
        sim_list = []
        dist01_list = []
        for d in distances:
            try:
                if d is None:
                    sim_list.append(-2.0)  # sentinel smaller than min valid
                    dist01_list.append(1.0)
                else:
                    d_val = float(d)
                    # clamp to 0..2
                    d_val = 0.0 if d_val < 0.0 else (2.0 if d_val > 2.0 else d_val)
                    sim_list.append(cosine_sim_from_chroma_distance(d_val))
                    dist01_list.append(distance01_from_chroma_distance(d_val))
            except Exception:
                sim_list.append(-2.0)
                dist01_list.append(1.0)
        
        # Combine all result data for easier processing (skip malformed entries)
        combined_results = []
        skipped_missing_meta = 0
        skipped_details = []
        max_len = min(len(sim_list), len(documents), len(metadatas), len(distances) if isinstance(distances, list) else 0)
        for i in range(max_len):
            meta = metadatas[i]
            doc = documents[i]
            if not isinstance(meta, dict) or 'answer' not in meta or doc is None:
                skipped_missing_meta += 1
                try:
                    skipped_details.append({
                        'index': i,
                        'meta_is_dict': isinstance(meta, dict),
                        'meta_type': type(meta).__name__ if meta is not None else 'NoneType',
                        'meta_keys': list(meta.keys()) if isinstance(meta, dict) else None,
                        'has_answer_key': ('answer' in meta) if isinstance(meta, dict) else False,
                        'doc_is_str': isinstance(doc, str),
                        'doc_is_none': doc is None,
                        'doc_preview': (doc[:60] + '...') if isinstance(doc, str) and len(doc) > 60 else (doc if isinstance(doc, str) else None)
                    })
                except Exception:
                    pass
                continue
            combined_results.append({
                'question': doc,
                'answer': meta['answer'],
                'similarity_-1_1': sim_list[i],
                'distance_0_1': dist01_list[i]
            })
        if skipped_missing_meta:
            pass

        # Check if we got significantly fewer results than expected (potential index corruption)
        expected_docs = min(n_results, total_count)
        actual_docs = len(combined_results)
        
        # Always try fallback if we have no results, regardless of expected count
        # This handles the case where database was empty but now has entries after upload
        if actual_docs == 0:
            logger.warning("ChromaDB query returned 0 results - trying manual fallback to check for recently added entries")
            fallback_result = perform_manual_similarity_search(query_lower, collection, query_embedding)
            if fallback_result[0] or fallback_result[1]:  # If fallback found something
                logger.info("Manual fallback found entries that ChromaDB query missed")
                return fallback_result
            else:
                logger.info("Manual fallback also found no results - database is truly empty or no relevant matches")
        elif actual_docs < expected_docs:  # Missing some results indicate indexing lag
            logger.warning(f"ChromaDB query returned {actual_docs}/{expected_docs} expected results - indexing lag detected")
            logger.info("Triggering manual fallback search to ensure all entries (including newest) are considered")
            return perform_manual_similarity_search(query_lower, collection, query_embedding)

        # Log ALL retrieved, well-formed entries with their metadata and similarity
        

        # Log first few distances/similarities for diagnostics
        
        
        # Sort all results by similarity score (highest first)
        combined_results.sort(key=lambda x: x['similarity_-1_1'], reverse=True)
        
        # Filter results that meet the similarity threshold for "best match" determination
        relevant_matches = [
            result for result in combined_results 
            if float(result.get('similarity_-1_1', -2.0)) >= SIMILARITY_THRESHOLD
        ]
        
        # Get the best match (highest similarity that meets threshold)
        best_match = relevant_matches[0] if relevant_matches else None
        
        # Always get top K matches for display, regardless of threshold
        # This ensures we always show top 3 unless there are fewer than 3 records
        top_matches = combined_results[:TOP_K_MATCHES]

        # If nothing passes the threshold, fall back to distance-based ranking for best match
        if not best_match and combined_results:
            
            # Use the highest similarity result as best match even if below threshold
            best_match = combined_results[0]

        # Boost exact string match if present
        for r in combined_results:
            if isinstance(r.get('question'), str) and r['question'].strip() == query_lower:
                
                best_match = r
                # Rebuild top_matches keeping best first and fill with next best by distance
                rest = [x for x in combined_results if x is not r]
                rest.sort(key=lambda x: x.get('distance_0_1', float('inf')))
                top_matches = [r] + rest[:max(0, TOP_K_MATCHES - 1)]
                break

        # If still no match or exact match not in the top-N retrieved, try direct document filter
        if not best_match:
            try:
                direct = collection.get(where_document={"$contains": query_lower}, include=["documents", "metadatas"])
                docs_direct = (direct or {}).get('documents') or []
                metas_direct = (direct or {}).get('metadatas') or []
                # Flatten single-level lists if needed
                if docs_direct and isinstance(docs_direct[0], list):
                    docs_direct = docs_direct[0]
                if metas_direct and isinstance(metas_direct[0], list):
                    metas_direct = metas_direct[0]
                # Prefer exact match; else accept contains
                for i, doc in enumerate(docs_direct):
                    if not isinstance(doc, str):
                        continue
                    if doc.strip() == query_lower or (query_lower in doc):
                        meta = metas_direct[i] if i < len(metas_direct) else {}
                        ans = meta['answer'] if isinstance(meta, dict) and 'answer' in meta else ''
                        # Exact match gets perfect score; contains gets neutral mid score
                        if doc.strip() == query_lower:
                            sim_val = 1.0
                        else:
                            sim_val = 0.0
                        dist_val = (1.0 - sim_val) / 2.0
                        best_match = {
                            'question': doc,
                            'answer': ans,
                            'similarity_-1_1': sim_val,
                            'distance_0_1': dist_val
                        }

                        # Start top_matches with the direct best match
                        top_matches = [best_match]

                        # Fill remaining slots up to TOP_K_MATCHES with next-best by similarity across full collection
                        try:
                            all_data_fill = collection.get(include=["embeddings", "documents", "metadatas"])
                            docs_fill = all_data_fill.get('documents') or []
                            metas_fill = all_data_fill.get('metadatas') or []
                            embs_fill = all_data_fill.get('embeddings') or []

                            # Flatten potential nesting from Chroma
                            if docs_fill and isinstance(docs_fill[0], list):
                                docs_fill = docs_fill[0]
                            if metas_fill and isinstance(metas_fill[0], list):
                                metas_fill = metas_fill[0]
                            if embs_fill and isinstance(embs_fill[0], list) and isinstance(embs_fill[0][0], list):
                                embs_fill = embs_fill[0]

                            # If embeddings are missing, compute them on the fly
                            have_stored_embs = bool(embs_fill) and isinstance(embs_fill, list)
                            if not have_stored_embs or (len(embs_fill) != len(docs_fill)):
                                try:
                                    doc_embs_fill = model.encode(docs_fill, batch_size=64, convert_to_numpy=True, normalize_embeddings=False)
                                except Exception:
                                    doc_embs_fill = []
                            else:
                                doc_embs_fill = embs_fill

                            q_vec = np.array(query_embedding, dtype=np.float32)
                            q_norm = np.linalg.norm(q_vec) + 1e-8
                            scored_fill = []
                            limit_len = min(len(docs_fill), len(metas_fill), len(doc_embs_fill))
                            for j in range(limit_len):
                                doc_j = docs_fill[j]
                                meta_j = metas_fill[j] if j < len(metas_fill) else {}
                                emb_source = doc_embs_fill[j]
                                emb_j = np.array(emb_source, dtype=np.float32)
                                if not isinstance(doc_j, str) or not isinstance(meta_j, dict) or 'answer' not in meta_j:
                                    continue
                                if doc_j == best_match['question']:
                                    continue
                                sim_j = float(np.dot(q_vec, emb_j) / (q_norm * (np.linalg.norm(emb_j) + 1e-8)))
                                dist01_j = (1.0 - sim_j) / 2.0
                                scored_fill.append({
                                    'question': doc_j,
                                    'answer': meta_j['answer'],
                                    'similarity_-1_1': sim_j,
                                    'distance_0_1': dist01_j
                                })
                            scored_fill.sort(key=lambda x: x['similarity_-1_1'], reverse=True)
                            needed = max(0, TOP_K_MATCHES - len(top_matches))
                            top_matches.extend(scored_fill[:needed])
                        except Exception:
                            # If filling fails, we still return the direct best match
                            pass

                        break
            except Exception:
                pass

        # As a final fallback, run a full-scan similarity over all embeddings (small dataset)
        if not best_match:
            try:
                all_data = collection.get(include=["embeddings", "documents", "metadatas"])
                docs_all = all_data.get('documents') or []
                metas_all = all_data.get('metadatas') or []
                embs_all = all_data.get('embeddings') or []

                # Flatten in case Chroma returns nested lists
                if docs_all and isinstance(docs_all[0], list):
                    docs_all = docs_all[0]
                if metas_all and isinstance(metas_all[0], list):
                    metas_all = metas_all[0]
                if embs_all and isinstance(embs_all[0], list) and isinstance(embs_all[0][0], list):
                    embs_all = embs_all[0]

                # Compute cosine similarity against all
                query_vec = np.array(query_embedding, dtype=np.float32)
                q_norm = np.linalg.norm(query_vec) + 1e-8
                scored = []
                for i in range(min(len(docs_all), len(metas_all), len(embs_all))):
                    doc_i = docs_all[i]
                    meta_i = metas_all[i] if i < len(metas_all) else {}
                    emb_i = np.array(embs_all[i], dtype=np.float32)
                    if not isinstance(doc_i, str) or not isinstance(meta_i, dict) or 'answer' not in meta_i:
                        continue
                    sim = float(np.dot(query_vec, emb_i) / (q_norm * (np.linalg.norm(emb_i) + 1e-8)))
                    # Convert to distance 0..1
                    dist01 = (1.0 - sim) / 2.0
                    scored.append({
                        'question': doc_i,
                        'answer': meta_i['answer'],
                        'similarity_-1_1': sim,
                        'distance_0_1': dist01
                    })

                # Sort by similarity desc
                scored.sort(key=lambda x: x['similarity_-1_1'], reverse=True)
                

                if scored:
                    best_match = scored[0]
                    top_matches = scored[:TOP_K_MATCHES]
                    # Log all entries with similarity for full collection
                    
            except Exception:
                pass

        # Omit full collection debug dump
        
        
        # Final guarantee: always return up to TOP_K_MATCHES by filling from full collection if needed
        try:
            if len(top_matches) < TOP_K_MATCHES:
                all_data_final = collection.get(include=["embeddings", "documents", "metadatas"])
                docs_final = all_data_final.get('documents') or []
                metas_final = all_data_final.get('metadatas') or []
                embs_final = all_data_final.get('embeddings') or []

                # Flatten possible nesting
                if docs_final and isinstance(docs_final[0], list):
                    docs_final = docs_final[0]
                if metas_final and isinstance(metas_final[0], list):
                    metas_final = metas_final[0]
                if embs_final and isinstance(embs_final[0], list) and isinstance(embs_final[0][0], list):
                    embs_final = embs_final[0]

                existing_questions = set([tm.get('question') for tm in top_matches if isinstance(tm, dict)])

                # If embeddings are missing, compute them on the fly
                have_stored_embs_f = bool(embs_final) and isinstance(embs_final, list)
                if not have_stored_embs_f or (len(embs_final) != len(docs_final)):
                    try:
                        doc_embs_final = model.encode(docs_final, batch_size=64, convert_to_numpy=True, normalize_embeddings=False)
                    except Exception:
                        doc_embs_final = []
                else:
                    doc_embs_final = embs_final

                q_vec_f = np.array(query_embedding, dtype=np.float32)
                q_norm_f = np.linalg.norm(q_vec_f) + 1e-8
                scored_final = []
                limit = min(len(docs_final), len(metas_final), len(doc_embs_final))
                for i in range(limit):
                    doc_i = docs_final[i]
                    meta_i = metas_final[i] if i < len(metas_final) else {}
                    emb_src = doc_embs_final[i]
                    emb_i = np.array(emb_src, dtype=np.float32)
                    if not isinstance(doc_i, str) or not isinstance(meta_i, dict) or 'answer' not in meta_i:
                        continue
                    if doc_i in existing_questions:
                        continue
                    sim_i = float(np.dot(q_vec_f, emb_i) / (q_norm_f * (np.linalg.norm(emb_i) + 1e-8)))
                    dist01_i = (1.0 - sim_i) / 2.0
                    scored_final.append({
                        'question': doc_i,
                        'answer': meta_i['answer'],
                        'similarity_-1_1': sim_i,
                        'distance_0_1': dist01_i
                    })

                scored_final.sort(key=lambda x: x['similarity_-1_1'], reverse=True)
                need_more = max(0, TOP_K_MATCHES - len(top_matches))
                if need_more:
                    top_matches.extend(scored_final[:need_more])
        except Exception:
            pass

        return best_match, top_matches
        
    except Exception as e:
        # Handle any errors during search
        logger.exception(f"Error during similarity search: {e}")
        return None, []

@app.route('/')
def index():
    """
    Main chatbot page route
    
    Returns:
        Rendered HTML template for the chatbot interface
    """
    # Initialize current dataset if not set
    initialize_current_dataset()
    
    # Check if vector database exists and has data
    
    collection = get_collection() if current_dataset else None
    
    # Count total entries in the database
    total_entries = 0
    if collection:
        try:
            # Get count of stored entries
            all_results = collection.get()
            total_entries = len(all_results['ids']) if all_results['ids'] else 0
        except Exception:
            total_entries = 0
    
    # Get available datasets
    available_datasets = list_available_datasets()
    
    # Render the chatbot template with database status and dataset info
    
    return render_template('chatbot.html', 
                         db_available=(collection is not None), 
                         total_entries=total_entries,
                         current_dataset=current_dataset,
                         available_datasets=available_datasets)

@app.route('/switch_dataset', methods=['POST'])
def switch_dataset():
    """
    Switch to a different dataset
    
    Returns:
        JSON response indicating success/failure
    """
    global current_dataset
    
    try:
        # Get dataset name from AJAX request
        data = request.get_json()
        dataset_name = data.get('dataset_name', '').strip()
        
        # Validate dataset name
        if not dataset_name:
            return jsonify({'success': False, 'message': 'Please select a dataset.'})
        
        # Validate dataset name format (alphanumeric and underscores only)
        if not dataset_name.replace('_', '').isalnum():
            return jsonify({'success': False, 'message': 'Invalid dataset name format.'})
        
        # Check if dataset exists
        available_datasets = list_available_datasets()
        if dataset_name not in available_datasets:
            return jsonify({'success': False, 'message': f'Dataset "{dataset_name}" not found.'})
        
        # Switch to the new dataset
        current_dataset = dataset_name
        
        # Test connection to new dataset
        collection = get_collection(current_dataset)
        if not collection:
            return jsonify({'success': False, 'message': f'Unable to connect to dataset "{dataset_name}".'})
        
        # Get entry count for new dataset
        try:
            all_results = collection.get()
            total_entries = len(all_results['ids']) if all_results['ids'] else 0
        except Exception:
            total_entries = 0
        
        logger.info(f"Switched to dataset: {current_dataset}")
        return jsonify({
            'success': True, 
            'message': f'Switched to dataset: {current_dataset}',
            'current_dataset': current_dataset,
            'total_entries': total_entries
        })
        
    except Exception as e:
        logger.error(f"Error switching dataset: {e}")
        return jsonify({'success': False, 'message': f'Error switching dataset: {str(e)}'})

@app.route('/refresh_datasets', methods=['POST'])
def refresh_datasets():
    """
    Refresh the list of available datasets
    
    Returns:
        JSON response with updated dataset list
    """
    global current_dataset
    
    try:
        # Get fresh list of available datasets
        available_datasets = list_available_datasets()
        
        # If current dataset is None but datasets exist, switch to first one
        if current_dataset is None and available_datasets:
            current_dataset = available_datasets[0]
            logger.info(f"Auto-switched to first available dataset: {current_dataset}")
        
        # If current dataset no longer exists, update it
        elif current_dataset and current_dataset not in available_datasets:
            if available_datasets:
                current_dataset = available_datasets[0]
                logger.info(f"Current dataset no longer exists, switched to: {current_dataset}")
            else:
                current_dataset = None
                logger.info("No datasets available, set current to None")
        
        # If no datasets exist at all, ensure current_dataset is None
        elif not available_datasets:
            current_dataset = None
            logger.info("No datasets available, set current to None")
        
        return jsonify({
            'success': True,
            'available_datasets': available_datasets,
            'current_dataset': current_dataset,
            'message': 'Datasets refreshed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error refreshing datasets: {e}")
        return jsonify({'success': False, 'message': f'Error refreshing datasets: {str(e)}'})


@app.route('/search', methods=['POST'])
def search():
    """
    Handle search requests from the chatbot interface
    
    Returns:
        JSON response with search results or error message
    """
    try:
    # Reduced debug output
        # Get the user's question from the request
        data = request.get_json()
        user_question_raw = data.get('question', '')
        user_question = normalize_text(user_question_raw)
        
        # Validate that a question was provided
        if not user_question:
            return jsonify({
                'success': False,
                'message': 'Please enter a question.'
            })
        
        # Get the vector database collection
        collection = get_collection()
        
        # Check if database is available
        if not collection:
            return jsonify({
                'success': False,
                'message': 'Vector database is not available. Please make sure the QA Manager app has been used to create some entries first.'
            })
        
        # Perform similarity search
        best_match, top_matches = search_similar_questions(user_question, collection)

        # Extra safety: ensure we always have up to TOP_K_MATCHES by filling here if earlier steps under-filled
        try:
            if len(top_matches) < TOP_K_MATCHES:
                all_data_resp = collection.get(include=["embeddings", "documents", "metadatas"])
                docs_resp = all_data_resp.get('documents') or []
                metas_resp = all_data_resp.get('metadatas') or []
                embs_resp = all_data_resp.get('embeddings') or []
                # Flatten nesting
                if docs_resp and isinstance(docs_resp[0], list):
                    docs_resp = docs_resp[0]
                if metas_resp and isinstance(metas_resp[0], list):
                    metas_resp = metas_resp[0]
                if embs_resp and isinstance(embs_resp[0], list) and isinstance(embs_resp[0][0], list):
                    embs_resp = embs_resp[0]

                # Compute embeddings on-the-fly if missing
                have_embs_resp = bool(embs_resp) and isinstance(embs_resp, list)
                if not have_embs_resp or (len(embs_resp) != len(docs_resp)):
                    try:
                        doc_embs_resp = model.encode(docs_resp, batch_size=64, convert_to_numpy=True, normalize_embeddings=False)
                    except Exception:
                        doc_embs_resp = []
                else:
                    doc_embs_resp = embs_resp

                existing_qs = set([tm.get('question') for tm in top_matches if isinstance(tm, dict)])
                q_vec_resp = model.encode(user_question).astype(np.float32)
                q_norm_resp = np.linalg.norm(q_vec_resp) + 1e-8
                scored_resp = []
                limit_resp = min(len(docs_resp), len(metas_resp), len(doc_embs_resp))
                for i in range(limit_resp):
                    doc_i = docs_resp[i]
                    meta_i = metas_resp[i] if i < len(metas_resp) else {}
                    emb_src = doc_embs_resp[i]
                    emb_i = np.array(emb_src, dtype=np.float32)
                    if not isinstance(doc_i, str) or not isinstance(meta_i, dict) or 'answer' not in meta_i:
                        continue
                    if doc_i in existing_qs:
                        continue
                    sim_i = float(np.dot(q_vec_resp, emb_i) / (q_norm_resp * (np.linalg.norm(emb_i) + 1e-8)))
                    dist01_i = (1.0 - sim_i) / 2.0
                    scored_resp.append({
                        'question': doc_i,
                        'answer': meta_i['answer'],
                        'similarity_-1_1': sim_i,
                        'distance_0_1': dist01_i
                    })
                scored_resp.sort(key=lambda x: x['similarity_-1_1'], reverse=True)
                need_extra = max(0, TOP_K_MATCHES - len(top_matches))
                if need_extra:
                    top_matches.extend(scored_resp[:need_extra])
        except Exception:
            pass
        
        # Check if best match meets the similarity threshold
        has_relevant_answer = bool(best_match) and float(best_match.get('similarity_-1_1', -2.0)) >= SIMILARITY_THRESHOLD
        db_answer = best_match['answer'] if has_relevant_answer else None
        
        # Get LLM response
        llm_answer = call_llm_for_answer(user_question, db_answer, has_relevant_answer)
        
        # Determine if we have a relevant answer above threshold
        has_relevant_best_match = bool(best_match) and float(best_match.get('similarity_-1_1', -2.0)) >= SIMILARITY_THRESHOLD
        
        # Always prepare top matches for display (regardless of threshold)
        top_matches_for_display = [
            {
                'question': match['question'],
                'answer': match['answer'],
                'similarity': round(float(match.get('similarity_-1_1', -2.0)), 3),
                'distance': round(float(match.get('distance_0_1', float('inf'))), 3)
            }
            for match in top_matches  # top_matches already contains up to TOP_K_MATCHES results
        ]
        
        if not has_relevant_best_match:
            # No relevant matches found above threshold, but show top matches anyway
            logger.info(f"No matches above threshold {SIMILARITY_THRESHOLD} for question: '{user_question}'")
            
            return jsonify({
                'success': True,
                'has_results': False,  # No results above threshold
                'show_top_matches': True,  # But show top matches anyway
                'message': "Sorry, I don't have information on that. Please try a different question.",
                'user_question': user_question,
                'llm_answer': llm_answer,  # LLM response
                'top_matches': top_matches_for_display,  # Always show top matches
                'threshold_used': SIMILARITY_THRESHOLD
            })
        else:
            # Relevant matches found above threshold
            logger.info(f"Best match similarity={round(float(best_match.get('similarity_-1_1', -2.0)), 3)} for question: '{user_question}'")
            return jsonify({
                'success': True,
                'has_results': True,
                'user_question': user_question,
                'llm_answer': llm_answer,  # LLM response
                'best_answer': best_match['answer'],
                'best_similarity': round(float(best_match.get('similarity_-1_1', -2.0)), 3),
                'top_matches': top_matches_for_display,  # Always show top matches
                'threshold_used': SIMILARITY_THRESHOLD
            })
        
    except Exception as e:
        # Handle any errors during processing
        logger.exception(f"Error in search endpoint: {e}")
        return jsonify({
            'success': False,
            'message': f'An error occurred while processing your question: {str(e)}'
        })

@app.route('/status')
def status():
    """
    Get database status information
    Refreshes database connection to get current entry count
    
    Returns:
        JSON response with database status and statistics
    """
    try:
        # Reduced debug output
        # Get fresh collection connection to see latest entries
        collection = get_collection()
        
        if not collection:
            return jsonify({
                'available': False,
                'total_entries': 0,
                'message': 'Vector database not found'
            })
        
        # Get current database statistics (refreshed)
        all_results = collection.get()
        total_entries = len(all_results['ids']) if all_results['ids'] else 0
        
        return jsonify({
            'available': True,
            'total_entries': total_entries,
            'threshold': SIMILARITY_THRESHOLD,
            'top_k': TOP_K_MATCHES,
            'model_name': 'all-MiniLM-L6-v2',
            'last_updated': 'Real-time (refreshed on each request)'
        })
        
    except Exception as e:
        logger.exception(f"Error in /status: {e}")
        return jsonify({
            'available': False,
            'total_entries': 0,
            'message': f'Error accessing database: {str(e)}'
        })

@app.route('/refresh', methods=['POST'])
def refresh():
    """
    Manually refresh the database connection and return current stats
    Forces a complete re-connection to ensure latest entries are indexed
    """
    try:
        # Reduced debug output
        
        # Get a fresh collection connection (simple refresh)
        collection = get_collection()
        
        if not collection:
            return jsonify({
                'available': False,
                'total_entries': 0,
                'message': 'Vector database not found'
            })

        all_results = collection.get()
        total_entries = len(all_results['ids']) if all_results['ids'] else 0

        return jsonify({
            'available': True,
            'total_entries': total_entries,
            'message': 'Database refreshed - automatic fallback ensures all entries are searchable'
        })
    except Exception as e:
        logger.exception(f"Error in /refresh: {e}")
        return jsonify({
            'available': False,
            'total_entries': 0,
            'message': f'Error refreshing database: {str(e)}'
        })

# Run the Flask chatbot application
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)