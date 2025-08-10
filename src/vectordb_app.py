# Import required libraries for Flask web application
import chromadb  # Vector database for storing embeddings
import json  # For handling JSON file operations
import logging  # For debug logging
import os  # For file system operations
import uuid  # For generating unique IDs for database entries

from dotenv import load_dotenv  # For loading .env file
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from sentence_transformers import SentenceTransformer  # For generating text embeddings
from werkzeug.utils import secure_filename  # For secure file upload handling
from utils import normalize_text  # Shared text normalization


# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)
# Set secret key for session management and flash messages
app.secret_key = os.environ["FLASK_SECRET_KEY"]

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads'  # Directory to store uploaded files temporarily
ALLOWED_EXTENSIONS = {'json'}  # Only allow JSON file uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize ChromaDB client for vector database operations
chroma_client = chromadb.PersistentClient(path="./vector_db")

# Global variables for current dataset
current_dataset = None  # No default dataset
collection = None

def get_or_create_collection(dataset_name):
    """
    Get or create a ChromaDB collection for a specific dataset
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'nutrition', 'fitness', 'health')
    
    Returns:
        Collection: ChromaDB collection object
    """
    collection_name = f"qa_collection_{dataset_name}"
    # Reduced debug output
    
    collection = chroma_client.get_or_create_collection(
        name=collection_name, 
        metadata={"hnsw:space": "cosine", "dataset": dataset_name}
    )
    
    try:
        # Prefer count() if available for quick stats
        current_count = collection.count() if hasattr(collection, 'count') else len((collection.get() or {}).get('ids', []))
        logger.info(f"Collection '{collection_name}' ready. Current count: {current_count}")
    except Exception as e:
        logger.warning(f"Unable to read collection count for {collection_name}: {e}")
    
    return collection

def list_available_datasets():
    """
    List all available datasets by scanning ChromaDB collections
    
    Returns:
        list: List of dataset names (empty list if none exist)
    """
    try:
        collections = chroma_client.list_collections()
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

# Initialize with first available dataset (if any)
available_datasets = list_available_datasets()
if available_datasets:
    current_dataset = available_datasets[0]
    collection = get_or_create_collection(current_dataset)
    logger.info(f"Initialized with existing dataset: {current_dataset}")
else:
    current_dataset = None
    collection = None
    logger.info("No datasets found - initialized with no active dataset")

# Initialize sentence transformer model for generating embeddings
# Using a lightweight but effective model for text similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

def force_collection_reindex():
    """
    Simple persistence to help ChromaDB indexing (legacy function - no longer critical)
    The chatbot now uses automatic fallback, so this is just a basic cleanup
    """
    try:
        # Simple persistence attempt
        if hasattr(chroma_client, 'persist'):
            chroma_client.persist()
        return True
    except Exception:
        return False

def allowed_file(filename):
    """
    Check if uploaded file has allowed extension (only JSON files)
    
    Args:
        filename (str): Name of the uploaded file
    
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_json_structure(data):
    """
    Validate that the JSON data has the correct structure
    Expected format: [{"questions": [...], "answer": "..."}, ...]
    
    Args:
        data: Parsed JSON data to validate
    
    Returns:
        bool: True if structure is valid, False otherwise
    """
    # Check if data is a list
    if not isinstance(data, list):
        logger.warning("JSON data is not a list")
        return False
    
    # Check each item in the list
    for i, item in enumerate(data):
        # Check if item is a dictionary
        if not isinstance(item, dict):
            logger.warning(f"Item {i} is not a dictionary")
            return False
        
        # Check if required keys exist
        if 'questions' not in item or 'answer' not in item:
            logger.warning(f"Item {i} missing required keys 'questions' or 'answer'")
            return False
        
        # Check if questions is a list
        if not isinstance(item['questions'], list):
            logger.warning(f"Item {i} 'questions' is not a list")
            return False
        
        # Check if answer is a string
        if not isinstance(item['answer'], str):
            logger.warning(f"Item {i} 'answer' is not a string")
            return False
        
        # Check if questions list is not empty
        if len(item['questions']) == 0:
            logger.warning(f"Item {i} has empty questions list")
            return False
        
        # Check if all questions are strings
        for j, question in enumerate(item['questions']):
            if not isinstance(question, str):
                logger.warning(f"Item {i}, question {j} is not a string")
                return False
    
    return True

def process_and_store_data(data):
    """
    Process JSON data by normalizing text and storing in vector database
    
    Args:
        data: Validated JSON data containing questions and answers
    
    Returns:
        int: Number of question-answer pairs processed
    """
    processed_count = 0  # Counter for processed entries
    # For observability, track count before
    try:
        before_count = collection.count() if hasattr(collection, 'count') else len((collection.get() or {}).get('ids', []))
    except Exception:
        before_count = None
    
    # Process each item in the data
    for item in data:
        # Normalize answer for consistency
        normalized_answer = normalize_text(item['answer'])
        
        # Process each question in the questions list
        for question in item['questions']:
            # Normalize question text
            normalized_question = normalize_text(question)
            
            # Generate embedding for the question using sentence transformer
            embedding = model.encode(normalized_question).tolist()
            
            # Generate unique ID for this entry
            entry_id = str(uuid.uuid4())
            
            # Store in ChromaDB collection
            # The question embedding is stored with answer as metadata
            collection.add(
                embeddings=[embedding],  # Question embedding
                documents=[normalized_question],  # Normalized question text
                metadatas=[{"answer": normalized_answer}],  # Normalized answer
                ids=[entry_id]  # Unique identifier
            )
            # Reduced debug output
            
            processed_count += 1  # Increment counter
    
        logger.info(f"Bulk upload completed: {processed_count} entries stored")
        # Reduced info output
        
        # For observability, track count after
        try:
            after_count = collection.count() if hasattr(collection, 'count') else len((collection.get() or {}).get('ids', []))
            logger.info(f"Collection count: {before_count} -> {after_count} (delta: +{processed_count})")
        except Exception:
            pass
    
    return processed_count

def get_all_entries():
    """
    Retrieve all entries from the current collection for display
    
    Returns:
        list: List of dictionaries containing entry data with IDs
    """
    if collection is None:
        return []
    
    try:
        # Get all data from the collection
        results = collection.get(include=['documents', 'metadatas'])
        
        entries = []
        if results and 'documents' in results and 'metadatas' in results:
            docs = results['documents']
            metas = results['metadatas']
            ids = results.get('ids', [])
            
            # Combine documents and metadata
            for i in range(len(docs)):
                if i < len(metas) and isinstance(metas[i], dict) and 'answer' in metas[i]:
                    entry = {
                        'id': ids[i] if i < len(ids) else f"unknown_{i}",
                        'question': docs[i],
                        'answer': metas[i]['answer']
                    }
                    entries.append(entry)
        
        return entries
    except Exception as e:
        logger.error(f"Error retrieving entries: {e}")
        return []

@app.route('/')
def index():
    """
    Main page displaying dataset management and current entries
    
    Returns:
        Rendered HTML template with current data
    """
    # Get current entries
    entries = get_all_entries()
    
    # Get available datasets
    available_datasets = list_available_datasets()
    
    return render_template('index.html', 
                         entries=entries,
                         current_dataset=current_dataset,
                         available_datasets=available_datasets)

@app.route('/switch_dataset', methods=['POST'])
def switch_dataset():
    """
    Switch to a different dataset
    
    Returns:
        Redirect to main page with success/error message
    """
    global current_dataset, collection
    
    # Get dataset name from form
    dataset_name = request.form.get('dataset_name', '').strip()
    
    # Validate dataset name
    if not dataset_name:
        flash("Please select a dataset.", 'error')
        return redirect(url_for('index'))
    
    # Validate dataset name format (alphanumeric and underscores only)
    if not dataset_name.replace('_', '').isalnum():
        flash("Invalid dataset name format.", 'error')
        return redirect(url_for('index'))
    
    # Check if dataset exists
    available_datasets = list_available_datasets()
    if dataset_name not in available_datasets:
        flash(f"Dataset '{dataset_name}' not found.", 'error')
        return redirect(url_for('index'))
    
    try:
        # Switch to the new dataset
        current_dataset = dataset_name
        collection = get_or_create_collection(current_dataset)
        
        flash(f"Switched to dataset: {current_dataset}", 'success')
        logger.info(f"Switched to dataset: {current_dataset}")
        
    except Exception as e:
        flash(f"Error switching to dataset: {str(e)}", 'error')
        logger.error(f"Error switching dataset: {e}")
    
    return redirect(url_for('index'))

@app.route('/create_dataset', methods=['POST'])
def create_dataset():
    """
    Create a new dataset
    
    Returns:
        Redirect to main page with success/error message
    """
    global current_dataset, collection
    
    # Get new dataset name from form
    new_dataset_name = request.form.get('new_dataset_name', '').strip()
    
    # Validate dataset name
    if not new_dataset_name:
        flash("Please enter a dataset name.", 'error')
        return redirect(url_for('index'))
    
    # Validate dataset name format (alphanumeric and underscores only)
    if not new_dataset_name.replace('_', '').isalnum():
        flash("Dataset name can only contain letters, numbers, and underscores.", 'error')
        return redirect(url_for('index'))
    
    # Check if dataset already exists
    available_datasets = list_available_datasets()
    if new_dataset_name in available_datasets:
        flash(f"Dataset '{new_dataset_name}' already exists.", 'error')
        return redirect(url_for('index'))
    
    try:
        # Create the new dataset
        new_collection = get_or_create_collection(new_dataset_name)
        
        # Switch to the new dataset
        current_dataset = new_dataset_name
        collection = new_collection
        
        flash(f"Created and switched to new dataset: {current_dataset}", 'success')
        logger.info(f"Created new dataset: {current_dataset}")
        
    except Exception as e:
        flash(f"Error creating dataset: {str(e)}", 'error')
        logger.error(f"Error creating dataset: {e}")
    
    return redirect(url_for('index'))

@app.route('/delete_dataset', methods=['POST'])
def delete_dataset():
    """
    Delete a dataset
    
    Returns:
        Redirect to main page with success/error message
    """
    global current_dataset, collection
    
    # Get dataset name to delete from form
    dataset_to_delete = request.form.get('dataset_to_delete', '').strip()
    
    # Validate dataset name
    if not dataset_to_delete:
        flash("Please select a dataset to delete.", 'error')
        return redirect(url_for('index'))
    
    # Check if dataset exists
    available_datasets = list_available_datasets()
    if dataset_to_delete not in available_datasets:
        flash(f"Dataset '{dataset_to_delete}' not found.", 'error')
        return redirect(url_for('index'))
    
    try:
        # Delete the collection
        collection_name = f"qa_collection_{dataset_to_delete}"
        chroma_client.delete_collection(name=collection_name)
        
        # If we deleted the current dataset, switch to another one or set to None
        if current_dataset == dataset_to_delete:
            remaining_datasets = [d for d in available_datasets if d != dataset_to_delete]
            if remaining_datasets:
                current_dataset = remaining_datasets[0]
                collection = get_or_create_collection(current_dataset)
                flash(f"Deleted dataset '{dataset_to_delete}' and switched to '{current_dataset}'.", 'success')
            else:
                current_dataset = None
                collection = None
                flash(f"Deleted dataset '{dataset_to_delete}'. No datasets remaining.", 'success')
        else:
            flash(f"Deleted dataset '{dataset_to_delete}'.", 'success')
        
        logger.info(f"Deleted dataset: {dataset_to_delete}")
        
    except Exception as e:
        flash(f"Error deleting dataset: {str(e)}", 'error')
        logger.error(f"Error deleting dataset: {e}")
    
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and process the JSON data
    
    Returns:
        Redirect to main page with success/error message
    """
    # Check if there's an active dataset
    if current_dataset is None or collection is None:
        flash("You can't upload data without creating a dataset first.", 'error')
        return redirect(url_for('index'))
    
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file selected for upload.', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if file was actually selected
    if file.filename == '':
        flash('No file selected for upload.', 'error')
        return redirect(url_for('index'))
    
    # Check if file has allowed extension
    if not (file and allowed_file(file.filename)):
        flash('Only JSON files are allowed.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Read and parse JSON file directly from memory
        data = json.load(file.stream)
        
        # Validate JSON structure
        if not validate_json_structure(data):
            flash('Invalid JSON structure. Please check the file format.', 'error')
            return redirect(url_for('index'))
        
        # Process and store the data
        processed_count = process_and_store_data(data)
        
        # Force persistence to ensure data is immediately available
        force_collection_reindex()
        
        flash(f'Successfully processed {processed_count} question-answer pairs from file "{file.filename}".', 'success')
        # Reduced info output
        
    except json.JSONDecodeError as e:
        flash(f'Error parsing JSON file: {str(e)}', 'error')
        logger.error(f"JSON parse error: {e}")
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        logger.error(f"File processing error: {e}")
    
    return redirect(url_for('index'))

@app.route('/add_entry', methods=['POST'])
def add_entry():
    """
    Add a new question-answer pair manually
    
    Returns:
        Redirect to main page with success/error message
    """
    # Check if there's an active dataset
    if current_dataset is None or collection is None:
        flash("You can't add data without creating a dataset first.", 'error')
        return redirect(url_for('index'))
    
    # Get and normalize question and answer from form data
    question = normalize_text(request.form.get('question', ''))
    answer = normalize_text(request.form.get('answer', ''))
    
    # Validate that both fields are provided
    if not question or not answer:
        flash('Both question and answer are required.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Generate embedding for the normalized question
        embedding = model.encode(question).tolist()
        
        # Generate unique ID for this entry
        entry_id = str(uuid.uuid4())
        
        # Store in ChromaDB collection
        collection.add(
            embeddings=[embedding],
            documents=[question],
            metadatas=[{"answer": answer}],
            ids=[entry_id]
        )
        # Reduced debug output
        
        # Note: Immediate reindexing removed - chatbot uses automatic fallback for newest entries
        logger.info("Single entry addition completed")

        flash('Entry added successfully!', 'success')
        
    except Exception as e:
        flash(f'Error adding entry: {str(e)}', 'error')
        logger.error(f"Error adding entry: {e}")
    
    return redirect(url_for('index'))

@app.route('/edit_entry', methods=['POST'])
def edit_entry():
    """
    Edit an existing entry
    
    Returns:
        JSON response indicating success/failure
    """
    try:
        # Get data from AJAX request
        data = request.get_json()
        entry_id = data.get('id')
        new_question = normalize_text(data.get('question', ''))
        new_answer = normalize_text(data.get('answer', ''))
        
        # Validate input
        if not entry_id or not new_question or not new_answer:
            return jsonify({'success': False, 'message': 'All fields are required.'})
        
        # Generate new embedding from normalized question
        new_embedding = model.encode(new_question).tolist()
        
        # Update the entry in ChromaDB
        collection.update(
            ids=[entry_id],
            embeddings=[new_embedding],
            documents=[new_question],
            metadatas=[{"answer": new_answer}]
        )
        
        logger.info(f"Updated entry {entry_id}")
        return jsonify({'success': True, 'message': 'Entry updated successfully.'})
        
    except Exception as e:
        logger.error(f"Error editing entry: {e}")
        return jsonify({'success': False, 'message': f'Error updating entry: {str(e)}'})

@app.route('/delete_entry', methods=['POST'])
def delete_entry():
    """
    Delete an entry from the database
    
    Returns:
        JSON response indicating success/failure
    """
    try:
        # Get data from AJAX request
        data = request.get_json()
        entry_id = data.get('id')
        
        # Validate input
        if not entry_id:
            return jsonify({'success': False, 'message': 'Entry ID is required.'})
        
        # Delete the entry from ChromaDB
        collection.delete(ids=[entry_id])
        
        logger.info(f"Deleted entry {entry_id}")
        return jsonify({'success': True, 'message': 'Entry deleted successfully.'})
        
    except Exception as e:
        logger.error(f"Error deleting entry: {e}")
        return jsonify({'success': False, 'message': f'Error deleting entry: {str(e)}'})

@app.route('/delete_all', methods=['POST'])
def delete_all():
    """
    Delete all entries from the current dataset
    
    Returns:
        JSON response indicating success/failure
    """
    try:
        if current_dataset is None or collection is None:
            return jsonify({'success': False, 'message': 'No active dataset.'})
        
        # Get all IDs first
        all_data = collection.get()
        all_ids = all_data.get('ids', [])
        
        if all_ids:
            # Delete all entries
            collection.delete(ids=all_ids)
            logger.info(f"Deleted all {len(all_ids)} entries from dataset {current_dataset}")
            return jsonify({'success': True, 'message': f'Deleted all {len(all_ids)} entries.'})
        else:
            return jsonify({'success': True, 'message': 'No entries to delete.'})
        
    except Exception as e:
        logger.error(f"Error deleting all entries: {e}")
        return jsonify({'success': False, 'message': f'Error deleting entries: {str(e)}'})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
