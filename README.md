# Intelligent Nutrition FAQ System (Nutrition Challenge Project)

## Overview

This AI nutrition FAQ system is a solution created for a challenge, based on a vector database-powered question-answering system specifically designed for nutrition and health information.
This solution combines a chatbot RAG approach and a vector db management system.
It is composed by 2 apps: a vector db management app and a chatbot which answers nutrition question retrieveing the answer-hint from the vector db.
The chatbot can be integrated in a patient facing mobile App or as it is, or after an expansion as a multi-turn chatbot, with personal (of the patient) memory, maybe integrated in an AI-agent system that retrives also different type of information, from other sources, to answer different questions from the user.

### Key Features

- **Semantic Search**: it uses vector embeddings to understand question meaning
- **Multi-Dataset Support**: Manage separate knowledge bases for different domains
- **LLM Integration**: Leverages state-of-the-art language models for conversational responses
- **Dual Interface**: Database Administration interface for data management and user-facing chatbot
- **Real-time Updates**: Immediate availability of newly added content
- **Fallback Mechanisms**: Robust error handling and automatic recovery systems

## System Architecture

### Core Components

1. **Vector Database Manager (`src/vectordb_app.py`)** - Port 5000
   - Administrative interface for dataset management
   - Data upload and CRUD operations
   - Real-time database monitoring

2. **Chatbot Interface (`src/chatbot_app.py`)** - Port 5001
   - User-facing FAQ interface
   - Semantic search and LLM response generation
   - Dataset switching and status monitoring

3. **Vector Database (`./vector_db/`)**
   - ChromaDB persistent storage
   - HNSW indexing for fast similarity search
   - Cosine similarity space for semantic matching

### Technology Stack

- **Web Framework**: Flask 3.1.1
- **Vector Database**: ChromaDB 1.0.16
- **Embedding Model**: all-MiniLM-L6-v2 (SentenceTransformers)
- **LLM Integration**: LiteLLM with OpenRouter API
- **Frontend**: Bootstrap 5.1.3 + Vanilla JavaScript
- **Storage**: File-based persistent vector storage

## Project Structure

The project follows a clean, organized structure that separates source code, data, and generated artifacts:

```
NutriCh_Project/
├── README.md                     # Project documentation
├── license.txt                   # License (Apache 2.0)
├── requirements.txt              # Python dependencies
├── run_apps.py                   # Launcher for both apps (5000/5001)
├── data/                         # Sample datasets
│   ├── nutrition_faqs.json       # Example nutrition FAQs
│   └── second_data.json          # Additional sample data for testing purposes only
├── src/                          # Source code
│   ├── chatbot_app.py            # Chatbot (port 5001)
│   ├── vectordb_app.py           # Vector DB manager (port 5000)
│   ├── templates/                # HTML templates
│   │   ├── chatbot.html          # Chatbot UI
│   │   └── index.html            # DB manager UI
│   └── utils/                    # Shared utilities
│       ├── __init__.py			  # init file
│       └── text_normalization.py # Funtion that normalize text for consistent storage and retrieval in RAG
├── vector_db/                    # ChromaDB persistent storage (it will be generated and it will be populated)
└── temp_uploads/                 # Temporary uploads for JSON import (it will be generated)
```

### Directory Organization

- **`src/`**: Contains all source code and templates, following Python best practices
- **`data/`**: Sample datasets and examples for testing and initial setup
- **`vector_db/`**: Persistent vector database storage (auto-generated)
- **Root level**: Configuration files and project management utilities

## Technical Decisions & Rationale

### Embedding Model Choice: all-MiniLM-L6-v2

**Why this model:**
it is a luight-weight model that has a good balance between accuracy and speed.
It has a size (80MB) that allows us to store on a local system without problems.
The vector has 384 dimentions which are sufficient for the task required.

**Trade-offs:**
It is a small model, so if we want more accuracy we need a bigger model but for this task the one used is sufficient.
It is good enogh for this small project. For production we need to use a better model.

The text gets normalized before stored and/or analyzed in order to ensure consistency during the RAG operations.

### Similarity Threshold: 0.75

This threshold is on a distance based on a cosine similarity. I used a standard cosine similarity (normalized by vector norms), not raw dot products. The similarity is then linearly mapped into a [0,1] "distance" scale for easier interpretation, rather than using Chroma’s native [0,2]  cosine distance scale which can be confusing to people not technically expert on the matter.
The normalization applied ensures that the cosine similarity we receive measures only the angle between vectors (direction), ignoring their magnitude. 
The vector magnitude might reflect irrelevant factors (for example word frequency or vector scaling) and in this task (but also in many others in RAG systems) we only want to assess how similar the content or meaning is, represented by vector direction.
In different tasks, if the magnitude itself carries meaningful information (example: importance, magnitude of a feature), ignoring it, by normalizing, could lose potentially useful data. In this latter case, using raw dot products or other similarity measures could be better.

**Rationale:**
The threshold value comes from testing. Multiple tests have been made and this threshold has been found high enough to remove not related questions but not too high to miss relevant information.
In the tests I noticed that the most relevant questions had a similarity >=0.75 so I selected the above-mentioned threshold. Some not really-relevant questions got 0.715 or similar scores and this made me put the threshold at 0.75.

**Calibration Process:**
- Values > 0.75: High confidence matches (excellent semantic similarity)
- Values 0.5-0.75: Medium confidence (potentially relevant)
- Values < 0.5: Low relevance

But anyway the threshold can be easily changed in the .env file if using the tool we see that the threshold is too high or too low.

### Vector Database: ChromaDB

**Selection Criteria:**
- **Ease of Use**: Simple Python API with minimal configuration
- **Persistence**: Built-in persistent storage without external dependencies
- **Performance**: HNSW indexing (Hierarchical Navigable Small World, a graph-based algorithm designed for efficient approximate nearest neighbor (ANN) searches in high-dimensional vector spaces) for sub-linear search complexity
- **Flexibility**: Support for metadata and filtering

**Alternatives Considered:**
- **Pinecone**: Excellent but requires cloud dependency and subscription
- **Weaviate**: More complex setup, overkill for this use case
- **FAISS**: Lower-level, requires more infrastructure code
- **vectordb2**: another simple solution in python

### LLM Integration: llama-3.3-70b-instruct as selected model, via OpenRouter

**Why these models:**
- It is a fairly good model eve if not advanced but at the same time it is not an overshoot for this task. It is more than sufficient for creafting an answer given the hint received from the RAG system.
- Llama model is slightly more empathic than many alternatives, I preferred its answers during the tests.
- Llama-3.3-70b-instruct is also usually (costs can change) much cheaper than other more famous LLMs.

**Why OpenRouter:**
- **Flexibility**: Easy model switching without code changes
- **Cost Control**: Competitive pricing and usage monitoring
- **Reliability**: Redundant infrastructure and fallback options

## Installation & Setup

### Prerequisites

- Python 3.13.5 (I used this but probably it works well also with previous versions)
- 4GB RAM minimum (probably it works also with lower quantity or RAM)
- 1GB free disk space (same as above)

### Quick Start

1. **Install**
   I provide the project as a zip file. Unzip and place it in your computer.

2. **Configure Environment**
   - Copy the example env file to `.env`:
     - Windows PowerShell:
       ```powershell
       Copy-Item -Path .env.example -Destination .env
       ```
     - macOS/Linux:
       ```bash
       cp .env.example .env
       ```
   - Open `.env` and set values:
     - `FLASK_SECRET_KEY`: long random string (required)
     - `OPENROUTER_API_KEY`: your key (optional; without it, LLM answers are disabled and a fallback is used)
     - `LLM_MODEL`: optional model id (default provided)
     - `APP_NAME`: optional app display name
     - `OPENROUTER_REFERER`: optional referer URL
     - `SIMILARITY_THRESHOLD`: optional float in [0,1] (default 0.75)
3. **Launch Applications**
   in the project folder:
   ```bash
   python run_apps.py
   ```
   Choose option 3 to run both applications. You can run also one of the two apps alone using the other options.

4. **Access Interfaces**
   - Database Manager: http://localhost:5000
   - Chatbot: http://localhost:5001

The first time it runs it will create a new chromadb and will download the all-MiniLM-L6-v2 model and store in the appropriate place.
Please be patient because it can take some time for the apps to start.

## Usage Instructions

### For DB Administration

1. **Dataset Management**
   - Create new datasets for different health nutrition domains (it works also for other non-nutrition domains but in order to have a smooth experience we must change the prompts)
   - Upload JSON files with question-answer pairs
   - Manually add individual entries
   - Monitor database statistics

2. **Data Format**
   See sample files in `data/` folder for the expected JSON structure:
   ```json
   [
     {
       "questions": [
         "What are good snacks for diabetics?",
         "Healthy snack options for diabetes?"
       ],
       "answer": "Greek yogurt, almonds, and vegetables with hummus are excellent choices."
     }
   ]
   ```

### For Chatbot usage

1. **Ask Questions**
   - Type natural language questions about nutrition (or other topics to test the system)
   - Use conversational language (not keywords)

2. **Interpret Responses**
   - **AI Answer**: Contextual response from LLM
   - **Database Response**: Direct match from knowledge base
   - **Related Questions**: Similar questions for exploration

3. **Switch Datasets**
   - Change between nutrition, fitness, or other health topics
   - Each dataset contains specialized knowledge

## Integration Possibilities

This system can be integrated in: 
- healthcare apps (web or mobile), 
- other patient/doctor-facing medical platforms, 
- corporate wellness programs
- and many more...

### Performance Trade-offs

The main trade-offs come from teh fact that I used light-weight solutions in order to have a non-heavy system on my personal (local) laptop.
Using bigger models or more performing vector dbs will improve the performances.

### Scalability Assumptions

1. **Dataset Size**: Optimized for 1,000-50,000 question-answer pairs per dataset but I think it can go further
2. **Concurrent Users**: It must be deployed in order to serve multiple users at the same time.
3. **Response Time**: Target <3 seconds for 95% of queries

### Content Assumptions

1. **Language**: Primarily English content and queries
2. **Domain**: Focused on nutrition and health topics
3. **Expertise Level**: Consumer-level health information (not clinical)
4. **Currency**: Static knowledge base, requiring manual updates

### Security Considerations

1. **API Keys**: Stored in environment variables (not production-ready, but for the challenge it is sufficient)
2. **Input Validation**: Basic sanitization implemented, but much more can be done. It requires time.
3. **Rate Limiting**: Not implemented (should be added for production)

## Expansion Opportunities

- the system needs to be modified in order to deal with different users.
- personalization with personal memory (for example for dietary restrictions and alergies).
- advanced search: using a more complex search mechanism will improve the results: hybrid search...
- content management: fact-checking against medical databases (maybe at the insertion of new data)

### Possible Technical Improvements

1. **Performance Optimization**
   - GPU acceleration for embeddings (but it can be an overshoot)
   - Caching layer for frequent queries
   - Database sharding for large datasets

2. **ML Pipeline**
   - Retraining on user feedback
   - Testing for different models
   - Real-time performance monitoring (in terms of speed and possible errors)

3. **Multi-turn Chatbot**
   - Integration of this project in a multiturn chatbot
   - Integration with an agent system that retrieves information from multiple sources
   - Integration with a multi-agent system that perform calls and operations based on user requirements

## Security Considerations for Production

### Authentication & Authorization
- Implement OAuth 2.0 for user authentication
- Role-based access control for admin functions
- API key management for programmatic access

### Data Protection
- Encrypt sensitive data at rest
- Use HTTPS for all communications
- Implement data retention policies

### Input Security
- Comprehensive input validation and sanitization
- Rate limiting to prevent abuse
- Database injection protection (though using a NoSQL solution)

## Monitoring

We can monitor this system live and offline:  

**live**  
-latency  
-cost per request  
-Availability/error rate  
-Retrieval coverage  
-% of answers with citations; % of cited text that actually supports the answer  
-Contradiction / hallucination score (using an LLM as a judge to automate the task)  
-Context utilization: fraction of retrieved tokens actually referenced (overlap of answer with retrieved chunks).  
-user feedbacks: thumbs up/down, "answer helped?", "missing doc?"; report-bad-citation.  

**offline**  
-testing the model and score its answers using ROUGE, BLUE, perplexity scores and similar.  
-using a LLM-as-a-judge to score the model.  
-using human expert revisions to validate the answers (especially if we change the model or if we insert a new dataset)  
-monitoring the scores of the similarity search.  
-monitor the data drift (change in language used, change in medical information in litterature...)  

We can use OpenTelemetry spans for each stage (it represents a single operation or a segment of the overall process tracked as a trace)  

**RAG-specific health checks**  
Index freshness: ingestion success, document counts, lag per source.  
Vector store health: query QPS, latency, memory/disk usage...  
Embedding cache hit rate  

We can build a third app that monitors the other two, live and offline and provides a dashboard for visual feedbacks on the monitoring.  

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

---

**Disclaimer**: This system is designed for a challenge. It should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.















