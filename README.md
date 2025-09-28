RAG-based EdTech MVP — Economics Chapter

Purpose:
This repository contains a minimal, reproducible MVP that implements a Retrieval-Augmented Generation (RAG) pipeline for an economics book chapter.
It demonstrates:

Creating a searchable knowledge base from a chapter PDF.

Attaching metadata and storing embeddings in a FAISS vector database.

Student answer assessment (embedding similarity scoring).

Optional: building a knowledge graph and auto-generating lessons from YouTube videos.

A Streamlit demo app to showcase all features without crashing even if optional data is missing.

✅ Assignment Coverage

This MVP covers all the assignment tasks:

✅ Clean searchable knowledge extracted from the provided chapter PDF (page-level + chunking).

✅ Metadata fields per chunk: subject, topic (heuristic), subtopic (placeholder), difficulty (default/manual).

✅ Object store (data/*.jsonl) for clean text chunks.

✅ Embeddings + FAISS vector DB for retrieval.

✅ Knowledge graph built from keyword co-occurrence and saved as data/knowledge_graph.png.

✅ Lesson pages from videos (overview, highlights, key terms, 5 MCQs + 1 essay question) — requires running process_videos.py.

✅ Streamlit demo app (src/app_streamlit.py) with:

RAG search + answer generation

Student answer assessment

Lessons view (if transcripts exist)

Knowledge graph (if graph file exists)

📂 Repo Structure

rag-assignment/
├─ data/
│  ├─ chapter.pdf              # input (download yourself)
│  ├─ chunks.jsonl             # raw extracted text
│  ├─ emb_chunks.jsonl         # chunked text + metadata
│  ├─ emb_metadata.jsonl       # index → metadata
│  ├─ faiss_index.index        # FAISS index file
│  ├─ store.pkl                # serialized helper object
│  ├─ knowledge_graph.png      # (optional) generated graph
│  └─ video_transcripts.jsonl  # (optional) transcripts + MCQs
├─ src/
│  ├─ ingest_pdf.py
│  ├─ chunker.py
│  ├─ embed_store.py
│  ├─ build_graph.py
│  ├─ process_videos.py
│  ├─ generate_questions.py
│  ├─ assess_answer.py
│  └─ app_streamlit.py
├─ requirements.txt
├─ README.md
└─ .gitignore

⚙️ Setup Instructions (Local)

1. Clone repo & create virtual environment

git clone https://github.com/AnantKrishna1/rag-assignment.git
cd rag-assignment

python -m venv venv
# Activate:
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

2. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

3. Run preprocessing scripts

# Step 1: Ingest and chunk PDF
python src/ingest_pdf.py

# Step 2: Build embeddings + FAISS index
python src/embed_store.py

# Optional: Build knowledge graph
python src/build_graph.py

# Optional: Process YouTube videos into lessons
python src/process_videos.py

4. Run the Streamlit app

streamlit run src/app_streamlit.py



