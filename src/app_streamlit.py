# src/app_streamlit.py

import streamlit as st
import json
from pathlib import Path
from assess_answer import retrieve_top_k, grade_answer
from generate_questions import generate
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("data")
TRANSCRIPTS = DATA_DIR / "video_transcripts.jsonl"
INDEX_PATH = DATA_DIR / "embed_store.pkl"  # <-- debug check

# -----------------------------
# Auto-build embed index if missing (for Streamlit Cloud)
# -----------------------------
from embed_store import build_embed_store  # make sure this exists in src/embed_store.py

if not INDEX_PATH.exists():
    st.info("⚠️ Index not found. Building embed index now...")
    try:
        build_embed_store()  # This should create data/embed_store.pkl
        st.success("✅ Embed index built successfully!")
    except Exception as e:
        st.error(f"Failed to build embed index: {e}")

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="EdTech RAG MVP")
st.title("RAG-based EdTech MVP — Economics Chapter")

# -----------------------------
# Debug snippet: check index file
# -----------------------------
if INDEX_PATH.exists():
    print("✅ Index file exists on cloud!")
else:
    print("⚠️ Index file NOT found on cloud!")

# -----------------------------
# Sidebar: Mode selection
# -----------------------------
mode = st.sidebar.selectbox(
    "Mode", ["Search / QA", "Lessons", "Assess answer", "Knowledge Graph"]
)

# -----------------------------
# Mode: Search / QA
# -----------------------------
if mode == "Search / QA":
    query = st.text_input("Enter question for RAG:")
    if st.button("Search & Answer"):
        results = retrieve_top_k(query, k=5)
        if not results:
            st.warning("⚠️ No results found. Please check if index is built (`python build_index.py`).")
        else:
            st.subheader("Top retrieved chunks:")
            context = ""
            for r in results:
                st.write(f"Page: {r.get('page')} | Topic: {r.get('topic')}")
                st.write(r.get("text")[:500] + "...")
                context += "\n\n" + r.get("text")
            try:
                prompt = (
                    f"Use the context below to answer the question.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {query}\nAnswer in concise points."
                )
                answer = generate(prompt)
                st.subheader("Generated Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Answer generation failed: {e}")

# -----------------------------
# Mode: Lessons
# -----------------------------
elif mode == "Lessons":
    st.header("Auto-generated Lesson Pages (videos)")
    if TRANSCRIPTS.exists():
        try:
            for line in open(TRANSCRIPTS, "r", encoding="utf-8"):
                rec = json.loads(line)
                with st.expander(
                    f"Video {rec['video_id']} — Key terms: {', '.join(rec.get('keyterms', [])[:5])}"
                ):
                    st.write("Overview:")
                    st.write(rec.get('text', '')[:800] + "...")
                    st.write("Highlights (timestamped):")
                    for h in rec.get('highlights', []):
                        st.write(f"- {int(h.get('start', 0))}s: {h.get('text', '')}")
                    st.write("MCQs:")
                    st.write(rec.get('mcqs', []))
                    st.write("Essay question:")
                    st.write(rec.get('essay', ''))
        except Exception as e:
            st.error(f"Error loading transcripts: {e}")
    else:
        st.info("No video transcript processed yet. Run `process_videos.py` first.")

# -----------------------------
# Mode: Assess answer
# -----------------------------
elif mode == "Assess answer":
    st.header("Student Answer Assessment")
    question = st.text_input("Question (for context):")
    student_answer = st.text_area("Paste student's answer here:")
    if st.button("Grade"):
        if not student_answer.strip():
            st.warning("⚠️ Please enter a student answer.")
        else:
            refs = [r["text"] for r in retrieve_top_k(question, k=5)]
            if not refs:
                st.warning("⚠️ No reference material found. Rebuild the index first.")
            else:
                score, sims = grade_answer(student_answer, refs)
                st.metric("Score (0-100)", score)
                st.write("Similarity scores per reference chunk:", sims)

# -----------------------------
# Mode: Knowledge Graph
# -----------------------------
elif mode == "Knowledge Graph":
    st.header("Knowledge Graph")
    gpath = DATA_DIR / "knowledge_graph.png"
    if gpath.exists():
        st.image(str(gpath))
    else:
        st.info("Run build_graph.py to create knowledge graph.")
