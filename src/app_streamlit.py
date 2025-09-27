# src/app_streamlit.py
import streamlit as st
import json
from pathlib import Path
from assess_answer import retrieve_top_k, grade_answer
from generate_questions import generate
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_DIR = Path("data")
TRANSCRIPTS = DATA_DIR / "video_transcripts.jsonl"

st.set_page_config(page_title="EdTech RAG MVP")

st.title("RAG-based EdTech MVP — Economics Chapter")

mode = st.sidebar.selectbox("Mode", ["Search / QA", "Lessons", "Assess answer", "Knowledge Graph"])

if mode == "Search / QA":
    query = st.text_input("Enter question for RAG:")
    if st.button("Search & Answer"):
        results = retrieve_top_k(query, k=5)
        st.subheader("Top retrieved chunks:")
        context = ""
        for r in results:
            st.write(f"Page: {r.get('page')} | Topic: {r.get('topic')}")
            st.write(r.get("text")[:500] + "...")
            context += "\n\n" + r.get("text")
        # generate answer with generator
        prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer in concise points."
        answer = generate(prompt)
        st.subheader("Generated Answer")
        st.write(answer)

elif mode == "Lessons":
    st.header("Auto-generated Lesson Pages (videos)")
    if TRANSCRIPTS.exists():
        for line in open(TRANSCRIPTS, "r", encoding="utf-8"):
            rec = json.loads(line)
            with st.expander(f"Video {rec['video_id']} — Key terms: {', '.join(rec['keyterms'][:5])}"):
                st.write("Overview:")
                st.write(rec['text'][:800] + "...")
                st.write("Highlights (timestamped):")
                for h in rec['highlights']:
                    st.write(f"- {int(h['start'])}s: {h['text']}")
                st.write("MCQs:")
                st.write(rec['mcqs'])
                st.write("Essay question:")
                st.write(rec['essay'])
    else:
        st.info("No video transcript processed yet. Run `process_videos.py` first.")

elif mode == "Assess answer":
    st.header("Student Answer Assessment")
    question = st.text_input("Question (for context):")
    student_answer = st.text_area("Paste student's answer here:")
    if st.button("Grade"):
        refs = [r["text"] for r in retrieve_top_k(question, k=5)]
        score, sims = grade_answer(student_answer, refs)
        st.metric("Score (0-100)", score)
        st.write("Similarity scores per reference chunk:", sims)

elif mode == "Knowledge Graph":
    st.header("Knowledge Graph")
    gpath = DATA_DIR / "knowledge_graph.png"
    if gpath.exists():
        st.image(str(gpath))
    else:
        st.info("Run build_graph.py to create knowledge graph.")