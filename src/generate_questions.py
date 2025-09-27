# src/generate_questions.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL = "google/flan-t5-small"  # small, free
DEVICE = "cpu"  # colab with GPU can use "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(DEVICE)

def generate(prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    out = model.generate(**inputs, max_length=max_length, num_beams=4)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_questions_for_text(text):
    # 5 MCQs
    mcq_prompt = (
        "Read the following passage and generate 5 multiple choice questions (each with 4 options A-D and indicate the correct letter). "
        "Make them balanced in difficulty. Passage:\n\n" + text[:4000]
    )
    mcq_out = generate(mcq_prompt, max_length=512)
    # 1 essay
    essay_prompt = "Read the passage and generate 1 exam-style essay question that tests deep understanding:\n\n" + text[:3000]
    essay_out = generate(essay_prompt, max_length=200)
    # Parse raw output? We'll return as strings
    return mcq_out.strip(), essay_out.strip()