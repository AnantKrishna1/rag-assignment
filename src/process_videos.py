# src/process_videos.py
from youtube_transcript_api import YouTubeTranscriptApi
import json
from pathlib import Path
from tqdm import tqdm
import yake
from generate_questions import generate_questions_for_text

DATA_DIR = Path("../data")
VIDEO_IDS = [
    # example: parse playlist to get IDs or manually list
    # 'OPV1BOs1ISI' etc. For the playlist you provided, you can extract IDs
]

OUT_TRANSCRIPTS = DATA_DIR / "video_transcripts.jsonl"
YAKE_KW = yake.KeywordExtractor(lan="en", n=2, top=8)

def get_transcript(video_id):
    try:
        trans = YouTubeTranscriptApi.get_transcript(video_id)
        return trans
    except Exception as e:
        print("Transcript error", video_id, e)
        return []

def timestamped_highlights(transcript, top_n=8):
    # aggregate sentences -> compute scores by keyword counts
    text = " ".join([t['text'] for t in transcript])
    kws = [k for k, _ in YAKE_KW.extract_keywords(text)]
    highlights = []
    for t in transcript:
        score = sum(1 for kw in kws if kw.lower() in t['text'].lower())
        highlights.append((t, score))
    highlights_sorted = sorted(highlights, key=lambda x: x[1], reverse=True)
    # return top_n unique timestamps
    top = []
    seen_t = set()
    for t, sc in highlights_sorted:
        start = int(t['start'])
        if start not in seen_t:
            top.append({"start": t['start'], "duration": t.get('duration',0), "text": t['text']})
            seen_t.add(start)
        if len(top) >= top_n:
            break
    return top

def process(video_id):
    trans = get_transcript(video_id)
    if not trans:
        return
    text = " ".join([t['text'] for t in trans])
    keyterms = [k for k,_ in YAKE_KW.extract_keywords(text)]
    highlights = timestamped_highlights(trans)
    # generate questions
    mcqs, essay = generate_questions_for_text(text)
    rec = {
        "video_id": video_id,
        "text": text,
        "keyterms": keyterms,
        "highlights": highlights,
        "mcqs": mcqs,
        "essay": essay
    }
    return rec

if __name__=="__main__":
    # list video ids manually or parse playlist
    # For demo, fill VIDEO_IDS with a few IDs
    VIDEO_IDS = ["OPV1BOs1ISI"]  # example
    with open(OUT_TRANSCRIPTS, "w", encoding="utf-8") as f:
        for vid in VIDEO_IDS:
            rec = process(vid)
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Saved transcripts and lessons to", OUT_TRANSCRIPTS)