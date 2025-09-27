# src/build_graph.py
import json
from pathlib import Path
import yake
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = Path("../data")
META_FILE = DATA_DIR / "emb_metadata.jsonl"
OUT_GRAPH_PNG = DATA_DIR / "knowledge_graph.png"

def extract_keywords(text, max_kw=3):
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=max_kw)
    kws = kw_extractor.extract_keywords(text)
    return [k for k, score in kws]

def build_graph():
    docs = []
    with open(META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    G = nx.Graph()
    for d in docs:
        kws = extract_keywords(d["text"], max_kw=3)
        for k in kws:
            G.add_node(k)
        # connect co-occurring keywords
        for i, a in enumerate(kws):
            for b in kws[i+1:]:
                if G.has_edge(a,b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a,b, weight=1)
    plt.figure(figsize=(12,12))
    pos = nx.spring_layout(G, k=0.8)
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=10)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw_networkx_edges(G, pos, width=[1 + w*0.2 for w in weights])
    plt.axis('off')
    plt.savefig(OUT_GRAPH_PNG, bbox_inches='tight')
    print("Saved graph to", OUT_GRAPH_PNG)

if __name__ == "__main__":
    build_graph()
