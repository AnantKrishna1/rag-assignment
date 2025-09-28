#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Done. Put your chapter PDF in data/chapter.pdf and run ingest scripts."
