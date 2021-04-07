#!/bin/sh
# Usage:
# nohup sh run.sh &

while true
do
    python ds2_arxiv/2.2.get_citations_from_gscholar.py
done