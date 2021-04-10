# ds2_arxiv

(1). download meta_data from arxiv using OIA interface. (321,752 papers)

(2). select papers from two categories: cs.LG and cs.AI. (93,911 papers) `data/harvest_LG_AI`

(3). download citation numbers from semantic scholar for (2). `data/citations_s2`

(4). select papers with >100 citations. (4,422 papers) `cited_100.pickle` (along with `data/harvest_LG_AI_100`, `data/citations_s2_100`)

(5). pass (4) through a pretrained BERT network. (4422x768) `BERT_features.pt` (same order as `cited_100.pickle`)

(6). save features of most cited 500 papers to csv  (500x768) `features.csv`

