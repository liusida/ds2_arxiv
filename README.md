# ds2_arxiv

(1). download meta_data from arxiv using OIA interface. (321,752 papers)

(2). select papers from two categories: cs.LG and cs.AI. (93,911 papers)

(3). download citation numbers from semantic scholar for (2).

(4). select papers with >100 citations. (4,422 papers)

(5). pass (4) through a pretrained BERT network. (`BERT_features.pt` (4422x768))

