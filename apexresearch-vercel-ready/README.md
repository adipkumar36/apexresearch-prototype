# ApexResearch AI – backend (complete)

## Chalane ka tareeka
Windows: `run.bat` double-click (venv + install + ingest + server, sab ek saath).
Mac/Linux: `./run.sh`
Phir http://localhost:8000 kholo.  Pehli baar model download hota hai (~470 MB), thoda wait karo.

Check: http://localhost:8000/api/health
- `model` = sentence-transformers/... hona chahiye (hashing-fallback matlab sentence-transformers install nahi hua)
- `index` = faiss-hnsw (faiss install nahi hua to numpy chalega, app phir bhi kaam karega)

## Features
1. Repository search (FAISS HNSW approximate nearest-neighbour) – apne papers `papers/` mein daalo, `python ingest.py papers`
2. Live arXiv + OpenAlex search – asli papers ke links ke saath (test: /api/live?q=your topic)
3. Sentence-level evidence – highlighted sentences + unke closest match ka text
4. Hindi / multilingual – Hindi proposal English papers se match hota hai.
   Live search ke liye Hindi keywords translate hote hain -> `ANTHROPIC_API_KEY` chahiye (bina key ke sirf repository check hota hai)
5. Novelty advisor – key ho to Claude likhta hai, warna rule-based advice
6. PDF report download – report ke neeche "Download PDF report" button
   (PDF mein Hindi text ke liye `APEX_PDF_FONT` = koi Devanagari .ttf, warna wo hissa "[non-Latin text]" dikhega)
7. Explainable analysis – transparency panel shows search sources, model, index and scoring method; idea decomposition gives passage scores; citation-aware flags suggest missing citations on overlapping passages; related-work suggestions list the closest five papers.
8. Open data discovery – live results include Semantic Scholar, Europe PMC and Crossref alongside arXiv and OpenAlex; CORE can be enabled with `APEX_CORE_API_KEY`. The top three arXiv matches are checked for extractable PDF full text before sentence evidence is produced. In Upload, use “Index arXiv abstracts” to add up to 1,000 results for a topic to the local vector database per request.
9. Batch review – Upload accepts a ZIP with up to 25 supported proposal files for per-file scoring. Keep the ZIP small; each file is limited to 20 MB.
10. Reviewer accounts – Reports supports account creation and sign-in; reports and review decisions sync through the backend database. Accounts are stored in the local SQLite database and are intended for a prototype, not an internet-facing deployment without production authentication and transport security.
11. Multilingual identification – analysis detects dominant and mixed languages in a proposal (including English, Hindi and other supported languages) and shows the detected language names in the report. `langdetect` is installed automatically from `requirements.txt`.
12. Side-by-side evidence and novelty map – reports show each flagged passage beside the matched source and extract proposal question, method, dataset, contribution and a gap to investigate. With `ANTHROPIC_API_KEY`, the map is AI-assisted; without it, fields are extractive and labelled as such.
13. Quote/reference filters – optional controls exclude quoted passages and a trailing References/Bibliography/Works Cited section from scoring. They do not edit the user's submitted text; the report states the counts excluded.
14. Multilingual demo – Upload includes a Hindi–English sample and per-passage language labels. Cross-language matching still depends on the configured multilingual embedding model and having related work in the repository or live search.
15. Evaluation dashboard – Upload accepts a UTF-8 CSV with `text_a,text_b,label` columns (`1` similar, `0` different), and computes accuracy, precision, recall, F1, confusion counts and runtime from those labelled pairs. It does not estimate scholarly source coverage.

## Settings (.env)
`.env.example` ko `.env` naam se copy karo. Sab optional hai.

## Notes
- Har analyzed submission DB mein save hoti hai (aage ki submissions usse compare hoti hain). Demo se pehle `apex.db` delete kar do to fresh start milega.
- Similarity % ke thresholds (`Embedder.low/high` in main.py) andaze se set hain; apne known papers pe test karke tune karna.
- Demo ke liye texts: DEMO.md
