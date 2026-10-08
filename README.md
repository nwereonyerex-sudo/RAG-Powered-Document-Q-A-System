# RAG-Powered Document Q&A System

Upload a CV, name a job, and get three things back: what the CV actually says, how to change it for that role, and what to expect in the job. Follow-up questions stay tied to the file. Under the page is a retrieval-augmented generation pipeline: loaders and a text splitter, a persistent vector database, and an LLM, orchestrated with LangChain.

RAG is the dominant pattern for production LLM applications in 2025–2026. A full pipeline shows how embeddings, chunking, and retrieval quality fit together, and how generation is deployed on top of a store rather than trained from scratch. LangChain, vector databases, and LLM orchestration now show up together in ML engineering roles. This repo is one project that uses all three, on a CV and on the other files people ask questions about: research papers, company reports, books, and legislation.

## The problem

A raw LLM will answer from memory. On a 10-Q, a paper, or a statute, that produces fluent text that may not be in the file. The useful system has to:

- Accept mixed files. Project Gutenberg is plain text. SEC EDGAR filings are HTML. arXiv papers, company reports, and legislation are PDFs.
- Cut long documents into chunks that still carry a source and a page.
- Search those chunks by meaning, not by keyword alone.
- Hand only the retrieved text to the model, and show where the answer came from.
- Let chunk size, overlap, and the number of chunks change without mixing old vectors into the new index.

## How it was solved

The pipeline is split so retrieval can be tuned without rewriting the model call.

1. **Ingest.** [`src/rag/ingest.py`](src/rag/ingest.py) loads PDF, TXT, Markdown, and HTML. Each chunk keeps `source`, `doc_type`, `page`, and `chunk_id`.
2. **Index.** [`src/rag/store.py`](src/rag/store.py) embeds chunks with `sentence-transformers/all-MiniLM-L6-v2` and stores them in persistent Chroma. The collection name includes chunk size and overlap, so a new splitter setting does not reuse old vectors.
3. **Retrieve.** The same module searches with maximum marginal relevance (diverse chunks) or plain similarity, optionally limited to one file.
4. **Answer.** [`src/rag/chain.py`](src/rag/chain.py) puts only the retrieved chunks in a normal question. If they do not contain the answer, the model is told to say so.
5. **Review.** The same module can read the whole file. **Describe this file** explains the content and how to make the writing stronger. **Improve this CV for the role** takes a job title and returns what the CV shows, edits aimed at that role, and what to expect in the job. CV edits stay inside the file. The job description is labeled as general knowledge about the role.
6. **Use.** [`app.py`](app.py) is a Streamlit app: upload a CV, type the role, read the review, then ask a follow-up. Retrieval knobs stay in the sidebar.
7. **Check.** [`scripts/eval_retrieval.py`](scripts/eval_retrieval.py) asks three questions against the bundled samples and fails if the expected fact is missing from the top chunks.

```mermaid
flowchart LR
  upload[Upload_CV] --> loaders[Loaders_PDF_TXT_HTML]
  loaders --> chunk[Recursive_splitter]
  chunk --> embed[MiniLM_embeddings]
  embed --> chroma[Chroma_persist]
  role[Job_role] --> review[Whole_file_review]
  chroma --> review
  review --> brief[Content_role_edits_and_expectations]
  question[Follow_up_question] --> retriever[Retriever]
  chroma --> retriever
  retriever --> answer[Grounded_answer]
```

| Piece | Where it lives |
| --- | --- |
| LangChain loaders, splitter, prompt, and chain | [`src/rag/ingest.py`](src/rag/ingest.py), [`src/rag/chain.py`](src/rag/chain.py) |
| Vector database | [`src/rag/store.py`](src/rag/store.py) |
| LLM provider switch, grounded answers, and CV-for-role review | [`src/rag/chain.py`](src/rag/chain.py) |
| Retrieval knobs and a repeatable check | [`app.py`](app.py), [`scripts/eval_retrieval.py`](scripts/eval_retrieval.py) |

## A fault found in use

Uploading a file and asking what it contained returned exactly: "I cannot find that in the uploaded documents."

The file had been chosen in the browser, but it was not in the search index. The app was still searching the bundled sample report, filing, and paper. The prompt is required to refuse when those passages do not contain the answer, so every question about the new file got that sentence.

A second limit made whole-file questions fail even after a successful index. A normal question sends only the top 4 chunks. Asking what the file is, or how to improve it, needs the whole document.

The fix is in [`app.py`](app.py) and [`src/rag/chain.py`](src/rag/chain.py):

- Choosing a file indexes it immediately. The bundled samples stay out unless that option is turned on.
- **Describe this file** and **Improve this CV for the role** read every stored chunk of that file, not only the top 4.
- A PDF with no selectable text is reported as unreadable instead of being treated as an empty search.

The same fault, and the other issues from the build, are written up in [`journal.md`](journal.md).

## Setup

Python 3.14 is what this machine used. A 3.11 or 3.12 environment is fine if that is what you have.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` for `gpt-4o-mini`. Leave the key empty and the app uses the local Hugging Face model named in `LOCAL_MODEL`. Embeddings stay local either way, so indexing does not need an API key.

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

Upload a PDF, TXT, Markdown, or HTML CV. The file is indexed when you select it. Type a job role and click **Improve this CV for the role**. The reply has three parts: what the CV shows, how to improve it for that role, and what to expect in the job. **Describe this file and how to improve it** reviews the document without a job title. A PDF with no selectable text is reported instead of being treated as an empty search.

Check retrieval without calling the LLM:

```bash
python scripts/eval_retrieval.py
```

## Corpora

The app does not crawl the web. Drop files into `data/raw/` or upload them. These public sources match the loaders:

- **Project Gutenberg** (plain text): [Frankenstein](https://www.gutenberg.org/cache/epub/84/pg84.txt)
- **SEC EDGAR** (HTML filings): [company search](https://www.sec.gov/edgar/search/). Save the filing document as `.html`.
- **arXiv** (PDF): a paper PDF such as [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)

`samples/` has three short stand-ins so the retrieval check runs with no download: a quarterly report, an HTML filing excerpt, and a paper-style note.

## Retrieval defaults

| Knob | Default | Why |
| --- | --- | --- |
| Chunk size | 1000 characters | Long enough for a paragraph, short enough to point at a page |
| Overlap | 200 characters | A sentence cut by the splitter still appears in the next chunk |
| Top k | 4 | Enough context for a short answer without flooding the prompt |
| Search | MMR | Pulls chunks that are relevant and not copies of each other |

Switch search to similarity when you want the nearest chunks only. A score threshold above 0 drops weak similarity hits. Changing chunk size or overlap selects a different Chroma collection. Re-index after that change.

## Layout

```
app.py                      Streamlit upload, chat, citations, knobs
src/rag/config.py           Chunking, retrieval, and provider settings
src/rag/ingest.py           PDF, text, and HTML loaders
src/rag/store.py            Chroma index and retriever
src/rag/chain.py            Grounded answers, full-file review, and CV-for-role review
scripts/eval_retrieval.py   Retrieval check against samples/
samples/                    Small report, filing, and paper
journal.md                  Issues hit during the build
```
