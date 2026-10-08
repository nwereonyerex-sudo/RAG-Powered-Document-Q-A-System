# Build journal

Issues hit while building the RAG-Powered Document Q&A System, and the fix for each one.

## GitHub CLI could not authenticate

`gh auth status` reported that the token in the keyring for `nwereonyerex-sudo` was invalid, so the CLI could not create a repository or open a pull request.

The repository already existed. The remote is [RAG-Powered-Document-Q-A-System](https://github.com/nwereonyerex-sudo/RAG-Powered-Document-Q-A-System). Work is committed on a branch per stage (`feat/scaffold`, `feat/ingest-store`, `feat/rag-chain`, `feat/streamlit-ui`, `feat/docs`) and merged to `main` only after that stage runs. `main` is what gets pushed to that remote.

## The virtualenv could not be created in the sandbox

`python3 -m venv .venv` failed with `Operation not permitted` on `.venv/include`. The sandbox blocks some filesystem calls the venv module needs.

The environment was created again with sandboxing disabled. `.venv/` is listed in `.gitignore`, so the environment stays local.

## This machine only has Python 3.14

`python3.12` and `python3.11` are not installed. Older scientific wheels often lag a new Python version, so a pin to an old LangChain or Chroma release would have failed to build.

`requirements.txt` asks for current releases (`langchain`, `chromadb`, `torch`, `sentence-transformers`) that publish macOS arm64 wheels for 3.14. Indexing uses the small local MiniLM embedding model, so the retrieval check does not need a GPU or an OpenAI key.

## `main` did not exist until the first commit

The folder was a git repository with no commits, so there was no `main` branch to merge into.

The scaffold commit was made on `feat/scaffold`. `main` was created from that commit. Later stages branch from `main` and merge back after their check passes.

## HTML loading required lxml by default

The first retrieval run died inside `BSHTMLLoader` before any chunks were stored. LangChain's loader defaults to the `lxml` parser, and that package was not installed.

The HTML loader now passes `bs_kwargs={"features": "html.parser"}`, which uses the parser in the Python standard library. SEC-style HTML files index without an extra dependency.

## One index cannot serve two chunk settings

Changing chunk size or overlap changes the vectors. Leaving both in the same Chroma collection would mix incompatible chunks and make retrieval scores meaningless.

`Settings.collection_name` includes the chunk size and overlap (`docs_minilm_c1000_o200`). Re-indexing after a knob change writes a separate collection. Re-uploading the same filename deletes that file's previous chunks first, so a second index of the same report does not duplicate it.

## The local model echoed the prompt

With no `OPENAI_API_KEY`, a question about the sample report used `HuggingFaceTB/SmolLM2-135M-Instruct`. Retrieval was right: the Acme chunk was first. The model did answer "Acme Robotics Q3 revenue was $42 million," but the pipeline also returned the system prompt and the context, because the local chat template echoes the full text.

`generate_answer` now keeps only the text after `<|im_start|>assistant`. The cited chunks are still shown separately in the app.

## Uploaded files were invisible to search

After a CV was uploaded, every question came back as "I cannot find that in the uploaded documents."

Three things caused that:

1. The file picker only held the file in the browser. Nothing was written to the Chroma index until **Index documents** was clicked, and that click was easy to miss. `data/raw/` still contained only an earlier copy of a form, not the CV the user had just selected.
2. **Also index the bundled samples** was on by default. Retrieval therefore returned the sample report, the sample filing, and the sample paper. The prompt tells the model to refuse when the retrieved text does not contain the answer, so the refusal was correct for those samples and useless for the CV.
3. Even after the CV was indexed (4 pages, 13 chunks), a question about the whole file only received the top 4 chunks. A summary or an improvement pass cannot be done from a fragment.

A separate PDF, a W-8BEN form, made the same sentence inevitable for another reason. `PyPDFLoader` extracted 1 page and 0 characters. The letters are drawn as vector shapes, not stored as text, and the file has no form fields. There was nothing to embed.

The running Streamlit process then failed with `ImportError: cannot import name 'describe_document' from 'rag.chain'` after that function was added. The process had loaded `rag.chain` at startup and kept the old module in memory. Restarting the app loaded the new function. A direct import of `describe_document` succeeded.

What changed:

- Selecting a file indexes it immediately. Samples are off unless the user opts in.
- A PDF with no selectable text raises a clear error instead of storing zero chunks and pretending the search worked.
- `describe_document` and `review_cv_for_role` read every chunk of the chosen file. The role review returns what the CV shows, how to edit it for that job, and what to expect in the role. The last section is general knowledge about the job, not new facts inserted into the CV.

## Document questions and CV matching were one page

The same screen uploaded a CV, accepted a job title, and also answered questions about filings and papers. A requirement match needs a different result from a grounded answer: each job requirement, the CV line that supports it, and a suggestion that does not invent a skill.

The Streamlit app now has two pages. Document Q&A keeps the index, the grounded prompt, and the citations. CV & Job Match extracts requirements and checks each one against retrieved CV text. If the CV says REST APIs and the posting asks for FastAPI, the match stays partial and the suggestion keeps the REST API line unless the candidate has actually used FastAPI.

## A job title is not the posting

Naming a role produced general advice about that job. It could not tell the candidate which lines in a specific posting to answer, or what to type into the CV.

The CV box and the job-description box are now separate. **Guide my CV from this job description** reads every chunk of both files. The reply has four parts: what the posting asks for, what the CV already covers, what to input in each section, and a step-by-step edit. Fill-in lines use blanks. The model is told not to invent employers, dates, or achievements. If only the posting is uploaded, the same button is a drafting guide.

## GitHub `main` already had a commit

The feature branches pushed. `main` was rejected because the remote already had an initial commit whose README was one sentence: the project title and the one-line goal.

That commit was merged into local `main` with `--allow-unrelated-histories`. The full README in this repo was kept, because it is the problem statement and the description of the pipeline. The one-line remote README is the ancestor of that file.
