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
