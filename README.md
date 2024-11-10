## Denser Chat

<img src="demo.gif" width="800" alt="Denser Chat">

Main features:

* Extract text and tables from PDFs.
* Build a chatbot with [denser-retriever](https://github.com/denser-org/denser-retriever)
* Support interactive Streamlit chatbot app with source highlighted in the PDF

## Installation

First clone the repository.

```bash
git clone https://github.com/denser-org/denser-chat.git
```

Go to the project directory and start a virtual environment. Make sure your python version is 3.11.

```bash
cd denser-chat
python -m venv .venv
source .venv/bin/activate
```

Run the following command to install the required packages.

```bash
pip install -e .
```

Or use this [poetry](https://python-poetry.org/docs/) command

```bash
poetry install
```

## Quick Start

Before building an index, we need to run docker-compose to start Elasticsearch and Milvus services in the background,
which are required for denser-retriever.

```bash
cd denser_chat
docker compose up -d
```

We run the following command to build a chatbot index. The first argument is the sources file which specify files used to build chatbots. Files can be local PDF files, URL PDFs, or URLs. The second argument is the output directory, and the third argument is the index name.

```bash
python build.py sources.txt output test_index
```

This command will build an index `test_index` via denser-retriever. Next we can start a streamlit app with the following
command. As the app relies on ChatGPT or Claude API, we need to set their keys (one is sufficient) in the environment variables.

```bash
export OPENAI_API_KEY="your-openai-key"
export CLAUDE_API_KEY="your-claude-key"
python -m http.server 8000 # start a simple http server to render PDFs
streamlit run demo.py -- --index_name test_index # start the streamlit app on a different terminal
```

Then you can start to ask questions such as "What is in-batch negative sampling ?" or "what parts have stop pins?". You can expect that the chatbot will return the answer with the source highlighted in the PDF.

### License

This project is licensed under the MIT License.