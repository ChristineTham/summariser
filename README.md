# Summariser

Generic text summariser based on langchain and Ollama. Understands the
following file formats:

* .txt
* .csv
* .pdf
* .docx
* .pptx
* .html

Also understands URLs and Youtube videos.

```sh
conda activate langchain
python summarise.py [file|dir|url|youtube ...]
```

Defaults to summarising recursively all files in "_input" and places summaries
in "_output" and generated markdown files in "_markdown"

## Other summarisers (deprecated)

* summcsv.py: Summarise a CSV file using Ollama and LangChain.
* summpdf.py: Summarise a PDF file using Ollama and LangChain.
* summyt.py: Summarise a YouTube video transcript using Ollama and LangChain.
* summweb.py: Summarise a web page using Ollama and LangChain.
* summhtml.py: Summarise a HTML file using Ollama and LangChain.

To use:

## Creating langchain conda environment

```sh
conda create -n summarise langchain bs4 lxml transformers ipykernel ipywidgets pytube pypdf tiktoken pypandoc markdownify
conda activate summarise
conda install pytorch torchvision -c pytorch
pip install langchain_community
pip install langchain_chroma
pip install youtube_transcript_api
pip install marker_pdf
pip install pymupdf4llm
pip install yt_dlp
```
