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

## Converb web pages and Youtube videos to Markdown

```sh
python news2md.py url # convert URL to markdown using newspaper3k
python webd2md.py url # convert URL to markdown using readability-lxml
python yt2md.py url   # convert YouTube video captions to markdown
```

## Convert PDF to Markdown using Marker PDF

```sh
marker_single input.pdf outdir_dir --batch_multiplier 20
```

## Other summarisers (deprecated)

* summcsv.py: Summarise a CSV file using Ollama and LangChain.
* summpdf.py: Summarise a PDF file using Ollama and LangChain.
* summweb.py: Summarise a web page using Ollama and LangChain.

To use:

## Creating langchain conda environment

```sh
conda create -n summarise langchain bs4 lxml transformers ipykernel ipywidgets pytube pypdf tiktoken pypandoc markdownify readability-lxml
conda activate summarise
conda install pytorch torchvision -c pytorch
pip install langchain_community
pip install langchain_chroma
pip install youtube_transcript_api
pip install marker_pdf
pip install pymupdf4llm
pip install yt_dlp
pip install newspaper3k
pip install ollama
```
