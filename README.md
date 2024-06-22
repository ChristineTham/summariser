# Summariser

Various text summarisers based on langchain and Ollama.

* summcsv.py: Summarise a CSV file using Ollama and LangChain.
* summpdf.py: Summarise a PDF file using Ollama and LangChain.
* summyt.py: Summarise a YouTube video transcript using Ollama and LangChain.
* summweb.py: Summarise a web page using Ollama and LangChain.
* summhtml.py: Summarise a HTML file using Ollama and LangChain.

To use:

```sh
conda activate langchain
python summxx.py content
```

## Creating langchain conda environment

```sh
conda create -n langchain langchain langchainhub bs4 lxml transformers ipykernel ipywidgets pytube pypdf tiktoken
conda activate langchain
conda install pytorch torchvision -c pytorch
pip install langchain_community
pip install langchain_chroma
pip install youtube_transcript_api
```
