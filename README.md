# Summarisation Tools

This repository contains a collection of tools designed to help you summarize various types of content efficiently. Whether it's text, documents, URLs, or YouTube videos, these tools are here to make the process seamless and effective. All text converted from various formats using the tool "tomd.py" (by default the files are contained in the "_input" folder) into Markdown (in the "_markdown" folder).

Finally a "summd.py" convert Markdown to summarised versions placed in the "_output" folder. The summaries are created using 3 summarisation methods:

* keyword extraction using YAKE
* [temporarily disabled] Up to 10 bullet points (configurable) for abstract generated using LSA (Latent Semantic Analysis) via [sumy](https://github.com/miso-belica/sumy)
* full summary by headings using the LLM of your choice using Ollama.

Various formats are supported by "tomd.py", including:

* .txt
* .csv
* .pdf
* .docx
* .pptx
* .html

Also understands URLs and Youtube videos.

## Basic usage

```sh
conda activate main
python tomd.py # Converts all files recursively in the "_input" folder
python tomd.py [file|dir|url|youtube ...]
python summd.py # by default converts all files recursively in the "_markdown" folder
python tomd.py [file|dir ...] # Converts only .txt and .md files
```

## Converb web pages and Youtube videos to Markdown

```sh
python news2md.py url # convert URL to markdown using newspaper3k
python webd2md.py url # convert URL to markdown using readability-lxml
python yt2md.py url   # convert YouTube video captions to markdown
```

## Convert PDF to Markdown using Marker PDF

```sh
marker_single --batch_multiplier 20 input.pdf outdir_dir
```

## Convert Powerpoint to Markdown using pptx2md

```sh
pptx2md input.pptx # generates out.md and img/
```

## Experiments

The "experiments" folder contains various experiments used to build these tools. These experiments may no longer work as they are no longer maintained.

## Creating summarise conda environment

Install [Ollama](https://ollama.com/download), then install [conda-forge](https://conda-forge.org/download/).

Install LLM model of your choice (if you have at least 64GB memory on a Mac M system, I recommend "gemma2:27b-instruct-fp16")

```sh
ollama pull gemma2:27b-instruct-fp16
conda create -n main python=3.12
conda activate main
conda install -c pytorch pytorch torchvision torchaudio
conda install bs4 lxml transformers ipykernel ipywidgets pypandoc markdownify readability-lxml matplotlib scipy sumy yake ollama-python lxml-html-clean yt-dlp youtube-transcript-api mlx mlx-lm python-frontmatter fastfetch imagemagick ffmpeg go hugo rust ruby compilers
pip install marker_pdf pymupdf4llm newspaper3k pptx2md pytubefix markitdown aksharamukha mistralai --upgrade
```
