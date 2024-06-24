# summarise.py
#
# Generic file to Markdown summariser
# Takes filenames or directories as command line arguments
# (by default all files in "_input" directory)
# Then summarise each file according to type into "_output/file-summ.md"
# as Markdown with headings retained where appropriate and bullet points in content.
# Currently accepted file types: .txt, .csv, .md, .pdf

import sys
import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36"
import argparse
import re
import requests
import pypandoc
import pymupdf4llm
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Customise to model and parameters of your choice
llm = Ollama(model="llama3:8b-instruct-fp16", temperature=0.3, num_ctx=8192)

def youtube_id(url):
    parsed_url = urlparse(url)
    video_id = None

    if 'youtu.be' in parsed_url.netloc:
        # The ID is the last part of the path
        video_id = parsed_url.path[1:]
    elif 'youtube.com' in parsed_url.netloc and 'watch' in parsed_url.path:
        # The ID is after 'v=' in the query string
        query_params = parse_qs(parsed_url.query)
        video_id = query_params['v'][0]
    else:
        raise ValueError("Invalid YouTube URL")

    return video_id

def output_file(s, prefix):
    input_prefix = "_input/"

    # Strip _input/ prefix if it exists
    if s.startswith(input_prefix):
        s = s[len(input_prefix):]

    # Add _output/ prefix
    s = prefix + s

    return s

def output_summary(filename: str, summary: str):
    base = os.path.splitext(filename)[0]
    
    summary_file = output_file(f"{base}.md", "_output/")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as file:
        file.write(summary)
    print(f'Converted to Markdown summary {summary_file}')
    
def output_text(filename, text):
    prompt = PromptTemplate.from_template(
        "You are an efficient text summarizer. Provide a summary in headings and bullet points on the document in Markdown format, without preamble, delimited by <document> and </document>. Document: <document>{context}</document>. Summary:"
    )
    chain = prompt | llm
    
    output_summary(filename, chain.invoke({"context": text}))
    
def output_md(filename, markdown):
    prompt = PromptTemplate.from_template(
        "You are an efficient text summarizer. Do not add preamble or explanations in your output. Summarize the following Markdown document into Markdown format. The document is delimited by <document> and </document>. Retain Markdown headings, and summarize content underneath heading into bullet points. Do not add any material not in the document. Document: <document>{context}</document>. Summary:"
    )

    chain = prompt | llm
    
    output_summary(filename, chain.invoke({"context": markdown}))
    
def save_markdown(filename, markdown):
    base = os.path.splitext(filename)[0]
    markdown_file = output_file(f"{base}.md", "_processed/")
    os.makedirs(os.path.dirname(markdown_file), exist_ok=True)
    with open(markdown_file, 'w') as ofile:
        ofile.write(markdown)
    print(f'Converted to Markdown {markdown_file}')
    
    output_md(filename, markdown)
    
def save_text(filename, text):
    base = os.path.splitext(filename)[0]
    text_file = output_file(f"{base}.txt", "_processed/")
    os.makedirs(os.path.dirname(text_file), exist_ok=True)
    with open(text_file, 'w') as ofile:
        ofile.write(text)
    print(f'Converted to TXT {text_file}')
    
    output_text(filename, text)

def html2md(filename, html):
    html_nostyle = re.sub(r'<style.*?>.*?</style>', '', html, flags=re.DOTALL)
    html_noscript = re.sub(r'<script.*?>.*?</script>', '', html_nostyle, flags=re.DOTALL)
    soup = BeautifulSoup(html_noscript, "html.parser")
    markdown = md(str(soup), default_title=True, heading_style="ATX")
    save_markdown(filename, markdown)

def process_txt(file):
    print("Processing Text file:", file.name)
    
    output_text(file.name, file.read())
    
def process_md(file):
    print("Processing Markdown file:", file.name)

    output_md(file.name, file.read())
    
def process_doc(file):
    print("Processing Doc file:", file.name)

    # Convert file to markdown
    markdown = pypandoc.convert_file(file.name, 'md')
    
    save_markdown(file.name, markdown)
    
def process_pdf(file):
    print("Processing PDF file:", file.name)

    # Convert file to markdown
    markdown = pymupdf4llm.to_markdown(file.name)
    
    save_markdown(file.name, markdown)
    
def process_html(file):
    print("Processing HTML file:", file.name)
    
    html2md(file.name, file.read())    

def process_csv(file):
    print("Processing CSV file:", file.name)
    
    loader = CSVLoader(file_path=file.name)
    output_text(file.name, loader.load())

def process_pdf2(file):
    print("Processing PDF file:", file.name)
    
    loader = PyPDFLoader(file.name)

    docs = loader.load()

    prompt = PromptTemplate.from_template(
        "You are an efficient text summarizer. Summarize the following page from a longer document, delimited by <page> and </page>, in headings and bullet points, without any preamble, in markdown format. Ignore any page header or footer. Do not add any material not in the document. If there is nothing to summarize, do not add any bullet points. Page: <page>{context}</page>. Summary:"
    )
    chain = prompt | llm

    summary = ""
    for doc in docs:
        summary = summary + chain.invoke({"context": doc.page_content})
        
    output_summary(file.name, summary)
        
def unknown_file(file):
    print("Unknown file type. No processing performed for:", file.name)

def process_url(url):
    print("Processing URL:", url)
    response = requests.get(url)
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    html2md(filename, response.text)    

def process_video(url):
    print("Processing Youtube video:", url)
    video = YoutubeLoader.from_youtube_url(url, add_video_info=True).load()
    save_text(youtube_id(url), video[0].page_content)
    
def process_path(path):
    extension_map = {
        '.txt': process_txt,
        '.md': process_md,
        '.html': process_html,
        '.csv': process_csv,
        '.pdf': process_pdf,
        '.docx': process_doc,
        '.pptx': process_doc
    }
    
    _, file_extension = os.path.splitext(path)
    func = extension_map.get(file_extension, unknown_file)
    with open(path, 'r') as f:
        func(f)

def process_dir(folder):
    for foldername, _, filenames in os.walk(folder):
        for filename in filenames:
            path = os.path.join(foldername, filename)
            process_path(path)            
            
def main():
    if len(sys.argv) < 2:
        print("Processing _input directory by default")
        path = "_input"
        process_dir(path)
    else:
        parser = argparse.ArgumentParser(description="summarise.py [file|dir|url, ...]")
        parser.add_argument("path", nargs="+", help="Path to a URL, file or directory")
        args = parser.parse_args()

        for path in args.path:
            parsed_input = urlparse(path)
            if bool(parsed_input.scheme):
                domain = parsed_input.netloc
                if 'youtube.com' in domain or 'youtu.be' in domain:
                    process_video(path)
                else:
                    process_url(path)
            elif os.path.isfile(path):
                process_path(path)
            elif os.path.isdir(path):
                process_dir(path)
            else:
                print(f'The file {path} does not exist')

if __name__ == "__main__":
    main()
