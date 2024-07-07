# summd.py
#
# Generic summariser for text files
# Takes filenames or directories as command line arguments
# (by default all files in "_markdown" directory)
# Then summarise each file into "_output/file-summ.md"
# as Markdown with headings retained where appropriate and bullet points in content.

import sys
import os
import argparse
import re
import pymupdf4llm

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Customise to model and parameters of your choice
llm = Ollama(model="llama3:8b-instruct-fp16", temperature=0.3, num_ctx=8192)
# llm = Ollama(model="command-r-plus:latest", temperature=0.3, num_ctx=131072)
# llm = Ollama(model="mixtral:8x22b", temperature=0.3, num_ctx=65536)

def output_file(s, prefix):
    input_prefix = "_markdown/"

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
    
def output_md(filename, markdown):
    prompt = PromptTemplate.from_template(
        "You are an efficient text summarizer. Do not add preamble or explanations in your output. Summarize the following document into Markdown format. The document is delimited by <document> and </document>. Retain Markdown headings, and summarize content underneath each heading into bullet points without preamble. Do not add any material not in the document. Document: <document>{context}</document>. Summary:"
    )

    chain = prompt | llm
    
    output_summary(filename, chain.invoke({"context": markdown}))
    
def process_path(path):
    _, file_extension = os.path.splitext(path)
    if (file_extension == ".md" or file_extension == ".txt"):
        print("Processing file:", path)
    else:
        print("Skipping unknown file ", path)
        return

    with open(path, 'r') as file:
        output_md(path, file.read())

def process_dir(folder):
    for foldername, _, filenames in os.walk(folder):
        for filename in filenames:
            path = os.path.join(foldername, filename)
            process_path(path)            
            
def main():
    if len(sys.argv) < 2:
        print("Processing _input directory by default")
        path = "_markdown"
        process_dir(path)
    else:
        parser = argparse.ArgumentParser(description="summarise.py [file|dir] ...")
        parser.add_argument("path", nargs="+", help="Path to a file or directory")
        args = parser.parse_args()

        for path in args.path:
            if os.path.isfile(path):
                process_path(path)
            elif os.path.isdir(path):
                process_dir(path)
            else:
                print(f'The file {path} does not exist')

if __name__ == "__main__":
    main()
