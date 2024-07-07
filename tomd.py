# tomd.py
#
# Convert many different types of files to Markdown
# Takes filenames or directories as command line arguments
# Then convert each file into markdown into "_markdown/file-summ.md"
# Markdown files are copied without converting
# Currently accepted file types: .html, .pdf, docx, .pptx

import sys
import os
import subprocess
import shutil
import argparse
import re
import pypandoc
import pymupdf4llm
from bs4 import BeautifulSoup
from markdownify import markdownify as md

def output_file(s, prefix):
    input_prefix = "_input/"

    # Strip _input/ prefix if it exists
    if s.startswith(input_prefix):
        s = s[len(input_prefix):]

    # Add _output/ prefix
    s = prefix + s

    return s
    
def save_markdown(filename, markdown):
    base = os.path.splitext(filename)[0]
    markdown_file = output_file(f"{base}.md", "_markdown/")
    os.makedirs(os.path.dirname(markdown_file), exist_ok=True)
    with open(markdown_file, 'w') as ofile:
        ofile.write(markdown)
    print(f'Converted to Markdown {markdown_file}')
    

def html2md(filename, html):
    html_nostyle = re.sub(r'<style.*?>.*?</style>', '', html, flags=re.DOTALL)
    html_noscript = re.sub(r'<script.*?>.*?</script>', '', html_nostyle, flags=re.DOTALL)
    soup = BeautifulSoup(html_noscript, "html.parser")
    markdown = md(str(soup), default_title=True, heading_style="ATX")
    save_markdown(filename, markdown)
    
def process_md(file):
    print("Copying Markdown file:", file.name)

    save_markdown(file.name, file.read())
    
def process_doc(file):
    print("Processing Word file:", file.name)

    # Convert file to markdown
    markdown = pypandoc.convert_file(file.name, 'md')
    
    save_markdown(file.name, markdown)
    
def process_ppt(file):
    print("Processing Powerpoint file:", file.name)
    base = os.path.splitext(file.name)[0]
    markdown_dir = output_file(base, "_markdown/")
    if os.path.exists(markdown_dir):
        shutil.rmtree(markdown_dir)
    os.makedirs(markdown_dir, exist_ok=True)
    result = subprocess.run(['pptx2md', file.name], stdout=subprocess.PIPE)
    shutil.move("out.md", markdown_dir)
    shutil.move("img", markdown_dir)
    print(result.stdout.decode('utf-8'))
    print("Converted to dir ", markdown_dir)

def process_pdf(file):
    print("Processing PDF file:", file.name)

    # Convert file to markdown
    markdown = pymupdf4llm.to_markdown(file.name)
    
    save_markdown(file.name, markdown)
    
def process_html(file):
    print("Processing HTML file:", file.name)
    
    html2md(file.name, file.read())    
        
def unknown_file(file):
    print("Unknown file type. No processing performed for:", file.name)
    
def process_path(path):
    extension_map = {
        '.md': process_md,
        '.html': process_html,
        '.pdf': process_pdf,
        '.docx': process_doc,
        '.pptx': process_ppt
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
        parser = argparse.ArgumentParser(description="python tomd.py file|dir ...")
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
