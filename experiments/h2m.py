import os
import argparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

llm = Ollama(model="llama3:8b-instruct-fp16", temperature=0.3, num_ctx=8192)

prompt = PromptTemplate.from_template(
    "You are an efficient text summarizer. Summarize the following Markdown document, delimited by <document> and </document>, in bullet points, without any preamble, in Markdown format. Group the bullet points underneath any headings you encounter. Do not add any material not in the document. Document: <document>{context}</document>. Summary:"
)

chain = prompt | llm

def convert_html_to_md(input_file):
    with open(input_file, 'r') as file:
        data = file.read()
    print("Converting from HTML ", input_file)
    soup = BeautifulSoup(data, "html.parser")
    markdown = md(str(soup), default_title=True, heading_style="ATX")

    base = os.path.splitext(input_file)[0]
    markdown_file = f"{base}.md"
    with open(markdown_file, 'w') as file:
        file.write(markdown)
    print(f'Converted to Markdown {markdown_file}')
    
    summ = chain.invoke({"context": markdown})
    summary_file = f"{base}-summ.md"
    with open(summary_file, 'w') as file:
        file.write(summ)
    print(f'Converted to Markdown summary {summary_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("html_files", nargs='+', help="HTML files to convert")
    args = parser.parse_args()

    for html_file in args.html_files:
        if os.path.isfile(html_file):
            convert_html_to_md(html_file)
        else:
            print(f'The file {html_file} does not exist')
