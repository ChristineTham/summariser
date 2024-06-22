import sys
import os
from markdownify import markdownify
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

def main():
    # Check if the user provided a file name as an argument
    if len(sys.argv) < 2:
        print("Please provide the filename of an HTML page.")
        return

    filename = sys.argv[1]
    
    llm = Ollama(model="llama3:8b-instruct-fp16", temperature=0.3, num_ctx=8192)
    
    prompt = PromptTemplate.from_template(
        "You are an efficient text summarizer. Summarize the following Markdown document, delimited by <document> and </document>, in bullet points, without any preamble, in Markdown format. Group the bullet points underneath any headings you encounter. Do not add any material not in the document. Document: <document>{context}</document>. Summary:"
    )
    chain = prompt | llm
    
    with open(filename, 'r') as file:
        html = file.read()

        markdown = markdownify(html, default_title=True, heading_style="ATX")
        print(chain.invoke({"context": markdown}))

if __name__ == "__main__":
    main()
