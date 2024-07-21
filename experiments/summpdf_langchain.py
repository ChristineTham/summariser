import sys
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

def main():
    # Check if the user provided a file name as an argument
    if len(sys.argv) < 2:
        print("Please provide a filename of a PDF document.")
        return

    filename = sys.argv[1]

    # Check if the file exists
    if not os.path.isfile(filename):
        print("The file does not exist.")
        return

    loader = PyPDFLoader(filename)

    docs = loader.load()

    # print("Filename: ", filename)
    # print("#pages: ", len(docs))
    
    llm = Ollama(model="llama3:8b-instruct-fp16", temperature=0.3, num_ctx=8192)
    
    prompt = PromptTemplate.from_template(
        "You are an efficient text summarizer. Summarize the following page from a longer document, delimited by <page> and </page>, in bullet points, without any preamble, in markdown format. Ignore any page header or footer. Group the bullet points underneath any headings you encounter. Do not add any material not in the document. If there is nothing to summarize, do not add any bullet points. Page: <page>{context}</page>. Summary:"
    )
    chain = prompt | llm

    for i, doc in enumerate(docs):
        print(chain.invoke({"context": doc.page_content}))
        # print("Page ", i, ": ", doc.page_content)

if __name__ == "__main__":
    main()
