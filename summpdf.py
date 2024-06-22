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
    
    llm = Ollama(model="llama3:70b", temperature=0.3, num_ctx=8192)
    
    prompt = PromptTemplate.from_template(
        "You are an efficient text heading extractor. Your task is to extract headings from the following text, delimited by <document> and </document>. Each heading should be delimited by <heading> and </heading>. Do not provide explanations or preamble. Text: <document>{context}</document>"
    )
    chain = create_stuff_documents_chain(llm, prompt)

    headings = chain.invoke({"context": docs})
    # print(headings)
    
    prompt = PromptTemplate.from_template(
        "You are an efficient text summarizer. For each heading delimited by <heading> and </heading>, provide a summary in bullet points on the document, delimited by <document> and </document>. Do not provide a preamble. Headings: {headings}. Document: <document>{context}</document>. Summary:"
    )
    chain = create_stuff_documents_chain(llm, prompt)

    summary = chain.invoke({"context": docs, "headings": headings})

    print(summary)

if __name__ == "__main__":
    main()
