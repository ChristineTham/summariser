import sys
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

def main():
    # Check if the user provided a file name as an argument
    if len(sys.argv) < 2:
        print("Please provide the URL of a Youtube video.")
        return

    url = sys.argv[1]

    docs = YoutubeLoader.from_youtube_url(url, add_video_info=True).load()
    print(docs[0].page_content[:500])
    
    llm = Ollama(model="llama3:70b", temperature=0.3, num_ctx=8192)
    
    prompt = PromptTemplate.from_template(
        "You are an efficient text summarizer. Provide a summary in bullet points on the document, delimited by <document> and </document>. Do not provide a preamble. Document: <document>{context}</document>. Summary:"
    )
    chain = create_stuff_documents_chain(llm, prompt)

    summary = chain.invoke({"context": docs})

    print(summary)

if __name__ == "__main__":
    main()
