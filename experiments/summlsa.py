import sys
import os
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer     
from sumy.utils import get_stop_words
import yake

LANGUAGE = "english"
SENTENCES_COUNT = 10

def main():
    # Check if the user provided a file name as an argument
    if len(sys.argv) < 2:
        print("Please provide a filename containing text or Markdown.")
        return

    filename = sys.argv[1]

    # Check if the file exists
    if not os.path.isfile(filename):
        print("The file does not exist.")
        return

    with open(filename, 'r') as file:
        markdown = file.read()

    # print("Length: ", len(markdown))
    
    # Extract keywords using YAKE
    print("## Keywords\n")
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(markdown)

    for kw in keywords:
        print(f"* [[{kw[0]}]]")    
    
    # Summarize using sumy LSA       
    print("\n## Abstract\n")    
    parser = PlaintextParser.from_string(markdown, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = LsaSummarizer(stemmer)     
    summarizer.stop_words = get_stop_words(LANGUAGE)                  
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(f"* {sentence}")      

if __name__ == "__main__":
    main()
