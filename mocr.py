import os
import json
import argparse
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

def pdf2md(url):
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": url
        },
        include_image_base64=True
    )

    # print(ocr_response)

    # output file is same as input file but extension is changed to .md
    output_file = os.path.splitext(os.path.basename(url))[0] + ".md"
    print(f"Writing to [{output_file}]")

    # iterate through ocr_response write to file as JSON
    with open(output_file, "w") as f:
        # iterate through ocr_response
        for page in ocr_response.pages:
            f.write(page.markdown)
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown using Mistral OCR")
    parser.add_argument("url", nargs="+", help="HTTP URL to PDF file")
    args = parser.parse_args()
    for url in args.url:
        pdf2md(url)
