import os
import base64
import argparse
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

def pdf2md(file):
    # determine if file is an URL
    if file.startswith("http"):
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": file
            },
            include_image_base64=True
        )
    else:
        pdf = client.files.upload(
            file={
                "file_name": file,
                "content": open(file, "rb"),
            },
            purpose="ocr"
        )
        signed_url = client.files.get_signed_url(file_id=pdf.id)
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url
            },
            include_image_base64=True
        )
        client.files.delete(file_id=pdf.id)

    # print(ocr_response)

    # output file is same as input file but extension is changed to .md
    output_file = os.path.splitext(os.path.basename(file))[0] + ".md"
    print(f"Writing to [{output_file}]")

    # iterate through ocr_response write to file as JSON
    with open(output_file, "w") as f:
        # iterate through ocr_response
        for page in ocr_response.pages:
            f.write(page.markdown)
            f.write("\n")

    # write images from ocr_response
    for page in ocr_response.pages:
        for image in page.images:
            image_data = base64.b64decode(image.image_base64)
            image_name = image.id
            print(f"Writing to [{image_name}]")
            with open(image_name, "wb") as f:
                f.write(image_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown using Mistral OCR")
    parser.add_argument("file", nargs="+", help="URL or PDF file")
    args = parser.parse_args()
    for file in args.file:
        print(f"Processing: {file}")
        pdf2md(file)
