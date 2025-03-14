import os
import json
import sys
from mistralai import Mistral

# Check if filename was provided as command line argument
if len(sys.argv) < 2:
    print("Usage: python mocr.py <document_url>")
    sys.exit(1)

# Get the document URL from command line argument
document_url = sys.argv[1]

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": document_url
    },
    include_image_base64=True
)

# print(ocr_response)

# output file is same as input file but extension is changed to .md
output_file = os.path.splitext(os.path.basename(document_url))[0] + ".md"
print(f"Writing to {output_file}")

# iterate through ocr_response write to file as JSON
with open(output_file, "w") as f:
    # iterate through ocr_response
    for page in ocr_response.pages:
        f.write(page.markdown)
        f.write("\n")
