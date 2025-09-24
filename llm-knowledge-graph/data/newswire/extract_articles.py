"""Extract text from locally stored PDF articles into a CSV file.

The script reads every PDF within ``llm-knowledge-graph/data/newswire/pdfs``
and writes their contents to ``articles.csv`` with the following columns:

- id: stem of the PDF filename
- date: file modification date (YYYY-MM-DD)
- text: concatenated text extracted from all pages
- newspapers: left empty for manual enrichment later
"""

import csv
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

DATA_PATH = Path("llm-knowledge-graph/data/newswire")
PDF_PATH = DATA_PATH / "pdfs"
ARTICLE_FILENAME = DATA_PATH / "articles.csv"


def extract_text_from_pdf(pdf_file: Path) -> str:
    """Return concatenated text from all pages of the PDF."""

    reader = PdfReader(str(pdf_file))
    page_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            page_text.append(text)

    return "\n\n".join(page_text)


def main() -> None:
    pdf_files = sorted(PDF_PATH.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_PATH}")

    with ARTICLE_FILENAME.open("w", encoding="utf8", newline="") as csvfile:
        fieldnames = ["id", "date", "text", "newspapers"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for pdf_file in pdf_files:
            article_id = pdf_file.stem
            print(f"Processing {article_id}")

            text = extract_text_from_pdf(pdf_file)
            if not text:
                print(f"Warning: no text extracted from {pdf_file}")

            modified = datetime.fromtimestamp(pdf_file.stat().st_mtime).strftime("%Y-%m-%d")

            writer.writerow(
                {
                    "id": article_id,
                    "date": modified,
                    "text": text,
                    "newspapers": "",
                }
            )


if __name__ == "__main__":
    main()
