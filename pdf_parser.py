import fitz  # PyMuPDF
import base64
import io
import re
from typing import Optional


def extract_text_from_pdf(pdf_file) -> str:
    """Extract all text from a PDF file object."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def extract_text_with_pages(pdf_bytes: bytes) -> dict:
    """
    Extract text from PDF with page numbers.
    Returns: {page_num: text_content}
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = {}
    for page_num, page in enumerate(doc):
        pages[page_num + 1] = page.get_text()
    doc.close()
    return pages


def find_text_location(pdf_bytes: bytes, search_text: str) -> list:
    """
    Find which pages contain a specific text snippet.
    Returns list of (page_num, rect) tuples.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    locations = []
    
    # Clean search text
    clean_search = " ".join(search_text.split())[:100]  # First 100 chars
    
    for page_num, page in enumerate(doc):
        instances = page.search_for(clean_search)
        for rect in instances:
            locations.append((page_num + 1, rect))
    
    doc.close()
    return locations


def highlight_text_in_pdf(pdf_bytes: bytes, text_to_highlight: str, 
                           highlight_color: tuple = (1, 0.9, 0.2)) -> bytes:
    """
    Highlight specific text in the PDF and return modified PDF bytes.
    highlight_color is RGB tuple (0-1 range), default is yellow.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Try different lengths of the search text
    search_variants = [
        text_to_highlight[:150].strip(),
        text_to_highlight[:100].strip(),
        text_to_highlight[:50].strip(),
        " ".join(text_to_highlight.split()[:15]),  # First 15 words
    ]
    
    highlighted = False
    for search_text in search_variants:
        if len(search_text) < 10:
            continue
        for page in doc:
            instances = page.search_for(search_text)
            if instances:
                for rect in instances:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=highlight_color)
                    annot.update()
                highlighted = True
        if highlighted:
            break
    
    result_bytes = doc.write()
    doc.close()
    return result_bytes


def get_pdf_page_as_image(pdf_bytes: bytes, page_num: int, zoom: float = 1.5) -> str:
    """
    Render a PDF page as a base64-encoded PNG image.
    page_num is 1-indexed.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    if page_num < 1 or page_num > len(doc):
        doc.close()
        return ""
    
    page = doc[page_num - 1]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    
    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_references_from_text(text: str) -> list:
    """
    Extract reference snippets from the paper text that can be used 
    as hyperlinks back to the original PDF.
    Returns list of short reference strings.
    """
    # Find sentences that look like key content (not headers, not too short)
    lines = text.split('\n')
    references = []
    
    for line in lines:
        line = line.strip()
        if len(line) > 50 and len(line) < 300:
            # Filter out headers and noise
            if not re.match(r'^\d+\.?\s*$', line):  # Not just a number
                if not re.match(r'^[A-Z\s]{2,}$', line):  # Not all caps header
                    references.append(line[:100])
    
    return references[:50]  # Return top 50 candidate references


def get_total_pages(pdf_bytes: bytes) -> int:
    """Get total number of pages in a PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    doc.close()
    return total