import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Section keyword map — no GPT calls needed
SECTION_KEYWORDS = {
    "topic":             ["abstract", "introduction", "overview", "this paper", "we propose", "we present"],
    "motivation":        ["motivation", "problem statement", "challenge", "existing methods fail", "limitation of"],
    "literature_review": ["related work", "literature review", "prior work", "previous work", "survey", "background"],
    "dataset":           ["dataset", "data collection", "corpus", "training data", "we collected", "samples", "annotated"],
    "methodology":       ["methodology", "method", "approach", "architecture", "model", "framework", "algorithm", "pipeline"],
    "results":           ["results", "evaluation", "experiment", "performance", "accuracy", "f1", "precision", "recall", "score", "baseline"],
    "insights":          ["analysis", "discussion", "insight", "finding", "observation", "we find"],
    "limitations":       ["limitation", "future work", "shortcoming", "weakness", "cannot", "fails to"],
    "future_research":   ["future work", "future direction", "open problem", "extension", "can be extended"],
}


def generate_references_for_sections(explanations: dict, full_text: str) -> dict:
    """
    Fast keyword-based extraction of reference phrases per section.
    No GPT calls — runs instantly.
    """
    references = {}
    text_lower = full_text.lower()

    for section_key, keywords in SECTION_KEYWORDS.items():
        found_phrases = []
        for kw in keywords:
            # Find keyword in text and extract surrounding sentence
            idx = text_lower.find(kw)
            if idx != -1:
                # Extract a window around the keyword
                start = max(0, idx - 20)
                end = min(len(full_text), idx + 120)
                snippet = full_text[start:end].strip()
                # Clean to sentence boundary
                snippet = re.split(r'[.\n]', snippet)[0].strip()
                if len(snippet) > 15:
                    found_phrases.append(snippet[:100])
        references[section_key] = found_phrases[:3]

    return references


def build_section_page_map(references: dict, page_texts: dict) -> dict:
    """
    Map each section to its most likely page number using keyword search.
    Fast text search — no API calls.
    """
    section_page_map = {}

    for section_key, phrases in references.items():
        best_page = 0
        keywords = SECTION_KEYWORDS.get(section_key, [])

        # First try the extracted phrases
        for phrase in phrases:
            page = _find_page_for_text(phrase.lower(), page_texts)
            if page > 0:
                best_page = page
                break

        # Fallback: search by section keywords directly
        if best_page == 0:
            for kw in keywords:
                page = _find_page_for_text(kw, page_texts)
                if page > 0:
                    best_page = page
                    break

        section_page_map[section_key] = best_page

    return section_page_map


def _find_page_for_text(search_text: str, page_texts: dict) -> int:
    """Find the first page containing the search text. Returns 0 if not found."""
    search_lower = search_text.lower().strip()
    for page_num, text in page_texts.items():
        if search_lower in text.lower():
            return page_num
    return 0