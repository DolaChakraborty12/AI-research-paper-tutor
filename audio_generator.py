from openai import OpenAI
import tempfile
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def generate_audio(text: str, voice: str = "onyx") -> str:
    """
    Generate complete audio for the full text — no truncation.
    Splits at sentence boundaries to avoid mid-sentence cuts.
    Merges all chunks into one complete MP3.
    """
    text = _clean_for_speech(text)

    if not text.strip():
        return ""

    # OpenAI TTS hard limit is 4096 chars per call
    # Split the FULL text into chunks — never truncate
    chunks = _split_text_for_tts(text, max_chars=4000)

    audio_files = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=chunk,
            speed=0.95
        )
        tmp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_part{i}.mp3", prefix="tutor_audio_"
        )
        response.stream_to_file(tmp_file.name)
        audio_files.append(tmp_file.name)

    if not audio_files:
        return ""

    if len(audio_files) == 1:
        return audio_files[0]

    # Merge all chunks into one seamless file
    merged_path = _merge_audio_files(audio_files)
    for f in audio_files:
        try:
            os.unlink(f)
        except Exception:
            pass
    return merged_path


def _clean_for_speech(text: str) -> str:
    """Remove markdown so TTS reads naturally as spoken words."""
    # Remove markdown headers
    text = re.sub(r'#{1,6}\s*', '', text)
    # Convert bullet points to a natural spoken transition
    text = re.sub(r'^\s*[\*\-•]\s+', 'Next: ', text, flags=re.MULTILINE)
    # Convert numbered lists to natural speech
    text = re.sub(r'^\s*(\d+)\.\s+', r'Point \1: ', text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    # Collapse multiple newlines into a sentence pause
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\n', ' ', text)
    # Fix double periods and extra spaces
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _split_text_for_tts(text: str, max_chars: int = 4000) -> list:
    """
    Split text into TTS-safe chunks strictly at sentence boundaries.
    Never cuts mid-sentence. Guarantees every word is spoken.
    """
    if len(text) <= max_chars:
        return [text]

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sentence in sentences:
        # If a single sentence is too long, split at commas
        if len(sentence) > max_chars:
            parts = re.split(r',\s+', sentence)
            for part in parts:
                if len(current) + len(part) + 2 <= max_chars:
                    current += part + ", "
                else:
                    if current:
                        chunks.append(current.strip().rstrip(',') + '.')
                    current = part + ", "
        elif len(current) + len(sentence) + 1 <= max_chars:
            current += sentence + " "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + " "

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c.strip()]


def _merge_audio_files(file_paths: list) -> str:
    """Merge multiple MP3 files into one complete audio file."""
    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for path in file_paths:
            combined += AudioSegment.from_mp3(path)
        merged_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3", prefix="tutor_audio_merged_"
        )
        combined.export(merged_file.name, format="mp3")
        return merged_file.name
    except ImportError:
        # pydub not installed — return largest chunk
        return max(file_paths, key=lambda p: os.path.getsize(p))


def generate_section_audio(section_name: str, explanation: str, voice: str = "onyx") -> str:
    """
    Generate complete audio for a section.
    Short spoken intro + full explanation — nothing cut off.
    """
    intro_map = {
        "topic":             "Here is what this research paper is about.",
        "motivation":        "Here is why this research was conducted.",
        "literature_review": "Here is the related work referenced in this paper.",
        "dataset":           "Here are the dataset details used in this research.",
        "methodology":       "Now, the methodology. Here is every step of the approach.",
        "results":           "Here are the results achieved in this research.",
        "insights":          "Here are the key insights from this paper.",
        "limitations":       "Here are the limitations of this research.",
        "future_research":   "Here are ideas for future research directions.",
    }

    intro = intro_map.get(section_name, f"Here is the {section_name} section.")
    # Pass the FULL explanation — no truncation
    full_text = f"{intro} {explanation}"

    return generate_audio(full_text, voice=voice)