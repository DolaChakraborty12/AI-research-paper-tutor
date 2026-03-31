from openai import OpenAI

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY
from modules.embedding_store import search
 
client = OpenAI(api_key=OPENAI_API_KEY)

# Conversation history for multi-turn chat
_chat_history = []


def reset_chat():
    """Reset the chat history."""
    global _chat_history
    _chat_history = []


def answer_question(question: str, paper_context: str = "") -> str:
    """
    Answer a user's question about the research paper.
    Uses RAG (retrieved chunks) + full paper context for best answers.
    Maintains conversation history for follow-up questions.
    """
    global _chat_history
    
    # Retrieve relevant chunks using embeddings
    try:
        relevant_chunks = search(question)
        retrieved_context = "\n\n---\n\n".join(relevant_chunks)
    except Exception:
        retrieved_context = paper_context[:3000] if paper_context else ""
    
    system_prompt = """You are Professor Alex, an expert researcher and passionate educator.
    
A user is asking you questions about a research paper they're studying.

Your teaching style:
- Answer like an experienced researcher — warm, thorough, patient
- Break down complex concepts into simple steps
- Use analogies and real-world examples
- If the student seems confused, re-explain from basics
- If the question is unclear, answer what you think they meant and clarify
- Always connect your answer back to the research paper's context
- End with a follow-up thought or question to deepen understanding

IMPORTANT: Never say "I don't know" — use the context provided and your knowledge to give the best answer possible.
"""
    
    user_message = f"""Based on this research paper content:

RELEVANT SECTIONS:
{retrieved_context}

STUDENT QUESTION: {question}

Please explain this clearly like a professor teaching a student."""
    
    # Add to history
    _chat_history.append({"role": "user", "content": user_message})
    
    # Keep history manageable (last 10 exchanges)
    history_to_send = _chat_history[-20:]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            *history_to_send
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content
    
    # Store assistant response in history
    _chat_history.append({"role": "assistant", "content": answer})
    
    return answer


def get_chat_history() -> list:
    """Return the current chat history."""
    return _chat_history


def generate_quiz_question(paper_text: str) -> str:
    """Generate a quiz question to test user understanding."""
    prompt = f"""Based on this research paper, generate ONE insightful quiz question 
    that tests deep understanding (not just memorization). 
    Make it thought-provoking. Include the answer after 'ANSWER:'.
    
    Paper: {paper_text[:3000]}"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professor creating exam questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    
    return response.choices[0].message.content