import streamlit as st
import os
import json
import tempfile
import base64
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Page Config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="AI Research Paper Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Environment Setup ────────────────────────────────────────────────────────
from config import OPENAI_API_KEY
 


# ─── Module Imports ───────────────────────────────────────────────────────────
from modules.pdf_parser import (
    extract_text_with_pages, highlight_text_in_pdf,
    get_pdf_page_as_image, get_total_pages
)
from modules.chunking import chunk_text
from modules.embedding_store import create_embeddings, search
from modules.tutor_explainer import tutor_explain_full, tutor_explain_section
from modules.audio_generator import generate_section_audio
from modules.chatbot import answer_question, reset_chat, get_chat_history
from modules.reference_extractor import (
    generate_references_for_sections, build_section_page_map
)
 
# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;600&display=swap');
 
  :root {
    --ink: #1a1a2e;
    --chalk: #f5f0e8;
    --accent: #c84b31;
    --gold: #d4a853;
    --sage: #4a7c59;
    --board: #1e3a2f;
  }
 
  html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
  }
 
  h1, h2, h3 { font-family: 'Playfair Display', serif; }
 
  .main-header {
    background: linear-gradient(135deg, var(--board) 0%, #0f2318 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 5px solid var(--gold);
    position: relative;
    overflow: hidden;
  }
  .main-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 28px,
      rgba(255,255,255,0.04) 28px, rgba(255,255,255,0.04) 29px
    );
  }
  .main-header h1 {
    color: #f0e6c8; font-size: 2rem; margin: 0; position: relative;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
  }
  .main-header p {
    color: #a8c5a0; margin: 0.3rem 0 0; position: relative; font-size: 1rem;
  }
 
  .section-card {
    background: white;
    border: 1px solid #e8e0d0;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-left: 4px solid var(--accent);
  }
  .section-card h3 {
    color: var(--ink); margin: 0 0 0.8rem; font-size: 1.1rem;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .section-card p { color: #444; line-height: 1.7; margin: 0; font-size: 0.95rem; }
 
  .ref-link {
    display: inline-block;
    background: #fff3e0;
    border: 1px solid var(--gold);
    color: #8b5e00;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    cursor: pointer;
    text-decoration: none;
    margin-left: 0.4rem;
    transition: all 0.2s;
  }
  .ref-link:hover { background: var(--gold); color: white; }
 
  .chat-bubble-user {
    background: linear-gradient(135deg, #e8f4fd, #d0e9f8);
    border: 1px solid #b8d9f0;
    border-radius: 12px 12px 4px 12px;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 0 0.5rem 2rem;
    color: #1a3a5c;
    font-size: 0.93rem;
  }
  .chat-bubble-ai {
    background: linear-gradient(135deg, #f0f8f0, #e8f5e9);
    border: 1px solid #b8ddb8;
    border-radius: 12px 12px 12px 4px;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 2rem 0.5rem 0;
    color: #1a3a1a;
    font-size: 0.93rem;
    line-height: 1.65;
  }
  .professor-label {
    font-size: 0.75rem; font-weight: 600; color: var(--sage);
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem;
  }
  .student-label {
    font-size: 0.75rem; font-weight: 600; color: #2c6fad;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem;
    text-align: right;
  }
 
  .upload-zone {
    background: linear-gradient(135deg, #f8f4ee, #fdf8f2);
    border: 2px dashed var(--gold);
    border-radius: 14px;
    padding: 3rem 2rem;
    text-align: center;
    transition: all 0.3s;
  }
  .upload-zone:hover { border-color: var(--accent); background: #fff9f5; }
 
  .progress-step {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.5rem 0; color: #666; font-size: 0.9rem;
  }
  .progress-step.done { color: var(--sage); font-weight: 600; }
  .progress-step.active { color: var(--accent); font-weight: 600; }
 
  .audio-section {
    background: linear-gradient(135deg, #1e3a2f, #0f2318);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.7rem 0;
  }
  .audio-section p { color: #a8c5a0; margin: 0 0 0.5rem; font-size: 0.85rem; font-weight: 600; }
 
  .pdf-viewer-container {
    border: 1px solid #ddd;
    border-radius: 8px;
    overflow: hidden;
  }
 
  .highlight-badge {
    display: inline-block;
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.78rem;
    color: #664d03;
    margin: 0 2px;
  }
 
  .stButton > button {
    background: linear-gradient(135deg, var(--accent), #a03825);
    color: white; border: none; border-radius: 8px;
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 600; padding: 0.55rem 1.4rem;
    transition: all 0.2s; box-shadow: 0 2px 6px rgba(200,75,49,0.3);
  }
  .stButton > button:hover {
    transform: translateY(-1px); box-shadow: 0 4px 12px rgba(200,75,49,0.4);
  }
 
  .sidebar .stButton > button {
    background: linear-gradient(135deg, var(--board), #0f2318);
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
  }
 
  div[data-testid="stExpander"] {
    border: 1px solid #e0d8cc; border-radius: 10px;
    overflow: hidden; margin: 0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)
 
# ─── Session State Init ───────────────────────────────────────────────────────
for key, default in {
    "pdf_bytes": None,
    "pdf_name": "",
    "full_text": "",
    "page_texts": {},
    "explanations": {},
    "references": {},
    "section_page_map": {},
    "audio_paths": {},
    "processing_done": False,
    "chat_messages": [],
    "current_pdf_view_page": 1,
    "highlighted_pdf_bytes": None,
    "embeddings_created": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
 
# ─── Constants ────────────────────────────────────────────────────────────────
SECTIONS = {
    "topic":           ("🔬", "Research Topic & Problem"),
    "motivation":      ("💡", "Motivation & Research Gap"),
    "literature_review": ("📚", "Literature Review"),
    "dataset":         ("🗄️", "Dataset Details"),
    "methodology":     ("⚙️", "Methodology & Approach"),
    "results":         ("📊", "Results & Evaluation"),
    "insights":        ("🔍", "Key Insights"),
    "limitations":     ("⚠️", "Limitations"),
    "future_research": ("🚀", "Future Research Ideas (AI Perspective)"),
}
 
# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎓 AI Research Paper Tutor</h1>
  <p>Upload a research paper and let Professor Alex explain it like a classroom lecture</p>
</div>
""", unsafe_allow_html=True)
 
# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Session Controls")
    
    if st.button("🔄 Upload New Paper"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        reset_chat()
        st.rerun()
    
    if st.session_state.processing_done:
        st.markdown("---")
        st.markdown("### 📄 PDF Viewer")
        total_pages = get_total_pages(st.session_state.pdf_bytes)
        st.markdown(f"**{st.session_state.pdf_name}** ({total_pages} pages)")
        
        page_num = st.number_input(
            "Jump to page", min_value=1, max_value=total_pages,
            value=st.session_state.current_pdf_view_page
        )
        
        if st.button("📖 Show Page"):
            st.session_state.current_pdf_view_page = page_num
        
        # Render PDF page as image
        pdf_to_show = (
            st.session_state.highlighted_pdf_bytes 
            if st.session_state.highlighted_pdf_bytes 
            else st.session_state.pdf_bytes
        )
        
        img_b64 = get_pdf_page_as_image(pdf_to_show, st.session_state.current_pdf_view_page)
        if img_b64:
            st.markdown(f"""
            <div class="pdf-viewer-container">
              <img src="data:image/png;base64,{img_b64}" style="width:100%; display:block;"/>
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.highlighted_pdf_bytes:
            st.download_button(
                "⬇️ Download Highlighted PDF",
                data=st.session_state.highlighted_pdf_bytes,
                file_name=f"highlighted_{st.session_state.pdf_name}",
                mime="application/pdf"
            )
        
        st.markdown("---")
        st.markdown("### 🗺️ Section Reference Map")
        for sk, page in st.session_state.section_page_map.items():
            icon, label = SECTIONS.get(sk, ("📌", sk))
            pg_str = f"p.{page}" if page > 0 else "not found"
            st.markdown(f"{icon} **{label[:25]}** — {pg_str}")
 
# ─── Main Content ─────────────────────────────────────────────────────────────
 
# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Upload
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.processing_done:
    st.markdown("""
    <div class="upload-zone">
      <div style="font-size:3rem;">📄</div>
      <h2 style="color:#3d2b1f; margin:0.5rem 0;">Upload Your Research Paper</h2>
      <p style="color:#7a6652;">PDF format only · Professor Alex will explain it section by section with voice</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "", type=["pdf"], label_visibility="collapsed"
    )
    
    if uploaded_file:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.success(f"✅ **{uploaded_file.name}** uploaded successfully!")
            
            voice_choice = st.selectbox(
                "🎙️ Professor's Voice",
                ["onyx", "echo", "fable", "nova", "shimmer"],
                index=0,
                help="onyx = deep authoritative | echo = balanced | fable = warm | nova = clear | shimmer = gentle"
            )
            
            generate_audio_flag = st.checkbox(
                "🔊 Generate voice explanations (takes ~2-3 min extra)", 
                value=True
            )
            
            if st.button("🎓 Start Lecture!", use_container_width=True):
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_bytes = uploaded_file.read()
 
                prog_bar = st.progress(0)
                status_msg = st.empty()
 
                # ── Step 1: Parse PDF (fast, ~1s) ──
                status_msg.markdown("**📖 Parsing PDF...**")
                page_texts = extract_text_with_pages(st.session_state.pdf_bytes)
                st.session_state.page_texts = page_texts
                full_text = "\n".join(page_texts.values())
                st.session_state.full_text = full_text
                prog_bar.progress(10)
 
                # ── Step 2: Embeddings + All Section Explanations IN PARALLEL ──
                status_msg.markdown("**⚡ Running all sections in parallel...**")
 
                section_keys = list(SECTIONS.keys())
 
                def explain_section(sk):
                    return sk, tutor_explain_section(sk, full_text)
 
                def build_embeddings():
                    chunks = chunk_text(full_text, chunk_size=400)
                    create_embeddings(chunks)
                    return True
 
                explanations = {}
                completed = 0
                total_tasks = len(section_keys) + 1  # +1 for embeddings
 
                with ThreadPoolExecutor(max_workers=6) as executor:
                    futures = {executor.submit(explain_section, sk): sk for sk in section_keys}
                    futures[executor.submit(build_embeddings)] = "__embeddings__"
 
                    for future in as_completed(futures):
                        task_key = futures[future]
                        try:
                            result = future.result()
                            if task_key == "__embeddings__":
                                st.session_state.embeddings_created = True
                            else:
                                sk, explanation = result
                                explanations[sk] = explanation
                                completed += 1
                                icon, label = SECTIONS[sk]
                                status_msg.markdown(f"**✅ Done: {label}** ({completed}/{len(section_keys)})")
                        except Exception as e:
                            if task_key != "__embeddings__":
                                explanations[task_key] = f"Could not generate explanation: {str(e)}"
                        prog_bar.progress(10 + int((completed / total_tasks) * 55))
 
                st.session_state.explanations = explanations
                prog_bar.progress(65)
 
                # ── Step 3: Reference mapping (fast, ~5s) ──
                status_msg.markdown("**🔗 Mapping references to PDF pages...**")
                references = generate_references_for_sections(explanations, full_text)
                st.session_state.references = references
                section_page_map = build_section_page_map(references, page_texts)
                st.session_state.section_page_map = section_page_map
                prog_bar.progress(75)
 
                # ── Step 4: Audio IN PARALLEL (optional) ──
                if generate_audio_flag:
                    status_msg.markdown("**🎙️ Generating all audio tracks in parallel...**")
 
                    def gen_audio(sk_exp):
                        sk, explanation = sk_exp
                        try:
                            path = generate_section_audio(sk, explanation, voice=voice_choice)
                            return sk, path
                        except Exception:
                            return sk, None
 
                    audio_paths = {}
                    audio_completed = 0
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        audio_futures = {
                            executor.submit(gen_audio, (sk, exp)): sk
                            for sk, exp in explanations.items()
                        }
                        for future in as_completed(audio_futures):
                            sk, path = future.result()
                            audio_paths[sk] = path
                            audio_completed += 1
                            prog_bar.progress(75 + int((audio_completed / len(explanations)) * 22))
                            status_msg.markdown(
                                f"**🎙️ Audio recorded: {audio_completed}/{len(explanations)}**"
                            )
 
                    st.session_state.audio_paths = audio_paths
 
                prog_bar.progress(100)
                status_msg.markdown("**✅ Lecture ready!**")
                time.sleep(0.5)
 
                st.session_state.processing_done = True
                st.rerun()
 
# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Show Lecture + Chat
# ══════════════════════════════════════════════════════════════════════════════
else:
    tab1, tab2 = st.tabs(["🎓 Professor's Lecture", "💬 Ask Professor Alex"])
    
    # ── TAB 1: Lecture ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown(f"""
        <div style="background:#f8f4ee; border-radius:10px; padding:1rem 1.5rem; 
                    border-left:4px solid #4a7c59; margin-bottom:1.5rem;">
          <strong>📄 Now explaining:</strong> {st.session_state.pdf_name}<br>
          <span style="color:#666; font-size:0.9rem;">
            Click <span class="highlight-badge">📌 pg.X</span> badges to jump to that page in the PDF viewer (sidebar)
          </span>
        </div>
        """, unsafe_allow_html=True)
        
        for sk, (icon, label) in SECTIONS.items():
            explanation = st.session_state.explanations.get(sk, "")
            ref_page = st.session_state.section_page_map.get(sk, 0)
            ref_phrases = st.session_state.references.get(sk, [])
            
            # Reference link
            ref_html = ""
            if ref_page > 0:
                ref_html = f'<a class="ref-link" href="#" onclick="void(0)">📌 pg.{ref_page}</a>'
            
            with st.expander(f"{icon}  {label}", expanded=(sk == "topic")):
                # Reference badge + highlight button
                col_ref, col_hl = st.columns([4, 1])
                with col_ref:
                    if ref_page > 0:
                        st.markdown(
                            f"📌 **Reference:** Page {ref_page} in the paper",
                            help="Click 'Highlight in PDF' to mark this section"
                        )
                with col_hl:
                    if ref_page > 0 and ref_phrases:
                        if st.button(f"🖊️ Highlight", key=f"hl_{sk}"):
                            # Highlight the relevant text and jump to page
                            highlighted = st.session_state.pdf_bytes
                            for phrase in ref_phrases[:2]:
                                try:
                                    highlighted = highlight_text_in_pdf(
                                        highlighted, phrase,
                                        highlight_color=(1.0, 0.9, 0.1)
                                    )
                                except Exception:
                                    pass
                            st.session_state.highlighted_pdf_bytes = highlighted
                            st.session_state.current_pdf_view_page = ref_page
                            st.success(f"✅ Highlighted! See page {ref_page} in sidebar PDF viewer.")
                
                # Explanation text
                st.markdown(f"""
                <div class="section-card">
                  <p>{explanation.replace(chr(10), '<br>')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Audio player
                audio_path = st.session_state.audio_paths.get(sk)
                if audio_path and os.path.exists(audio_path):
                    st.markdown("""
                    <div class="audio-section">
                      <p>🎙️ PROFESSOR ALEX — VOICE EXPLANATION</p>
                    </div>
                    """, unsafe_allow_html=True)
                    with open(audio_path, "rb") as af:
                        st.audio(af.read(), format="audio/mp3")
                
                # Show reference phrases
                if ref_phrases:
                    with st.expander("🔍 Source phrases from paper", expanded=False):
                        for phrase in ref_phrases[:3]:
                            st.markdown(f"> _{phrase}_")
    
    # ── TAB 2: Chat ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1e3a2f,#0f2318); 
                    border-radius:10px; padding:1rem 1.5rem; margin-bottom:1rem; 
                    border-left:4px solid #d4a853;">
          <span style="color:#a8c5a0; font-size:0.95rem;">
            💬 <strong style="color:#f0e6c8;">Ask Professor Alex</strong> anything about this paper — 
            follow-up questions, concept clarifications, or deeper explanations
          </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat history display
        chat_display = st.container()
        with chat_display:
            if not st.session_state.chat_messages:
                st.markdown("""
                <div class="chat-bubble-ai">
                  <div class="professor-label">Professor Alex</div>
                  Welcome! I've just finished explaining this paper. What would you like to 
                  understand better? You can ask me about any aspect — the methodology, 
                  the results, the dataset, or even ask me to explain concepts from scratch. 🎓
                </div>
                """, unsafe_allow_html=True)
            
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="student-label">You</div>
                    <div class="chat-bubble-user">{msg["content"]}</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="professor-label">Professor Alex</div>
                    <div class="chat-bubble-ai">{msg["content"].replace(chr(10), "<br>")}</div>
                    """, unsafe_allow_html=True)
        
        # Input
        st.markdown("---")
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_question = st.text_input(
                "Your question:",
                placeholder="e.g. 'Can you explain the attention mechanism used in more detail?'",
                label_visibility="collapsed",
                key="chat_input"
            )
        with col_send:
            send_clicked = st.button("Send 📨", use_container_width=True)
        
        # Suggestion chips
        st.markdown("**Quick questions:**")
        q_cols = st.columns(3)
        suggestions = [
            "Explain the methodology step by step",
            "What are the main limitations?",
            "How does this compare to other approaches?",
            "Can you explain the evaluation metrics?",
            "What dataset was used and why?",
            "Give me future research ideas",
        ]
        for i, sugg in enumerate(suggestions):
            with q_cols[i % 3]:
                if st.button(f"💬 {sugg}", key=f"sugg_{i}", use_container_width=True):
                    user_question = sugg
                    send_clicked = True
        
        if send_clicked and user_question.strip():
            st.session_state.chat_messages.append({
                "role": "user", "content": user_question
            })
            
            with st.spinner("Professor Alex is thinking..."):
                answer = answer_question(
                    user_question, 
                    paper_context=st.session_state.full_text
                )
            
            st.session_state.chat_messages.append({
                "role": "assistant", "content": answer
            })
            st.rerun()
        
        # Clear chat
        if st.session_state.chat_messages:
            if st.button("🗑️ Clear chat history"):
                st.session_state.chat_messages = []
                reset_chat()
                st.rerun()