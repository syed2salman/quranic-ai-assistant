import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv
import json
import os
import difflib
import requests

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Quranic AI Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME / CSS  (Islamic-inspired: deep green + gold)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Poppins:wght@300;400;500;600&display=swap');

:root {
    --green-deep: #0d4d3c;
    --green: #14795c;
    --gold: #d4af37;
    --gold-soft: #e7cf86;
    --cream: #faf8f1;
    --ink: #1f2d28;
}

.stApp {
    background: linear-gradient(180deg, #f6f4ec 0%, #eef3ef 100%);
    font-family: 'Poppins', sans-serif;
    color: var(--ink);
}

.hero {
    position: relative;
    background: linear-gradient(135deg, var(--green-deep) 0%, var(--green) 100%);
    border-radius: 18px;
    padding: 34px 30px;
    margin-bottom: 8px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(13,77,60,0.25);
    border: 1px solid rgba(212,175,55,0.35);
}
.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        repeating-linear-gradient(45deg, rgba(255,255,255,0.05) 0 2px, transparent 2px 16px),
        repeating-linear-gradient(-45deg, rgba(255,255,255,0.05) 0 2px, transparent 2px 16px);
    opacity: 0.7;
    pointer-events: none;
}
.hero h1 {
    font-family: 'Amiri', serif;
    color: #fff;
    font-size: 2.5rem;
    margin: 0;
    position: relative;
    z-index: 1;
    text-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.hero p {
    color: var(--gold-soft);
    font-size: 1.05rem;
    margin: 6px 0 0 0;
    position: relative;
    z-index: 1;
    letter-spacing: 0.3px;
}
.hero .divider {
    height: 3px;
    width: 90px;
    background: linear-gradient(90deg, var(--gold), transparent);
    margin-top: 14px;
    border-radius: 3px;
    position: relative;
    z-index: 1;
}

.disclaimer {
    background: #fff9e8;
    border: 1px solid var(--gold);
    border-left: 5px solid var(--gold);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 16px 0;
    font-size: 0.92rem;
    color: #6b5a1e;
}

.answer-card {
    background: #ffffff;
    color: var(--ink);
    border-radius: 14px;
    padding: 22px 24px;
    border-left: 5px solid var(--green);
    box-shadow: 0 6px 18px rgba(20,121,92,0.10);
    margin: 10px 0 18px 0;
    font-size: 1.02rem;
    line-height: 1.7;
}

.source-card {
    background: #f4faf7;
    color: var(--ink);
    border: 1px solid #d5e8df;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
}
.source-tag {
    display: inline-block;
    background: var(--green);
    color: #fff;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: 6px;
    letter-spacing: 0.4px;
}

.section-h {
    font-family: 'Amiri', serif;
    color: var(--green-deep);
    font-size: 1.4rem;
    margin: 6px 0 4px 0;
    border-bottom: 2px solid var(--gold-soft);
    padding-bottom: 4px;
    display: inline-block;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #ffffff;
    color: #1f2d28 !important;
    border: 2px solid #d5e8df;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 1rem;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: #8a9a93 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--gold);
    box-shadow: 0 0 0 2px rgba(212,175,55,0.2);
}

.stButton > button {
    background: linear-gradient(135deg, var(--green-deep), var(--green));
    color: #fff;
    border: 1px solid var(--gold);
    border-radius: 10px;
    padding: 8px 20px;
    font-weight: 500;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--green), var(--green-deep));
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(13,77,60,0.25);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d4d3c 0%, #0a3d30 100%);
}
section[data-testid="stSidebar"] * {
    color: #eef7f2 !important;
}
section[data-testid="stSidebar"] .sb-title {
    font-family: 'Amiri', serif;
    color: var(--gold-soft) !important;
    font-size: 1.5rem;
}

.footer {
    text-align: center;
    color: #7a8a83;
    font-size: 0.85rem;
    margin-top: 30px;
    padding-top: 12px;
    border-top: 1px solid #d5e8df;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD QURAN STRUCTURE (JSON)
# ============================================================
try:
    with open("quran_complete_structure.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        quran_stats = data["quran_stats"]
        quran_structure = data["quran_structure"]
        if isinstance(quran_structure, str):
            quran_structure = json.loads(quran_structure)
except FileNotFoundError:
    quran_stats = {}
    quran_structure = []

# ============================================================
# BANNER  (contains title, tagline, and feature highlights)
# ============================================================
try:
    st.image("banner.png", use_container_width=True)
except Exception:
    st.markdown("""
    <div class="hero">
        <h1>📖 Quranic AI Assistant</h1>
        <p>Explore the Quran through AI-powered search &nbsp;•&nbsp; by Classic Mentor</p>
        <div class="divider"></div>
    </div>
    """, unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
ℹ️ <b>Educational tool.</b> Answers are AI-generated from a Quran translation and may be
incomplete or imprecise. For religious rulings, please consult qualified scholars and authentic sources.
</div>
""", unsafe_allow_html=True)

# ============================================================
# SHARED HELPERS
# ============================================================
def extract_text(ans):
    """Newer Gemini models may return content as a list of parts; get clean text either way."""
    c = getattr(ans, "content", ans)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for item in c:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    return str(c)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ============================================================
# PATH 1 — STRUCTURED QUERY HANDLER  (rule-based, from JSON)
# ============================================================
def answer_structure_query(q, quran_stats, quran_structure):
    q = q.lower().strip()

    def find_surah(qq):
        surah_names = [s['name_en'].lower() for s in quran_structure]
        match = difflib.get_close_matches(qq, surah_names, n=1, cutoff=0.5)
        if match:
            for s in quran_structure:
                if s['name_en'].lower() == match[0]:
                    return s
        return None

    surah_found = None
    for s in quran_structure:
        if s['name_en'].lower() in q or s['name_ar'] in q:
            surah_found = s
            break
    if not surah_found:
        surah_found = find_surah(q)

    if surah_found:
        if any(p in q for p in ["verse in surah", "ayat in surah", "how many verses in surah",
                                "how many ayat in surah", "verses in surah"]):
            return f"Surah {surah_found['name_en']} has {surah_found['verses']} verses."
        if any(p in q for p in ["which juz", "surah in which juz", "which para"]):
            return f"Surah {surah_found['name_en']} starts in Juz {surah_found['juz']}."
        if any(p in q for p in ["how many ruku in surah", "ruku in surah"]):
            return f"Surah {surah_found['name_en']} has {surah_found['rukus']} Rukus."

    if "how many surah" in q or "total surah" in q:
        return f"There are {quran_stats['total_surahs']} Surahs in the Quran."
    if any(p in q for p in ["list all surah", "names of surah", "all surah names"]):
        return "Here are the names of all Surahs:\n\n" + ", ".join(s['name_en'] for s in quran_structure)
    if any(p in q for p in ["total ayat", "how many ayat", "total verse", "how many verse"]):
        return f"There are {quran_stats['total_verses']} verses (Ayat) in the Quran."
    if "total ruku" in q or "how many ruku" in q:
        return f"There are {quran_stats['total_rukus']} Rukus in the Quran."
    if "total sajda" in q or "how many sajda" in q:
        return f"There are {quran_stats['total_sajdas']} places of Sajda in the Quran."
    if any(p in q for p in ["where sajda", "list sajda", "sajda ayah"]):
        sajda_list = [f"{s['surah']} (Ayah {s['ayah']})" for s in quran_stats["sajda_places"]]
        return "Sajda occurs in these places:\n" + "\n".join(sajda_list)

    return None

# ============================================================
# PATH 2 — SURAH SUMMARY  (detect + fetch full surah + summarize)
# ============================================================
def get_surah_number(surah_obj, quran_structure):
    """Return the surah's number: from an explicit key if present, else its position in the ordered list."""
    for key in ("number", "surah_no", "surah_number", "index", "id"):
        if key in surah_obj:
            try:
                return int(surah_obj[key])
            except (ValueError, TypeError):
                pass
    for i, s in enumerate(quran_structure, start=1):
        if s is surah_obj:
            return i
    return None

def detect_surah_summary_request(q, quran_structure):
    """If the query asks to summarize/explain a named surah, return its number and object."""
    ql = q.lower().strip()
    trigger_words = ["summary", "summarize", "summarise", "overview",
                     "what is surah", "explain surah", "tell me about surah",
                     "about surah", "theme of surah", "main theme", "what is in surah"]
    if not any(w in ql for w in trigger_words):
        return None

    found = None
    for s in quran_structure:
        if s['name_en'].lower() in ql or s.get('name_ar', '') in ql:
            found = s
            break
    if not found:
        names = [s['name_en'].lower() for s in quran_structure]
        match = difflib.get_close_matches(ql, names, n=1, cutoff=0.4)
        if match:
            found = next(s for s in quran_structure if s['name_en'].lower() == match[0])
    if not found:
        return None

    num = get_surah_number(found, quran_structure)
    return {"surah": found, "number": num} if num else None

def fetch_surah_text(surah_number):
    """Fetch the full English (Saheeh International) translation of a surah from Al Quran Cloud."""
    url = f"https://api.alquran.cloud/v1/surah/{surah_number}/en.sahih"
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    d = r.json()["data"]
    verses = [f"{a['numberInSurah']}. {a['text']}" for a in d["ayahs"]]
    return d["englishName"], d.get("englishNameTranslation", ""), "\n".join(verses)

# ============================================================
# BUILD RAG PIPELINE (once) — persisted Chroma + sources chain
# ============================================================
QURAN_PATH = "Saheeh_quran.pdf"
CHROMA_DIR = "./chroma_db"

def build_chain(db, llm, prompt_template, k):
    retriever = db.as_retriever(search_kwargs={"k": k})
    answer_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt_template
        | llm
    )
    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=answer_from_docs)

if "db" not in st.session_state:
    with st.spinner("🔄 Preparing the Quran knowledge base..."):
        try:
            hfembedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
                db = Chroma(persist_directory=CHROMA_DIR, embedding_function=hfembedding)
            else:
                if not os.path.exists(QURAN_PATH):
                    st.error(f"❌ Quran PDF not found: {QURAN_PATH}")
                    st.stop()

                loader = PyPDFLoader(QURAN_PATH)
                pages = loader.load()
                if not pages:
                    st.error("❌ No pages loaded from the PDF.")
                    st.stop()

                total_chars = sum(len(p.page_content) for p in pages if p.page_content)
                if total_chars < 100:
                    st.error("❌ PDF content appears empty or unreadable.")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                split_docs = splitter.split_documents(pages)
                db = Chroma.from_documents(
                    documents=split_docs,
                    embedding=hfembedding,
                    persist_directory=CHROMA_DIR
                )

            load_dotenv()
            GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
            if not GOOGLE_API_KEY:
                st.error("❌ GOOGLE_API_KEY not found. Add it to your .env file (local) "
                         "or HF Space Secrets (deployment).")
                st.stop()

            llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",  # current model; if this ever 404s, run list_models and pick one that's shown
                api_key=GOOGLE_API_KEY,
                temperature=0.3
            )

            template_string = """
You are a respectful Islamic educational assistant.
Use ONLY the provided Quranic context to answer the question.
If the context does not contain the answer, say so honestly and do NOT invent verses.
Do not issue religious rulings (fatwas); keep answers educational and factual.

Context from Quran:
{context}

Question: {question}

Answer (grounded in the context above):
"""
            prompt_template = ChatPromptTemplate.from_template(template_string)

            st.session_state.db = db
            st.session_state.llm = llm
            st.session_state.prompt_template = prompt_template
            st.success("✅ Quranic AI Assistant is ready!")

        except Exception as e:
            st.error(f"❌ Setup failed: {e}")
            st.code(f"{type(e).__name__}: {e}")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sb-title">📖 Quranic AI Assistant</div>', unsafe_allow_html=True)
    st.caption("by Classic Mentor")

    st.markdown("### ⚙️ Settings")
    k_chunks = st.slider("Passages to retrieve (k)", 1, 6, 3,
                         help="How many Quran passages to pull as context.")

    st.markdown("### 🎯 Sample Questions")
    for s in ["How many verses in Surah Al-Baqarah?",
              "Which Juz is Surah Ya-Sin in?",
              "Give a summary of Surah Ya-Sin",
              "What does the Quran say about patience?"]:
        st.markdown(f"• {s}")

    if quran_structure:
        st.markdown(f"### 📚 Surahs Loaded: {len(quran_structure)}")
        with st.expander("View all Surah names"):
            for idx, s in enumerate(quran_structure, start=1):
                st.markdown(f"**{idx}. {s['name_en']}**")
    else:
        st.warning("⚠️ Surah data not loaded.")

# ============================================================
# QUERY INPUT + ANSWER  (three paths)
# ============================================================
st.markdown('<div class="section-h">💬 Ask your question</div>', unsafe_allow_html=True)
query = st.text_input("", placeholder="e.g., Give a summary of Surah Ya-Sin  —or—  What does the Quran say about patience?")

if query:
    struct_answer = answer_structure_query(query, quran_stats, quran_structure)
    summary_req = detect_surah_summary_request(query, quran_structure)

    # PATH 1 — structured facts
    if struct_answer:
        st.markdown('<div class="section-h">📊 Answer — from Quran Structure</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-card">{struct_answer}</div>', unsafe_allow_html=True)

    # PATH 2 — surah summary (fetch full surah, then summarize)
    elif summary_req and "llm" in st.session_state:
        try:
            with st.spinner("📖 Fetching the full surah and summarizing..."):
                eng_name, eng_trans, surah_text = fetch_surah_text(summary_req["number"])
                summary_prompt = f"""You are a respectful Islamic educational assistant.
Below is the full English translation of Surah {eng_name} ({eng_trans}).
Write a clear, respectful educational summary of its main themes, message, and structure.
Base the summary ONLY on the text provided. Do not issue religious rulings.

Surah text:
{surah_text}

Summary:"""
                resp = st.session_state.llm.invoke(summary_prompt)
                summary_text = extract_text(resp).strip()

            title = f"📖 Summary — Surah {eng_name}"
            if eng_trans:
                title += f" ({eng_trans})"
            st.markdown(f'<div class="section-h">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{summary_text}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Could not summarize that surah: {e}")

    # PATH 3 — semantic RAG over the Quran text
    elif "db" in st.session_state:
        try:
            with st.spinner("🔍 Searching through the Quran..."):
                chain = build_chain(
                    st.session_state.db,
                    st.session_state.llm,
                    st.session_state.prompt_template,
                    k_chunks
                )
                response = chain.invoke(query)
                answer = response["answer"]

            st.markdown('<div class="section-h">📖 Answer — from Quran Text</div>', unsafe_allow_html=True)
            answer_text = extract_text(answer).strip()
            if answer_text:
                st.markdown(f'<div class="answer-card">{answer_text}</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ No specific answer found in the Quran text.")

            with st.expander(f"📚 Sources used ({len(response['context'])} passages retrieved)"):
                for i, doc in enumerate(response["context"], start=1):
                    page = doc.metadata.get("page", "unknown")
                    preview = doc.page_content[:500].strip()
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="source-tag">Passage {i} • page {page}</span><br>'
                        f'{preview} …</div>',
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.error(f"❌ Query failed: {e}")
    else:
        st.warning("⚠️ Assistant is still setting up. Please wait a moment and try again.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit • RAG with LangChain, HuggingFace &amp; Gemini • Deployed on Hugging Face Spaces
</div>
""", unsafe_allow_html=True)
