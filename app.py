import os
import time
import streamlit as st
import wandb
import nbformat
from types import ModuleType
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Muat environment variables dari file .env
load_dotenv()

# --- KONFIGURASI HALAMAN UTAMA STREAMLIT ---
st.set_page_config(
    page_title="Multi-Model RAG Benchmark Dashboard",
    page_icon="🤖",
    layout="wide"
)

# --- CUSTOM CSS UNTUK STYLE PREMIUM ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stAlert p { font-weight: 500; }
    h1 { color: #1E3A8A; }
    h2 { color: #1F2937; }
    </style>
""", unsafe_allow_html=True)

# Header Utama
st.title("🤖 Multi-Model RAG Benchmark Dashboard")
st.markdown("Analisator performa *Large Language Model* secara *real-time*: Membandingkan akurasi jawaban model **Murni (Tanpa Konteks)** versus **Sistem RAG (Dengan Dokumen Pendukung)**.")
st.markdown("---")


# --- IMPORT RETRIEVER LANGSUNG DARI FILE .IPYNB ---
@st.cache_resource
def load_retriever_from_notebook():
    notebook_path = "rag_dan_ai.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    notebook_module = ModuleType("rag_notebook")
    
    for cell in nb.cells:
        if cell.cell_type == "code":
            source_code = cell.source.strip()
            if not source_code:
                continue

            clean_lines = [
                line for line in source_code.splitlines()
                if not (line.strip().startswith('!') or line.strip().startswith('%'))
            ]
            clean_code = "\n".join(clean_lines)

            if clean_code.strip():
                try:  
                    exec(clean_code, notebook_module.__dict__)
                    if "retriever" in notebook_module.__dict__:
                        break
                except Exception:
                    continue

    if "retriever" in notebook_module.__dict__:
        return notebook_module.retriever, notebook_module.__dict__.get("embeddings")
    else:
        raise AttributeError("Variabel 'retriever' tidak ditemukan di dalam file notebook kamu.")


# Eksekusi pemuatan dokumen statis bawaan dengan status bar
try:
    with st.spinner("📦 Menghubungkan ke basis data dokumen Chroma dari notebook..."):
        retriever_statis, embeddings_model = load_retriever_from_notebook()
    st.toast("✅ Pangkalan dokumen RAG statis siap digunakan!", icon="🔥")
    
except Exception as e:
    st.error(f"⚠️ Gagal membaca berkas .ipynb: {e}")
    class DummyEmbeddings:
        def embed_documents(self, texts): return [[0.0]*1536 for _ in texts]
        def embed_query(self, text): return [0.0]*1536

    class DummyRetriever:
        def invoke(self, query):
            class Document:
                page_content = "Konteks cadangan: Sistem pendukung keputusan bantuan sosial menggunakan metode AHP."
            return [Document()]
    retriever_statis = DummyRetriever()
    embeddings_model = DummyEmbeddings()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --- STRUKTUR PANEL INPUT ---
col_input, col_info = st.columns([2, 1], gap="large")

with col_input:
    # Fitur Unggah File Custom (PDF atau TXT)
    uploaded_file = st.file_uploader(
        "📄 Unggah Dokumen Custom (.txt atau .pdf) untuk RAG Dinamis:",
        type=["txt", "pdf"]
    )
    query = st.text_area(
        "💬 Pertanyaan Uji Coba Model:", 
        value="Apa topik utama yang dibahas dalam dokumen ini?",
        height=100
    )
    tombol_jalan = st.button("🚀 Jalankan Benchmark Komparatif", type="primary", use_container_width=True)

with col_info:
    st.markdown("### 📋 Parameter Sistem")
    if uploaded_file is not None:
        st.success(f"📂 Mode RAG: Dinamis ({uploaded_file.name})")
    else:
        st.info("📦 Mode RAG: Statis (Bawaan Notebook)")

    # --- SAKELAR INTERAKTIF W&B ---
    aktifkan_wandb = st.checkbox("📊 Aktifkan Sinkronisasi W&B", value=False, help="Centang untuk mengirim hasil benchmark ke cloud Weights & Biases")
    
    st.info("""
    - **Vector Store:** Chroma DB (Local/Ephemeral)
    - **Metode Split:** Recursive Character
    - **Log Tracking:** Weights & Biases Disabled
    """)


# --- PROSES EVALUASI & PENAMPILAN HASIL ---
if tombol_jalan:
    with st.spinner("🔄 Sedang mengevaluasi prompt pada seluruh klaster LLM..."):
        
        # --- LOGIKA PENYUNTINGAN RETRIEVER (STATIS VS DINAMIS) ---
        active_retriever = retriever_statis
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.txt'):
                    text_content = uploaded_file.read().decode("utf-8")
                elif uploaded_file.name.endswith('.pdf'):
                    import pypdf
                    pdf_reader = pypdf.PdfReader(uploaded_file)
                    text_content = "".join([page.extract_text() for page in pdf_reader.pages])
                
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                from langchain_community.vectorstores import Chroma
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs_split = text_splitter.create_documents([text_content])
                
                # Inisialisasi basis data Chroma ephemeral di memori RAM hulu
                vectorstore_dinamis = Chroma.from_documents(docs_split, embeddings_model)
                active_retriever = vectorstore_dinamis.as_retriever(search_kwargs={"k": 3})
                st.toast("🔥 Sukses membuat Vectorstore dinamis dari file unggahan!", icon="⚡")
            except Exception as ex_dinamis:
                st.error(f"Gagal memproses file custom: {ex_dinamis}. Beralih menggunakan basis data notebook.")
                active_retriever = retriever_statis

        # Inisialisasi log monitoring W&B
    if aktifkan_wandb:
        os.environ["WANDB_MODE"] = "online"
        run = wandb.init(
            project="Generative-AI-RAG",
            name=f"RAG_Web_Benchmark_{int(time.time())}",
            reinit=True
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"
        run = wandb.init(project="Generative-AI-RAG", name="RAG_Web_Benchmark")

        # Status awal jawaban default
        jwb_gn_murni = jwb_gn_rag = "⚠️ Limit Kuota API Tercapai (429)"
        jwb_gc_murni = jwb_gc_rag = "⚠️ Limit Kuota API Tercapai (429)"
        jwb_ds_murni = jwb_ds_rag = "❌ Gagal Memproses Modul"

        prompt_murni = ChatPromptTemplate.from_messages([("human", "{input}")])
        system_prompt = "Anda adalah asisten tugas tanya jawab yang profesional dan objektif. Gunakan konteks berikut untuk menjawab pertanyaan secara padat dan jelas:\n\n{context}"
        prompt_rag = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

        # ==========================================
        # MODEL A: GEMINI 2.5 FLASH (TEMP 0.3)
        # ==========================================
        try:
            llm_gemini_n = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            chain_gn = prompt_murni | llm_gemini_n | StrOutputParser()
            jwb_gn_murni = chain_gn.invoke({"input": query})
            
            rag_gn = ({"context": active_retriever | format_docs, "input": RunnablePassthrough()} | prompt_rag | llm_gemini_n | StrOutputParser())
            jwb_gn_rag = rag_gn.invoke(query)
        except Exception:
            pass

        time.sleep(4)  # Jeda RPM Aman

        # ==========================================
        # MODEL B: GEMINI 2.5 FLASH (TEMP 0.7)
        # ==========================================
        try:
            llm_gemini_c = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
            chain_gc = prompt_murni | llm_gemini_c | StrOutputParser()
            jwb_gc_murni = chain_gc.invoke({"input": query})
            
            rag_gc = ({"context": active_retriever | format_docs, "input": RunnablePassthrough()} | prompt_rag | llm_gemini_c | StrOutputParser())
            jwb_gc_rag = rag_gc.invoke(query)
        except Exception:
            pass

        # ==========================================
        # MODEL C: DEEPSEEK-R1 (DIRECT CLIENT API)
        # ==========================================
        try:
            from huggingface_hub import InferenceClient
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            client = InferenceClient(provider="hf-inference", api_key=hf_token)
            
            # Murni
            messages_murni = [{"role": "user", "content": query}]
            completion_murni = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
                messages=messages_murni, max_tokens=500
            )
            jwb_ds_murni = completion_murni.choices[0].message.content

            # RAG Dinamis / Statis
            konteks_dokumen = format_docs(active_retriever.invoke(query))
            prompt_lengkap_rag = f"{system_prompt.format(context=konteks_dokumen)}\n\nPertanyaan: {query}"
            messages_rag = [{"role": "user", "content": prompt_lengkap_rag}]
            completion_rag = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
                messages=messages_rag, max_tokens=500
            )
            jwb_ds_rag = completion_rag.choices[0].message.content
        except Exception as e:
            jwb_ds_murni = jwb_ds_rag = f"Error DeepSeek: {e}"

        jwb_gpt_murni = jwb_pro_murni = jwb_gemma_murni = "💤 Model Diparkir (Status: Idle)"
        jwb_gpt_rag = jwb_pro_rag = jwb_gemma_rag = "💤 Model Diparkir (Status: Idle)"

    # --- RENDER STRUKTUR OUTPUT INTERAKTIF ---
    st.success("🎉 Analisis Benchmark Berhasil Diselesaikan!")
    
    st.header("📊 Hasil Komparasi Output Jawaban")
    st.markdown("Gunakan tab di bawah untuk melihat perbandingan mendalam pada setiap model LLM.")

    # Membuat Sistem Tab untuk masing-masing Model AI
    tab1, tab2, tab3, tab4 = st.tabs([
        "🟢 Gemini 2.5 Flash (Temp 0.3)", 
        "🔵 Gemini 2.5 Flash (Temp 0.7)", 
        "🐋 DeepSeek-R1 (Qwen-7B)", 
        "⚪ Model Idle (ChatGPT/Pro/Gemma)"
    ])

    with tab1:
        st.subheader("Model Evaluasi: Gemini 2.5 Flash (Deterministik - Temp 0.3)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🔴 Tanpa RAG (Jawaban Basis Data Dasar)")
            st.warning(jwb_gn_murni)
        with c2:
            st.markdown("### 🟢 Dengan RAG (Jawaban Basis Dokumen)")
            st.info(jwb_gn_rag)

    with tab2:
        st.subheader("Model Evaluasi: Gemini 2.5 Flash (Kreatif - Temp 0.7)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🔴 Tanpa RAG (Jawaban Basis Data Dasar)")
            st.warning(jwb_gc_murni)
        with c2:
            st.markdown("### 🟢 Dengan RAG (Jawaban Basis Dokumen)")
            st.info(jwb_gc_rag)

    with tab3:
        st.subheader("Model Evaluasi: DeepSeek-R1 Distill Qwen 7B (Penalaran / Reasoning)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🔴 Tanpa RAG (Jawaban Basis Data Dasar)")
            st.warning(jwb_ds_murni)
        with c2:
            st.markdown("### 🟢 Dengan RAG (Jawaban Basis Dokumen)")
            st.info(jwb_ds_rag)

    with tab4:
        st.subheader("💤 Model Evaluasi Tambahan (Status: Idle)")
        st.write("Model berikut sedang dikomentar di repositori untuk menghemat alokasi token harian:")
        st.text(f"- OpenAI ChatGPT Mini: {jwb_gpt_rag}")
        st.text(f"- Gemini 2.5 Pro: {jwb_pro_rag}")
        st.text(f"- Google Gemma 2: {jwb_gemma_rag}")

    # --- SIMPAN DATA KE W&B CLOUD ---
    data_tabel = [
        {"Model": "Gemini Flash 0.3", "Murni": jwb_gn_murni, "RAG": jwb_gn_rag},
        {"Model": "Gemini Flash 0.7", "Murni": jwb_gc_murni, "RAG": jwb_gc_rag},
        {"Model": "DeepSeek-R1 7B", "Murni": jwb_ds_murni, "RAG": jwb_ds_rag},
    ]

    try:
        compare_table = wandb.Table(columns=["Model", "Jawaban Murni", "Jawaban dengan RAG"])
        for baris in data_tabel:
            compare_table.add_data(
                str(baris["Model"]), 
                str(baris["Murni"]), 
                str(baris["RAG"])
            )
        wandb.log({"Tabel_Benchmark_Web_RAG": compare_table})
    except Exception:
        pass
    
    try:
        run.finish()
    except Exception:
        pass