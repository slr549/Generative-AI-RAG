import os
from dotenv import load_dotenv
import wandb

# 1. Load Environment Variables (Keamanan API Key)
load_dotenv()

# Pastikan API Key terpasang
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY tidak ditemukan. Pastikan sudah disetting di .env atau Google Colab Secrets.")

# 2. Inisialisasi Weights & Biases (Experiment Tracking)
# Kita akan mencatat jalannya eksperimen ini di WandB
wandb.init(project="rag-pdf-gemini", name="rag-vs-pure-llm-experiment", mode="disabled")  # Mode disabled untuk testing lokal

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# FITUR WAJIB 1: PDF Loader
# ==========================================
# Ganti 'dokumen.pdf' dengan file PDF pengetahuan baru yang ingin kamu pakai
pdf_path = "dokumen.pdf" 

if not os.path.exists(pdf_path):
    # Membuat file PDF dummy jika belum ada untuk testing
    print(f"Pengingat: Silakan siapkan file '{pdf_path}' di direktori kamu.")
    # Sediakan file PDF asli kamu di sini
    loader = None
else:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # ==========================================
    # FITUR WAJIB 2: Text Splitting (Chunking)
    # ==========================================
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # ==========================================
    # PERSYARATAN: Embeddings & Vector Database
    # ==========================================
    # Menggunakan Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Menyimpan ke ChromaDB (In-Memory untuk contoh ini)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Mengambil 3 chunk teratas

    # PERSYARATAN: Setup LLM (gemini-1.5-flash)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # ==========================================
    # PERBANDINGAN: AI Murni vs AI + RAG (WandB Tracking)
    # ==========================================
    
    # Pertanyaan spesifik yang jawabannya HANYA ADA di dalam PDF kamu
    pertanyaan = "Apa kesimpulan utama atau aturan khusus yang dibahas dalam dokumen ini?"

    print(f"\n[INFO] Menjalankan Eksperimen untuk Pertanyaan: '{pertanyaan}'\n")

    # --- Skenario 1: AI Murni (Tanpa RAG) ---
    print("--- Menguji AI Murni (Tanpa RAG) ---")
    respons_murni = llm.invoke(pertanyaan)
    jawaban_murni = respons_murni.content
    print(jawaban_murni)
    print("-" * 40)

    # --- Skenario 2: AI dengan RAG ---
    print("--- Menguji AI dengan RAG ---")
    
    # Membuat Prompt System untuk RAG
    system_prompt = (
        "Anda adalah asisten cerdas. Jawab pertanyaan berikut hanya berdasarkan konteks "
        "dokumen yang disediakan di bawah ini. Jika Anda tidak tahu jawabannya, katakan bahwa "
        "informasi tidak ditemukan dalam dokumen.\n\n"
        "Konteks:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Membuat RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Jalankan RAG
    respons_rag = rag_chain.invoke({"input": pertanyaan})
    jawaban_rag = respons_rag["answer"]
    print(jawaban_rag)
    
    # FITUR BONUS: Source Citation (Menampilkan rujukan dokumen asli)
    print("\n[BONUS] Sumber Rujukan:")
    for i, doc in enumerate(respons_rag["context"]):
        print(f"   Rujukan ke-{i+1}: (Halaman {doc.metadata.get('page', 0) + 1}) -> {doc.page_content[:150]}...")
    print("-" * 40)

    # --- LOGGING KE WEIGHTS & BIASES ---
    # Mencatat hasil perbandingan ke dashboard WandB
    wandb.log({
        "pertanyaan": pertanyaan,
        "jawaban_ai_murni": jawaban_murni,
        "jawaban_ai_rag": jawaban_rag
    })
    
    print("[SUCCESS] Hasil perbandingan berhasil dicatat ke Weights & Biases!")

# Selesaikan sesi WandB
wandb.finish()