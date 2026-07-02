import os
from dotenv import load_dotenv
import wandb

# 1. Load Environment Variables (Keamanan API Key)
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY tidak ditemukan di file .env!")

# 2. Inisialisasi Weights & Biases
wandb.init(project="rag-pdf-comparison", name="gemini-vs-gemma-rag", mode="disabled")  # Mode disabled untuk testing lokal

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# FITUR WAJIB 1 & 2: Loader & Splitting
# ==========================================
pdf_path = "test.pdf"  # Pastikan file ini ada di foldermu

if not os.path.exists(pdf_path):
    print(f"Silakan siapkan file '{pdf_path}' di direktori proyek Anda terlebih dahulu.")
else:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # ==========================================
    # PERSYARATAN: Embeddings & Vector DB
    # ==========================================
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Pertanyaan uji coba
    pertanyaan = "Apa kesimpulan utama atau aturan khusus yang dibahas dalam dokumen ini?"

    # ==========================================
    # EKSPERIMEN: Pengujian 2 Model Berbeda
    # ==========================================
    # Daftar model yang akan kita uji
    target_models = {
        "Gemini 1.5 Flash": "gemini-1.5-flash",
        "Gemma 2 9B": "gemma2-9b-it"
    }

    # Prompt Template untuk RAG
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

    # Tempat menyimpan data log untuk dikirim ke WandB Table
    data_log = []

    for model_name, model_code in target_models.items():
        print(f"\n[RUNNING] Menguji Model: {model_name}...")
        
        # Inisialisasi LLM sesuai giliran
        llm = ChatGoogleGenerativeAI(model=model_code, temperature=0.3)
        
        # 1. Tes Tanpa RAG (AI Murni)
        respons_murni = llm.invoke(pertanyaan)
        jawaban_murni = respons_murni.content
        
        # 2. Tes Dengan RAG
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        respons_rag = rag_chain.invoke({"input": pertanyaan})
        jawaban_rag = respons_rag["answer"]

        # --- BARIS PRINT INI UNTUK TESTING LOKAL ===
        print(f"\n==== Hasil {model_name.upper()} ====")
        print(f"🤖 AI Murni (Tanpa RAG):\n{jawaban_murni}\n")
        print(f"📚 AI + RAG (Dengan PDF):\n{jawaban_rag}")
        print("==================================\n")
        
        # Simpan hasilnya ke list data
        data_log.append([model_name, pertanyaan, jawaban_murni, jawaban_rag])
        
        print(f"[DONE] Selesai memproses {model_name}.")

    # ==========================================
    # LOGGING KE WEIGHTS & BIASES (Tabel Perbandingan)
    # ==========================================
    # Membuat tabel interaktif di WandB agar dosenmu bisa melihat perbandingannya dengan mudah
    columns = ["Model", "Pertanyaan", "Jawaban AI Murni (Tanpa RAG)", "Jawaban AI + RAG"]
    comparison_table = wandb.Table(columns=columns, data=data_log)
    
    # Kirim tabel ke dashboard WandB
    wandb.log({"Tabel_Perbandingan_Model": comparison_table})
    
    print("\n[SUCCESS] Semua eksperimen selesai! Silakan cek link dashboard Weights & Biases kamu.")

# Tutup run WandB
wandb.finish()