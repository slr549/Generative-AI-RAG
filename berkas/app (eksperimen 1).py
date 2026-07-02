import os
from dotenv import load_dotenv
import wandb
# import gemma
# import openai

# 1. Load environment variables from .env file (Keamanan API Key)
load_dotenv()

# Pastikan API key sudah diatur di file .env
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY tidak ditemukan/error. Pastikan sudah diatur dan tidak salah ketik di file .env atau di Google Colab Secrets.")
api_key = os.getenv("WANDB_API_KEY")

# Inisialisasi Wandb dengan API key dari environment variable
# wandb.login(key=api_key)

# 2. Inisialisasi  Weights & Biases (wandb) untuk pelacakan eksperimen
wandb.init(project="Generative_AI_RAG", entity="your_wandb_entity", name="rag-vs-pure-lim-experiment", mode="disabled", config={
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 10
})

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# =================================================================
# FITUR WAJIB 1: PDF Loader
# =================================================================

# Ganti "dokumen.pdf" dengan file PDF pengetahuan baru yang ingin Anda muat
pdf_path = "dokumen.pdf"

if not os.path.exists(pdf_path):
    # Membuat file PDF dummy jika tidak ada
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Ini adalah dokumen PDF dummy untuk pengujian. Silakan ganti dengan dokumen PDF pengetahuan Anda.")
    pdf.output(pdf_path)
    # Menampilkan pesan bahwa file PDF dummy telah dibuat
    print(f"File PDF dummy '{pdf_path}' telah dibuat. Silakan ganti dengan dokumen PDF pengetahuan Anda.")
    # Sediakan file PDF asli kamu disini
    loader = None
else:
    # Memuat dokumen PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

# =================================================================
# FITUR WAJIB 2: Text Splitter (Chunking)
# =================================================================

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# =================================================================
# FITUR WAJIB 3: Embedding dengan Google Generative AI & Vector Database/Store (Chroma)
# =================================================================

# Menggunakan Google Generative AI untuk membuat embedding
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Menyimpan ke ChromaDB (In-Memory untuk pengujian, bisa diganti ke persistent storage)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(searc_kwargs={"k": 3}) # get 3 chunk up

# PERSYARATAN:  Setup LLM (gemini-1.5-flash/gemma)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_output_tokens=1024)

# =================================================================
# PERBANDINGAN: RAG + AI vs Pure LLM (WandB Tracking)
# =================================================================

# Pertanyaan spesifik yang jawabannya HANYA ADA di dalam pdf kamu
pertanyaan = "Apa kesimpulan utama atau aturan khusus yang dibahas dalam dokumen ini?"

print(f"\n[INFO] Menjalankan Eksperimen untuk Pertanyaan: '{pertanyaan}'\n")

# --- Skenario 1: AI Murni (Tanpa RAG) ---
print("--- Menguji AI Murni (Tanpa RAG) ---")
respons_murni = llm.invoke(pertanyaan)
jawaban_murni = respons_murni.content
print(f"Jawaban AI Murni: {jawaban_murni}\n")
print("-" * 40)

# --- Skenario 2: AI + RAG (Retrieval-Augmented Generation) ---
print("--- Menguji AI + RAG (Retrieval-Augmented Generation) ---")

# Membuat Prompt System untuk RAG
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan. Gunakan informasi dari dokumen untuk memberikan jawaban yang akurat."),
    ("user", "{context}\n\nPertanyaan: {question}") 
]) 
system_prompt = (
    "Anda adalah asisten AI cerdas. Jawab pertanyaan berikut hanya berdasarkan konteks "
    "dokumen yang disediakan di bawah ini. jika Anda tidak tahu jawabannya. katakan bahwa "
    "informasi tidak ditemukan dalam dokumen.\n\n"
    "Konteks:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{question}"),
    ("human", "{input}"),
])

# Membuat chain RAG
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Menjalankan RAG untuk pertanyaan
respons_rag = rag_chain.invoke({"question": pertanyaan})
jawaban_rag = respons_rag["output_text", "answers"][0]["content"] if "answers" in respons_rag else respons_rag["output_text"]
print(f"Jawaban AI + RAG: {jawaban_rag}\n")

# FITUR BONUS: Source Citation (Menampilkan rujukan dokumen asli)
print("\n[BONUS] Sumber RUjukan:")
for i, doc in enumerate(respons_rag["context"]):
    print(f" Rujukan ke-{i+1}: (Halaman {doc.metadata.get('page', 0) + 1}) -> {doc.page_content[:150]}..., sumber: {doc.metadata.get('source', 'Tidak ada sumber')})")
print("-" * 40)

# --- Logging ke WandB ---
# Log hasil eksperimen perbandingan ke dashboard WandB
wandb.log({
    "pertanyaan": pertanyaan,
    "jawaban_ai_murni": jawaban_murni,
    "jawaban_ai_rag": jawaban_rag,
})

print("[SUCCESS] Hasil perbandingan berhasil dicatat ke Weights & Biases:")

# Selesaikan sesi WandB
wandb.finish()