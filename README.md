🤖 Multi-Model RAG Benchmark Dashboard
Sistem komparasi performa Large Language Model (LLM) secara real-time untuk menganalisis perbedaan kualitas jawaban antara model Murni (Tanpa Konteks) versus Sistem RAG (Retrieval-Augmented Generation) menggunakan basis data vektor Chroma.

🌟 Fitur
✅ Virtual Jupyter Notebook Execution: Mengompilasi dan mengeksekusi kode dari berkas .ipynb secara aman langsung ke modul virtual Python untuk memuat retriever otomatis.

✅ Dual-Mode RAG (Statis & Dinamis):

Statis: Otomatis terhubung dengan database Chroma DB lokal bawaan notebook.

Dinamis: Pengguna dapat mengunggah file .txt atau .pdf baru via web untuk di-parsing instan ke database RAM (ephemeral).

✅ Multi-Model Live Comparison: Membandingkan klaster LLM secara berdampingan (side-by-side) antara Gemini 2.5 Flash (Temp 0.3 & 0.7) dan DeepSeek-R1 Distill Qwen 7B via Hugging Face API.

✅ Interactive Analytics UI: Tampilan visual output berbasis komponen st.tabs dan kolumnisasi kontras agar hasil evaluasi sangat mudah dibaca.

✅ W&B Sync Toggle: Sakelar interaktif pada UI untuk mengaktifkan atau menonaktifkan pengiriman metrik eksperimen ke cloud dashboard Weights & Biases secara langsung.

📋 Requirements
Python 3.9+

Google Gemini API Key

Hugging Face Access Token

Weights & Biases Account (Opsional)

🚀 Instalasi

1. Clone/Download Project

Bash
git clone <url-repositori-github-anda>
cd "Generative AI & RAG"
Buat & Aktifkan Virtual Environment

Bash
python -m venv .venv
Windows (PowerShell): .venv\Scripts\Activate.ps1

Linux/macOS: source .venv/bin/activate

Install Dependencies

Bash
pip install streamlit langchain-core langchain-google-genai langchain-community huggingface_hub langchain-text-splitters chromadb wandb nbformat pypdf torchvision torch
Setup Environment Variables
Buat file bernama .env di folder utama proyek dan isi dengan API keys Anda (tanpa tanda kutip):

Cuplikan kode
GOOGLE_API_KEY=your_gemini_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
💻 Cara Menggunakan
Pastikan berkas Jupyter Notebook utama bernama rag_dan_ai.ipynb sudah berada di direktori yang sama dengan app.py.

Jalankan server Streamlit UI melalui terminal Anda:

Bash
streamlit run app.py
Buka browser di alamat: http://localhost:8501

📁 Struktur Project
Plaintext
Generative AI & RAG/
├── .venv/ # Virtual environment Python
├── rag_dan_ai.ipynb # Jupyter Notebook (Database Chroma asal)
├── app.py # Script utama Streamlit UI Dashboard
├── .env # File konfigurasi API Keys & Tokens (Rahasia)
└── README.md # Dokumentasi proyek
🎯 Cara Kerja
Initialization: Web mendeteksi dan memuat modul retriever + embeddings langsung dari file rag_dan_ai.ipynb.

Select Document Mode: Gunakan database bawaan notebook (statis) ATAU unggah file PDF/TXT baru melalui panel input (dinamis).

Query & Execution: Pengguna mengetik pertanyaan, lalu menekan tombol benchmark. Sistem akan melakukan pencarian dokumen yang relevan.

Multi-LLM Response: Pertanyaan diproses oleh Gemini (Temp 0.3), Gemini (Temp 0.7), dan DeepSeek-R1 secara bergantian untuk menghasilkan dua versi jawaban (Murni vs RAG).

Tracking: Jika sakelar centang W&B diaktifkan, tabel rangkuman komparasi akan otomatis diunggah ke peladen cloud W&B.

📊 Weights & Biases
Untuk merekam hasil benchmark eksperimen secara online:

Centang kotak "📊 Aktifkan Sinkronisasi W&B" pada parameter sistem di sebelah kanan web.

Jalankan benchmark komparatif.

Sistem otomatis membuat run baru dan mengunggah ringkasan tabel ke dashboard akun W&B Anda di [https://wandb.ai](https://wandb.ai).

⚙️ Konfigurasi Default RAG Dinamis
Text Splitter: RecursiveCharacterTextSplitter

Chunk Size: 1000 karakter

Chunk Overlap: 200 karakter

Search Kwargs: k=3 (Mengambil 3 dokumen teratas paling relevan)

🔒 Keamanan
❌ JANGAN PERNAH melakukan commit atau membagikan file .env ke publik GitHub.

✅ Pastikan file .env sudah terdaftar di dalam file .gitignore proyek Anda.

👨‍💻 Author
slr549 - Project Multi-Model RAG Benchmark
