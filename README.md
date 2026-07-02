Berikut adalah isi mentah dari file `README.md` yang sudah dikonversi penuh ke dalam sintaks Markdown asli tanpa terpotong kotak pembungkus biasa. Kamu tinggal menyalin (_copy_) seluruh teks di bawah ini, lalu buat file baru bernama **`README.md`** di VS Code dan langsung tempel (_paste_) ke dalamnya:

---

# 🤖 Multi-Model RAG Benchmark Dashboard

Sistem komparasi performa _Large Language Model_ (LLM) secara _real-time_ untuk menganalisis perbedaan kualitas jawaban antara model **Murni (Tanpa Konteks)** versus **Sistem RAG (Retrieval-Augmented Generation)** menggunakan basis data vektor Chroma.

## 🌟 Fitur

- **✅ Virtual Jupyter Notebook Execution:** Mengompilasi dan mengeksekusi kode dari berkas `.ipynb` secara aman langsung ke modul virtual Python untuk memuat _retriever_ otomatis.
- **✅ Dual-Mode RAG (Statis & Dinamis):**
- _Statis:_ Otomatis terhubung dengan database Chroma DB lokal bawaan notebook.
- _Dinamis:_ Pengguna dapat mengunggah file `.txt` atau `.pdf` baru via web untuk di-parsing instan ke database RAM (_ephemeral_).

- **✅ Multi-Model Live Comparison:** Membandingkan klaster LLM secara berdampingan (_side-by-side_) antara **Gemini 2.5 Flash** (Temp 0.3 & 0.7) dan **DeepSeek-R1 Distill Qwen 7B** via Hugging Face API.
- **✅ Interactive Analytics UI:** Tampilan visual output berbasis komponen `st.tabs` dan kolumnisasi kontras agar hasil evaluasi sangat mudah dibaca.
- **✅ W&B Sync Toggle:** Sakelar interaktif pada UI untuk mengaktifkan atau menonaktifkan pengiriman metrik eksperimen ke _cloud dashboard_ Weights & Biases secara langsung.

## 📋 Requirements

- Python 3.9+
- Google Gemini API Key
- Hugging Face Access Token
- Weights & Biases Account (Opsional)

## 🚀 Instalasi

1. **Clone/Download Project**

```bash
git clone https://github.com/slr549/Generative-AI-RAG.git
cd "Generative AI & RAG"

```

2. **Buat & Aktifkan Virtual Environment**

```bash
python -m venv .venv

```

- _Windows (PowerShell):_ `.venv\Scripts\Activate.ps1`
- _Linux/macOS:_ `source .venv/bin/activate`

3. **Install Dependencies**

```bash
pip install streamlit langchain-core langchain-google-genai langchain-community huggingface_hub langchain-text-splitters chromadb wandb nbformat pypdf torchvision torch

```

4. **Setup Environment Variables**
   Buat file bernama `.env` di folder utama proyek dan isi dengan API keys Anda (tanpa tanda kutip):

```env
GOOGLE_API_KEY=your_gemini_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key

```

## 💻 Cara Menggunakan

1. Pastikan berkas Jupyter Notebook utama bernama `rag_dan_ai.ipynb` sudah berada di direktori yang sama dengan `app.py`.
2. Jalankan server Streamlit UI melalui terminal Anda:

```bash
streamlit run app.py

```

3. Buka browser di alamat: `http://localhost:8501`

## 📁 Struktur Project

```text
Generative AI & RAG/
├── .venv/                     # Virtual environment Python
├── rag_dan_ai.ipynb           # Jupyter Notebook (Database Chroma asal)
├── app.py                     # Script utama Streamlit UI Dashboard
├── .env                       # File konfigurasi API Keys & Tokens (Rahasia)
└── README.md                  # Dokumentasi proyek

```

## 🎯 Cara Kerja

1. **Initialization:** Web mendeteksi dan memuat modul _retriever_ + _embeddings_ langsung dari file `rag_dan_ai.ipynb`.
2. **Select Document Mode:** Gunakan database bawaan notebook (statis) ATAU unggah file PDF/TXT baru melalui panel input (dinamis).
3. **Query & Execution:** Pengguna mengetik pertanyaan, lalu menekan tombol benchmark. Sistem akan melakukan pencarian dokumen yang relevan.
4. **Multi-LLM Response:** Pertanyaan diproses oleh Gemini (Temp 0.3), Gemini (Temp 0.7), dan DeepSeek-R1 secara bergantian untuk menghasilkan dua versi jawaban (Murni vs RAG).
5. **Tracking:** Jika sakelar centang W&B diaktifkan, tabel rangkuman komparasi akan otomatis diunggah ke peladen _cloud_ W&B.

## 📊 Weights & Biases

Untuk merekam hasil benchmark eksperimen secara online:

1. Centang kotak **"📊 Aktifkan Sinkronisasi W&B"** pada parameter sistem di sebelah kanan web.
2. Jalankan benchmark komparatif.
3. Sistem otomatis membuat run baru dan mengunggah ringkasan tabel ke dashboard akun W&B Anda di `[https://wandb.ai](https://wandb.ai)`.

## ⚙️ Konfigurasi Default RAG Dinamis

- **Text Splitter:** `RecursiveCharacterTextSplitter`
- **Chunk Size:** 1000 karakter
- **Chunk Overlap:** 200 karakter
- **Search Kwargs:** `k=3` (Mengambil 3 dokumen teratas paling relevan)

## 🔒 Keamanan

- ❌ **JANGAN PERNAH** melakukan commit atau membagikan file `.env` ke publik GitHub.
- ✅ Pastikan file `.env` sudah terdaftar di dalam file `.gitignore` proyek Anda.
- kalau bingung, sudah disiapkan file .env tanpa token/clear.

## 👨‍💻 Author

- **Rahanz** - Project Multi-Model RAG Benchmark
