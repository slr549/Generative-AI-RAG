# 🤖 Multi-Model RAG Benchmark Dashboard

Aplikasi berbasis web interaktif menggunakan framework **Streamlit** untuk menganalisis dan membandingkan performa respons _Large Language Model_ (LLM) secara _real-time_. Sistem ini mengevaluasi perbedaan kualitas antara model **Murni (Tanpa Konteks)** dan **Sistem RAG (Retrieval-Augmented Generation)** menggunakan basis data vektor lokal maupun dokumen yang diunggah secara dinamis.

---

## 🚀 Fitur Utama

- **Virtual Jupyter Notebook Execution:** Mengompilasi dan mengeksekusi sel kode dari berkas `.ipynb` secara aman langsung ke modul virtual Python untuk memuat _retriever_ dan _embeddings_ tanpa memicu _crash screen_.
- **Dual-Mode RAG Architecture:**
  - _Statis:_ Otomatis terhubung dengan Vector Store Chroma lokal yang diinisialisasi melalui Jupyter Notebook.
  - _Dinamis:_ Pengguna dapat mengunggah dokumen kustom (`.txt` atau `.pdf`) secara langsung via antarmuka web untuk diekstraksi ke dalam basis data Chroma _ephemeral_ (sementara di RAM).
- **Multi-Model Live Comparison:** Membandingkan klaster LLM secara berdampingan (_side-by-side_) yang mencakup **Gemini 2.5 Flash** (pada variasi suhu _deterministik_ 0.3 & _kreatif_ 0.7) serta model penalaran **DeepSeek-R1 (Qwen-7B)** via Hugging Face Inference API.
- **Interactive Analytics UI:** Hasil keluaran model disajikan menggunakan komponen `st.tabs` dan kolumnisasi kontras untuk mempermudah evaluasi kualitas jawaban.
- **Interactive Weights & Biases Sync Toggle:** Sakelar interaktif pada UI untuk mengaktifkan atau menonaktifkan pencatatan metrik eksperimen ke _cloud dashboard_ W&B secara _on-the-fly_.

---

## 🛠️ Arsitektur Kode & Komponen Utama

### 1. Ekstraksi Modul Virtual Notebook (.ipynb)

Aplikasi membaca struktur JSON dari notebook Jupyter secara mentah menggunakan `nbformat`. Melalui penyaringan ekspresi reguler, instruksi terminal (seperti `!pip` atau `%env`) diabaikan, dan sisa kode Python dieksekusi secara terisolasi menggunakan `types.ModuleType` untuk mengamankan objek `retriever`.

### 2. Pipeline RAG Dinamis

Ketika berkas baru terdeteksi oleh `st.file_uploader`, sistem memicu alur kerja berikut:

1.  **Parsing Teks:** Menggunakan `pypdf` untuk dokumen PDF atau dekorator standard _decoding_ untuk TXT.
2.  **Text Splitting:** Memecah dokumen menjadi fragmen kecil dengan `RecursiveCharacterTextSplitter` (_chunk size_: 1000, _overlap_: 200).
3.  **Vector Store Generation:** Memetakan fragmen ke dalam database vektor memori jangka pendek lewat `Chroma.from_documents` menggunakan model _embeddings_ bawaan.

### 3. Kendali Sinkronisasi W&B Interaktif

Menggunakan variabel kondisi Boolean dari komponen `st.checkbox` untuk memanipulasi variabel lingkungan `os.environ["WANDB_MODE"]`. Jika dicentang, status berubah menjadi `"online"`, memicu `wandb.init()` global dan mengunggah skema `wandb.Table` berisi rekaman jawaban komparatif ke peladen awan.

---

## 📦 Prasyarat & Instalasi

### 1. Kloning Repositori & Aktivasi Virtual Environment

```bash
git clone <url-repositori-github-anda>
cd "Generative AI & RAG"
python -m venv .venv

Windows (PowerShell): .venv\Scripts\Activate.ps1

Linux/macOS: source .venv/bin/activate

==============================================
2. Instalasi Dependensi
Pasang seluruh pustaka pustaka yang diperlukan ke dalam .venv:

Bash
pip install streamlit langchain-core langchain-google-genai langchain-community huggingface_hub langchain-text-splitters chromadb wandb nbformat pypdf torchvision torch
==============================================

==============================================
3. Konfigurasi Kredensial Lingkungan (.env)
Buat berkas bernama .env di direktori akar proyek, kemudian masukkan token API berikut (pastikan tanpa menggunakan tanda kutip):

Cuplikan kode
GOOGLE_API_KEY=AIzaSy...IsiApiKeyGeminiAnda
HUGGINGFACEHUB_API_TOKEN=hf_...IsiTokenHuggingFaceInferenceAnda
WANDB_API_KEY=...IsiApiKeyWeightsAndBiasesAnda
==============================================

==============================================
🖥️ Cara Menjalankan Aplikasi
Pastikan berkas Jupyter Notebook utama bernama rag_dan_ai.ipynb berada di direktori yang sama dengan app.py.

Jalankan server aplikasi lokal melalui terminal Anda:

Bash
streamlit run app.py
Aplikasi secara otomatis membuka peramban lokal pada alamat http://localhost:8501.

📊 Struktur Tampilan Eksperimen UI
Panel Kiri: Area input unggah berkas kustom dan area teks pengetikan klausa pertanyaan uji coba.

Panel Kanan: Papan indikator status mode RAG yang aktif beserta sakelar penentu transmisi data W&B.

Main Container: Dokumentasi komparasi visual berbasis Tab Layout yang menyajikan kontras teks jawaban reguler versus jawaban kontekstual (RAG) pada tiap-tiap model AI yang diuji.
==============================================
```
