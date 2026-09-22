# AI Greeting Universitas Mikroskil

Sistem penyapa calon mahasiswa berbasis computer vision (deteksi orang +
warna pakaian) dan RAG (tanya-jawab berbasis dokumen resmi Mikroskil).

Dokumen sumber kebenaran project ini:

- **AI Greeting Mikroskil Blueprint** -- kenapa sistem dirancang seperti ini.
- **AI Greeting Mikroskil - Master Development & Build Plan** -- checkpoint
  apa yang dikerjakan sekarang, dan aturan yang mengikat selama development.

Status saat ini: **Checkpoint 01 -- Foundation** (kerangka project, belum
ada logika vision/RAG apa pun).

## Menjalankan

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env         # lalu isi GEMINI_API_KEY
python main.py
pytest tests/unit
```

## Struktur Project

Lihat Bagian 6 di Master Development & Build Plan untuk penjelasan lengkap
setiap folder dan di checkpoint mana file tersebut diisi.

```
main.py                    Titik masuk aplikasi
config/                    Konfigurasi (.yaml) + pemuat & validatornya
src/core/                  Tipe data inti, enum, exception, logger
src/vision/                Kamera, deteksi orang, tracking, warna baju
src/application/           State machine, sapaan, percakapan, orchestrator
src/knowledge/             Loader dokumen, chunking, embedding, vector store
src/rag/                   Retrieval + penyusunan prompt + pipeline jawaban
src/llm/                   Interface & implementasi provider LLM
src/tts/                   Interface & implementasi text-to-speech
src/ui/                    Input pengguna & tampilan chat
data/                      Dokumen, vector store, cache audio, log
scripts/                   Skrip pemeriksaan manual per komponen
tests/                     Unit test & integration test
```

## Aturan yang Mengikat

Lihat Bagian 2 Build Plan. Ringkas: modular, satu sumber state, tidak boleh
mengarang fakta Universitas Mikroskil, satu checkpoint pada satu waktu.
