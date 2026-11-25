# Analisis Ulasan Pengguna Aplikasi MyTelkomsel (Google Play Store)
### Tugas Besar Machine Learning & Big Data (Semester 5)

Repositori ini memuat seluruh kode program, pipeline pre-processing teks bahasa Indonesia, dan draf laporan akademik untuk analisis ulasan aplikasi **MyTelkomsel** (`com.telkomsel.telkomselcm`) pada Google Play Store.

---

## 📁 Struktur Berkas

```text
Pre-Processing Data/
├── MyTelkomsel_Review_Preprocessing_WordCloud.ipynb  # Notebook mandiri (Google Colab ready)
├── LAPORAN_ANALISIS_MYTELKOMSEL.md                  # Draf laporan akademik lengkap (Bab 1 - Bab 6)
├── build_notebook.py                                # Skrip otomatisasi pembangun notebook
└── README.md                                        # Panduan ringkas penggunaan repositori
```

---

## 🚀 Panduan Eksekusi di Google Colab

1. Kunjungi [Google Colaboratory](https://colab.research.google.com/).
2. Klik tab **Upload** dan unggah berkas `MyTelkomsel_Review_Preprocessing_WordCloud.ipynb`.
3. Klik **Runtime** $\rightarrow$ **Run all** (atau tekan kombinasi tombol `Ctrl + F9`).
4. Notebook akan secara otomatis:
   * Menginstal pustaka pendukung (`google-play-scraper`, `PySastrawi`, `wordcloud`, dll.).
   * Melakukan *scraping* 10.000 ulasan terbaru dengan proteksi *rate-limiting*.
   * Menyaring ulasan kosong, duplikat, dan ulasan pendek (< 3 kata).
   * Menjalankan pipeline *pre-processing*: Case Folding, Cleaning, Slang Normalization, Tokenizing, Stopword Removal, dan Stemming Sastrawi berkecepatan tinggi dengan **Unique Word Caching** (~2 menit).
   * Menganalisis frekuensi Top 20 kata (*Unigram*) per kelas sentimen.
   * Menghasilkan dan menyimpan **3 Word Cloud resolusi tinggi (1200x600)**: Keseluruhan (`Blues`), Positif (`Greens`), dan Negatif (`Reds`).
   * Mengemas seluruh berkas CSV dan gambar PNG ke dalam `hasil_analisis_mytelkomsel.zip` lalu memicu pengunduhan otomatis ke komputer Anda.

---

## 📊 Ringkasan Metodologi & Spesifikasi

| Parameter | Spesifikasi |
|---|---|
| **Aplikasi Target** | MyTelkomsel (`com.telkomsel.telkomselcm`) |
| **Bahasa & Negara** | `lang='id'`, `country='id'` |
| **Volume Data** | 10.000 ulasan terbaru (*Natural Sampling*) |
| **Pelabelan Sentimen** | Rating 1–2 (Negatif), Rating 3 (Netral), Rating 4–5 (Positif) |
| **Pembersihan Khusus** | Kamus *hybrid* bahasa gaul + istilah telekomunikasi & *Custom Domain Stopwords* |
| **Optimasi Stemming** | *Unique Word Caching* (memangkas waktu dari ~60 menit menjadi ~2 menit) |
| **Output Visual** | Distribusi Rating/Sentimen, Bar Chart Top 20 Kata, dan 3 Gambar Word Cloud PNG |
