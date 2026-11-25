import json
import os
import re
import string
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

# Scraping
from google_play_scraper import Sort, reviews

# NLP Bahasa Indonesia
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Visualisasi Kata
from wordcloud import WordCloud

# ─── Konfigurasi ────────────────────────────────────────────────────────────────

# Kata-kata negasi yang akan digabung dengan kata berikutnya (misal: 'tidak bagus' → 'tidak_bagus')
NEGATION_WORDS = {'tidak', 'belum', 'bukan', 'jangan', 'tanpa', 'tak', 'tiada'}

# Kata domain yang TIDAK boleh di-stem oleh Sastrawi (agar tetap intuitif di word cloud)
STEM_EXCEPTIONS = {
    # Domain telekomunikasi
    'paket', 'jaringan', 'sinyal', 'kuota', 'pulsa', 'internet',
    'aplikasi', 'login', 'loading', 'verifikasi', 'notifikasi',
    'transaksi', 'pembayaran', 'pelayanan', 'berlangganan', 'registrasi',
    'promo', 'bonus', 'diskon', 'tagihan', 'pascabayar', 'prabayar',
    'provider', 'operator', 'server', 'error', 'bug', 'lag', 'hang',
    'otp', 'restart', 'download', 'upload', 'streaming',
    # Adjektiva/kata umum yang rawan over-stem oleh Sastrawi
    'jelas', 'kualitas', 'gratis', 'stabil', 'malas', 'puas',
    'habis', 'bagus', 'cepat', 'lambat', 'murah', 'mahal',
    'aktif', 'lancar', 'sesuai', 'guna', 'masuk',
}

# Kata-kata berkonotasi negatif — difilter dari word cloud Positif
NEGATIVE_LEXICON = {
    # Keluhan umum
    'mahal', 'error', 'bug', 'lag', 'hang', 'lambat', 'lelet', 'berat',
    'susah', 'sulit', 'kecewa', 'buruk', 'jelek', 'parah', 'payah',
    'gagal', 'rusak', 'hancur', 'mati', 'hilang', 'hangus', 'potong',
    'sedot', 'boros', 'rugi', 'sayang', 'busuk', 'ampas',
    'loading', 'aneh', 'ribet', 'rumit', 'ganggu',
    # Token negasi (negatif di konteks positif)
    'tidak_bisa', 'tidak_jelas', 'tidak_stabil', 'tidak_sesuai',
    'tidak_guna', 'tidak_masuk', 'tidak_pernah', 'tidak_lancar',
    'belum_bisa', 'tidak_ada',
    # Kata harapan/keluhan tersirat
    'moga', 'tolong', 'mohon',
    # Kata netral tapi noisy di konteks positif
    'sisa', 'cek', 'jaring',
}

# Kata-kata berkonotasi positif — difilter dari word cloud Negatif
POSITIVE_LEXICON = {
    'bagus', 'mantap', 'keren', 'lancar', 'cepat', 'mudah', 'murah',
    'puas', 'senang', 'suka', 'best', 'gratis', 'stabil', 'praktis',
    'nyaman', 'setia', 'percaya', 'manfaat', 'reward', 'hadiah',
    'terima_kasih', 'terimakasih',
}

# ────────────────────────────────────────────────────────────────────────────────


def setup_nltk():
    """Download resource NLTK jika belum tersedia."""
    resources = {
        'tokenizers/punkt': 'punkt',
        'tokenizers/punkt_tab': 'punkt_tab',
        'corpora/stopwords': 'stopwords',
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def load_slang_dict(path='kamus_slang.json'):
    """Muat kamus slang dari file JSON eksternal."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, path)
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def step_1_scraping(app_id='com.telkomsel.telkomselcm', target_reviews=10000, batch_size=200):
    print(f"\n{'='*70}\n[TAHAP 1] SCRAPING DATA DARI GOOGLE PLAY STORE\n{'='*70}")
    print(f"Target: {target_reviews:,} ulasan terbaru untuk aplikasi '{app_id}'...")

    all_reviews = []
    continuation_token = None
    batch_num = 1
    start_time = time.time()

    while len(all_reviews) < target_reviews:
        count_to_fetch = min(batch_size, target_reviews - len(all_reviews))
        try:
            result, continuation_token = reviews(
                app_id,
                lang='id',
                country='id',
                sort=Sort.NEWEST,
                count=count_to_fetch,
                continuation_token=continuation_token
            )
            if not result:
                print("⚠️ Tidak ada ulasan tambahan yang dapat diambil.")
                break

            all_reviews.extend(result)
            elapsed = time.time() - start_time
            print(f"Batch {batch_num:02d}: Berhasil mengambil {len(result)} ulasan. "
                  f"Total: {len(all_reviews):,}/{target_reviews:,} ({elapsed:.1f} dtk)")
            batch_num += 1

            if not continuation_token:
                print("ℹ️ Mencapai akhir halaman ulasan yang tersedia.")
                break

            time.sleep(0.5)  # Delay proteksi rate limit
        except Exception as e:
            print(f"❌ Error pada batch {batch_num}: {e}")
            time.sleep(2)
            continue

    df_raw = pd.DataFrame(all_reviews)
    cols = ['content', 'score', 'at', 'appVersion', 'thumbsUpCount']
    existing_cols = [c for c in cols if c in df_raw.columns]
    df_raw = df_raw[existing_cols]

    output_csv = 'ulasan_mentah.csv'
    df_raw.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Scraping selesai! {len(df_raw):,} ulasan tersimpan di '{output_csv}'.")
    return df_raw


def step_2_screening(df_raw):
    print(f"\n{'='*70}\n[TAHAP 2] DATA SCREENING & PELABELAN SENTIMEN\n{'='*70}")
    # 1. Hapus null
    df_clean = df_raw.dropna(subset=['content']).copy()
    null_dropped = len(df_raw) - len(df_clean)

    # 2. Hapus duplikat
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['content']).copy()
    dedup_dropped = before_dedup - len(df_clean)

    # 3. Hapus ulasan < 3 kata
    df_clean['word_count'] = df_clean['content'].apply(lambda x: len(str(x).strip().split()))
    before_len = len(df_clean)
    df_clean = df_clean[df_clean['word_count'] >= 3].reset_index(drop=True)
    short_dropped = before_len - len(df_clean)

    # 4. Pelabelan Sentimen
    def label_sentiment(score):
        if score in [1, 2]:
            return 'Negatif'
        elif score == 3:
            return 'Netral'
        elif score in [4, 5]:
            return 'Positif'
        return 'Lainnya'

    df_clean['sentiment'] = df_clean['score'].apply(label_sentiment)

    output_csv = 'ulasan_screening.csv'
    df_clean.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"- Data awal                  : {len(df_raw):,} ulasan")
    print(f"- Ulasan null dibuang        : {null_dropped:,}")
    print(f"- Ulasan duplikat dibuang    : {dedup_dropped:,}")
    print(f"- Ulasan pendek (<3 kata)    : {short_dropped:,}")
    print(f"- Sisa lolos screening       : {len(df_clean):,} ulasan ({len(df_clean)/len(df_raw)*100:.1f}%)")
    print(f"✅ Data screening disimpan ke '{output_csv}'.")

    # Visualisasi Distribusi Rating & Sentimen
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    score_counts = df_clean['score'].value_counts().sort_index()
    sns.barplot(x=score_counts.index, y=score_counts.values, ax=axes[0], palette='Blues_r')
    axes[0].set_title('Distribusi Rating Bintang (1 - 5)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Jumlah Ulasan')
    for i, v in enumerate(score_counts.values):
        axes[0].text(i, v + (max(score_counts.values)*0.015),
                     f"{v:,}\n({v/len(df_clean)*100:.1f}%)", ha='center', fontsize=9)

    sentiment_order = ['Negatif', 'Netral', 'Positif']
    sent_counts = df_clean['sentiment'].value_counts().reindex(sentiment_order)
    sns.barplot(x=sent_counts.index, y=sent_counts.values, ax=axes[1],
                palette=['#e74c3c', '#f39c12', '#2ecc71'])
    axes[1].set_title('Distribusi Kelas Sentimen', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Sentimen')
    axes[1].set_ylabel('Jumlah Ulasan')
    for i, v in enumerate(sent_counts.values):
        axes[1].text(i, v + (max(sent_counts.values)*0.015),
                     f"{v:,}\n({v/len(df_clean)*100:.1f}%)", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('distribusi_sentimen.png', dpi=300)
    plt.close()
    print("📁 Visualisasi disimpan ke 'distribusi_sentimen.png'.")
    return df_clean


def step_3_preprocessing(df_clean):
    print(f"\n{'='*70}\n[TAHAP 3] PIPELINE PRE-PROCESSING DATA TEKS\n{'='*70}")

    # Muat kamus slang dari JSON eksternal
    slang_dict = load_slang_dict()
    print(f"📖 Kamus slang dimuat: {len(slang_dict):,} entri dari 'kamus_slang.json'.")

    # 1. Case Folding
    print("⏳ Menjalankan Case Folding...")
    df_clean['case_folded'] = df_clean['content'].apply(lambda x: str(x).lower())

    # 2. Text Cleaning
    print("⏳ Menjalankan Text Cleaning (Regex & Karakter)...")
    def clean_text(text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[@#]\w+', '', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)       # karakter berulang
        text = re.sub(r'\d+', '', text)                    # angka
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)        # non-ASCII (emoji, dll.)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df_clean['cleaned'] = df_clean['case_folded'].apply(clean_text)

    # 3. Slang Normalization (dari file JSON)
    print("⏳ Menjalankan Normalisasi Slang & Istilah Telekomunikasi...")
    def normalize_slang(text):
        words = text.split()
        return ' '.join([slang_dict.get(w, w) for w in words])

    df_clean['slang_normalized'] = df_clean['cleaned'].apply(normalize_slang)

    # 4. Negation Handling — gabungkan kata negasi dengan kata berikutnya
    print("⏳ Menjalankan Negation Handling (tidak_X, belum_Y, ...)...")
    def handle_negation(text):
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            if words[i] in NEGATION_WORDS and i + 1 < len(words):
                result.append(f"{words[i]}_{words[i+1]}")
                i += 2
            else:
                result.append(words[i])
                i += 1
        return ' '.join(result)

    df_clean['negation_handled'] = df_clean['slang_normalized'].apply(handle_negation)

    # 5. Tokenizing dengan nltk.word_tokenize & Stopword Removal
    print("⏳ Menjalankan Tokenizing (nltk) & Stopword Removal...")
    nltk_stops = set(stopwords.words('indonesian'))
    sastrawi_stops = set(StopWordRemoverFactory().get_stop_words())
    custom_stops = {
        'telkomsel', 'mytelkomsel', 'aplikasi', 'app', 'apk',
        'nya', 'nih', 'sih', 'deh', 'dong', 'kan', 'tuh',
        'kak', 'min', 'admin', 'bintang', 'kasih',
        'kali', 'versi', 'banget', 'sangat', 'sekali',
        'bisa', 'ada', 'dari', 'yang', 'untuk',
        'dengan', 'ini', 'itu', 'di', 'ke', 'pada', 'dan', 'atau',
        'saya', 'kamu', 'kami', 'kita', 'mereka', 'dia',
        # Token negasi generik yang tidak informatif di word cloud
        'tidak_ada', 'belum_ada', 'tidak_punya',
    }
    # CATATAN: 'tidak', 'belum', 'bukan' TIDAK masuk stopwords
    # karena sudah digabung jadi token negasi (tidak_X) di langkah sebelumnya.
    # Kata bermakna seperti 'bantu', 'solusi', 'perbaiki', 'update'
    # sengaja TIDAK dimasukkan stopwords agar tetap muncul di word cloud.
    all_stops = nltk_stops.union(sastrawi_stops).union(custom_stops)
    # Pastikan kata negasi tidak masuk stopwords
    all_stops -= NEGATION_WORDS

    def tokenize_and_filter(text):
        tokens = word_tokenize(text)
        return [w for w in tokens if w not in all_stops and len(w) > 2]

    df_clean['tokens'] = df_clean['negation_handled'].apply(tokenize_and_filter)
    df_clean['stopwords_removed'] = df_clean['tokens'].apply(lambda x: ' '.join(x))

    # 6. Optimasi Stemming Sastrawi (Unique Word Caching + Exception List)
    print("⏳ Menjalankan Stemming Sastrawi dengan Unique Word Caching & Exception List...")
    stemmer = StemmerFactory().create_stemmer()

    unique_tokens = set()
    for t_list in df_clean['tokens']:
        unique_tokens.update(t_list)

    print(f"💡 Jumlah kata unik (vocabulary) yang di-stem: {len(unique_tokens):,} kata unik")
    print(f"🛡️ Exception list: {len(STEM_EXCEPTIONS)} kata domain tidak akan di-stem.")
    start_stem = time.time()
    stem_cache = {}
    for word in tqdm(unique_tokens, desc="Stemming"):
        # Kata dengan underscore (token negasi) → stem bagian setelah _
        if '_' in word:
            parts = word.split('_', 1)
            base = parts[1]
            if base in STEM_EXCEPTIONS:
                stem_cache[word] = word
            else:
                stem_cache[word] = f"{parts[0]}_{stemmer.stem(base)}"
        elif word in STEM_EXCEPTIONS:
            stem_cache[word] = word
        else:
            stem_cache[word] = stemmer.stem(word)
    print(f"⚡ Stemming selesai dalam {time.time() - start_stem:.1f} detik!")

    def apply_cache(tokens):
        stemmed = []
        for w in tokens:
            root = stem_cache.get(w, w)
            if root not in all_stops and len(root) > 2:
                stemmed.append(root)
        return stemmed

    df_clean['stemmed_tokens'] = df_clean['tokens'].apply(apply_cache)
    df_clean['teks_final'] = df_clean['stemmed_tokens'].apply(lambda x: ' '.join(x))

    df_final = df_clean[df_clean['teks_final'].str.strip() != ''].copy().reset_index(drop=True)
    output_csv = 'ulasan_terproses.csv'
    df_final.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Pre-processing selesai! {len(df_final):,} ulasan bersih tersimpan di '{output_csv}'.")
    return df_final


def step_4_analysis_and_visualization(df_final):
    print(f"\n{'='*70}\n[TAHAP 4] ANALISIS FREKUENSI KATA & VISUALISASI WORD CLOUD\n{'='*70}")

    # ── Unigram Analysis (manual Counter) ────────────────────────────────────
    def get_top_unigrams(corpus, top_n=20):
        words = []
        for doc in corpus:
            words.extend(doc.split())
        return Counter(words).most_common(top_n)

    corpus_all = df_final['teks_final']
    corpus_pos = df_final[df_final['sentiment'] == 'Positif']['teks_final']
    corpus_neg = df_final[df_final['sentiment'] == 'Negatif']['teks_final']

    top20_all = get_top_unigrams(corpus_all, 20)
    top20_pos = get_top_unigrams(corpus_pos, 20)
    top20_neg = get_top_unigrams(corpus_neg, 20)

    # Simpan Tabel Top 20 Unigram
    df_top20 = pd.DataFrame({
        'Rank': range(1, 21),
        'Kata (Semua)': [w[0] for w in top20_all],
        'Frekuensi (Semua)': [w[1] for w in top20_all],
        'Kata (Positif)': [w[0] for w in top20_pos],
        'Frekuensi (Positif)': [w[1] for w in top20_pos],
        'Kata (Negatif)': [w[0] for w in top20_neg],
        'Frekuensi (Negatif)': [w[1] for w in top20_neg]
    })
    df_top20.to_csv('top20_unigrams_per_sentimen.csv', index=False, encoding='utf-8')
    print("📁 Tabel Top 20 Unigram disimpan ke 'top20_unigrams_per_sentimen.csv'.")

    # ── Bigram & Trigram Analysis (sklearn CountVectorizer) ───────────────────
    print("⏳ Menjalankan analisis Bigram & Trigram dengan CountVectorizer...")

    def get_top_ngrams(corpus, ngram_range, top_n=20):
        """Hitung n-gram paling sering dari korpus teks menggunakan sklearn."""
        corpus_list = corpus.dropna().tolist()
        if not corpus_list:
            return []
        vec = CountVectorizer(ngram_range=ngram_range, token_pattern=r'(?u)\b\w[\w_]+\b')
        bag = vec.fit_transform(corpus_list)
        sum_words = bag.sum(axis=0)
        words_freq = [(word, int(sum_words[0, idx])) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:top_n]

    # Bigrams
    top20_bigram_all = get_top_ngrams(corpus_all, (2, 2), 20)
    top20_bigram_pos = get_top_ngrams(corpus_pos, (2, 2), 20)
    top20_bigram_neg = get_top_ngrams(corpus_neg, (2, 2), 20)

    # Trigrams
    top20_trigram_all = get_top_ngrams(corpus_all, (3, 3), 20)
    top20_trigram_pos = get_top_ngrams(corpus_pos, (3, 3), 20)
    top20_trigram_neg = get_top_ngrams(corpus_neg, (3, 3), 20)

    # Simpan Bigram
    max_len_bi = max(len(top20_bigram_all), len(top20_bigram_pos), len(top20_bigram_neg), 1)
    def pad_ngrams(ngram_list, length):
        padded = ngram_list + [('', 0)] * (length - len(ngram_list))
        return padded[:length]

    bi_all = pad_ngrams(top20_bigram_all, max_len_bi)
    bi_pos = pad_ngrams(top20_bigram_pos, max_len_bi)
    bi_neg = pad_ngrams(top20_bigram_neg, max_len_bi)

    df_bigram = pd.DataFrame({
        'Rank': range(1, max_len_bi + 1),
        'Bigram (Semua)': [w[0] for w in bi_all],
        'Frekuensi (Semua)': [w[1] for w in bi_all],
        'Bigram (Positif)': [w[0] for w in bi_pos],
        'Frekuensi (Positif)': [w[1] for w in bi_pos],
        'Bigram (Negatif)': [w[0] for w in bi_neg],
        'Frekuensi (Negatif)': [w[1] for w in bi_neg],
    })
    df_bigram.to_csv('top20_bigrams_per_sentimen.csv', index=False, encoding='utf-8')
    print("📁 Tabel Top 20 Bigram disimpan ke 'top20_bigrams_per_sentimen.csv'.")

    # Simpan Trigram
    max_len_tri = max(len(top20_trigram_all), len(top20_trigram_pos), len(top20_trigram_neg), 1)
    tri_all = pad_ngrams(top20_trigram_all, max_len_tri)
    tri_pos = pad_ngrams(top20_trigram_pos, max_len_tri)
    tri_neg = pad_ngrams(top20_trigram_neg, max_len_tri)

    df_trigram = pd.DataFrame({
        'Rank': range(1, max_len_tri + 1),
        'Trigram (Semua)': [w[0] for w in tri_all],
        'Frekuensi (Semua)': [w[1] for w in tri_all],
        'Trigram (Positif)': [w[0] for w in tri_pos],
        'Frekuensi (Positif)': [w[1] for w in tri_pos],
        'Trigram (Negatif)': [w[0] for w in tri_neg],
        'Frekuensi (Negatif)': [w[1] for w in tri_neg],
    })
    df_trigram.to_csv('top20_trigrams_per_sentimen.csv', index=False, encoding='utf-8')
    print("📁 Tabel Top 20 Trigram disimpan ke 'top20_trigrams_per_sentimen.csv'.")

    # ── Bar Chart: Unigram Top 20 ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=False)
    def plot_bars(ax, unigram_list, title, palette):
        words = [item[0] for item in unigram_list][::-1]
        counts = [item[1] for item in unigram_list][::-1]
        sns.barplot(x=counts, y=words, ax=ax, palette=palette)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Frekuensi')
        for i, v in enumerate(counts):
            ax.text(v + (max(counts)*0.01), i, f"{v:,}", va='center', fontsize=9)

    plot_bars(axes[0], top20_all, 'Top 20 Kata: Keseluruhan Ulasan', 'Blues_d')
    plot_bars(axes[1], top20_pos, 'Top 20 Kata: Ulasan Positif (Rating 4-5)', 'Greens_d')
    plot_bars(axes[2], top20_neg, 'Top 20 Kata: Ulasan Negatif (Rating 1-2)', 'Reds_d')
    plt.tight_layout()
    plt.savefig('barchart_top20_kata.png', dpi=300)
    plt.close()
    print("📁 Bar chart Unigram disimpan ke 'barchart_top20_kata.png'.")

    # ── Bar Chart: Bigram Top 15 ─────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(22, 8), sharey=False)
    top15_bi_all = top20_bigram_all[:15]
    top15_bi_pos = top20_bigram_pos[:15]
    top15_bi_neg = top20_bigram_neg[:15]

    plot_bars(axes2[0], top15_bi_all, 'Top 15 Bigram: Keseluruhan', 'Blues_d')
    plot_bars(axes2[1], top15_bi_pos, 'Top 15 Bigram: Positif (Rating 4-5)', 'Greens_d')
    plot_bars(axes2[2], top15_bi_neg, 'Top 15 Bigram: Negatif (Rating 1-2)', 'Reds_d')
    plt.tight_layout()
    plt.savefig('barchart_top15_bigram.png', dpi=300)
    plt.close()
    print("📁 Bar chart Bigram disimpan ke 'barchart_top15_bigram.png'.")

    # ── 3 Word Clouds (dengan Sentiment-Aware Filtering) ────────────────────
    def filter_corpus_by_sentiment(corpus, exclude_words):
        """Filter kata-kata dari korpus yang tidak sesuai sentimennya."""
        def filter_doc(doc):
            words = doc.split()
            filtered = [w for w in words if w not in exclude_words]
            return ' '.join(filtered)
        return corpus.apply(filter_doc)

    def make_wordcloud(corpus_text, title, colormap, output_png):
        text_data = ' '.join(corpus_text.dropna())
        if not text_data.strip():
            print(f"⚠️ Korpus kosong, word cloud '{output_png}' tidak dihasilkan.")
            return
        wc = WordCloud(
            width=1200, height=600,
            background_color='white',
            colormap=colormap,
            max_words=120,
            random_state=42,
            collocations=False
        ).generate(text_data)

        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=15, fontweight='bold', pad=18)
        plt.tight_layout(pad=0)
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🖼️ Word Cloud disimpan ke '{output_png}'.")

    # Word Cloud Keseluruhan — tanpa filter
    make_wordcloud(corpus_all, 'Word Cloud: Keseluruhan Ulasan MyTelkomsel',
                   'Blues', 'wordcloud_keseluruhan.png')

    # Word Cloud Positif — filter kata-kata negatif agar word cloud bersih
    corpus_pos_filtered = filter_corpus_by_sentiment(corpus_pos, NEGATIVE_LEXICON)
    print(f"🛡️ Word Cloud Positif: {len(NEGATIVE_LEXICON)} kata negatif difilter.")
    make_wordcloud(corpus_pos_filtered,
                   'Word Cloud: Ulasan Positif MyTelkomsel (Rating 4 - 5)',
                   'Greens', 'wordcloud_positif.png')

    # Word Cloud Negatif — filter kata-kata positif agar word cloud bersih
    corpus_neg_filtered = filter_corpus_by_sentiment(corpus_neg, POSITIVE_LEXICON)
    print(f"🛡️ Word Cloud Negatif: {len(POSITIVE_LEXICON)} kata positif difilter.")
    make_wordcloud(corpus_neg_filtered,
                   'Word Cloud: Ulasan Negatif MyTelkomsel (Rating 1 - 2)',
                   'Reds', 'wordcloud_negatif.png')


def main():
    print("="*70)
    print("  PIPELINE ANALISIS ULASAN APLIKASI MYTELKOMSEL (LOCAL RUNNER)")
    print("="*70)
    setup_nltk()
    df_raw = step_1_scraping(target_reviews=10000)
    df_clean = step_2_screening(df_raw)
    df_final = step_3_preprocessing(df_clean)
    step_4_analysis_and_visualization(df_final)
    print(f"\n{'='*70}\n🎉 SELURUH TAHAPAN SELESAI DENGAN SUKSES!\n{'='*70}")


if __name__ == '__main__':
    main()
