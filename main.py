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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
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

# Alasan (A.6): Impor pustaka emoji jika tersedia untuk fallback demojize sentimen;
# jika belum terpasang, tetap berjalan memakai kamus EMOJI_SENTIMENT_MAP bawaan.
try:
    import emoji
    HAS_EMOJI_LIB = True
except ImportError:
    HAS_EMOJI_LIB = False

# ─── Konfigurasi ────────────────────────────────────────────────────────────────

# Kata-kata negasi yang akan digabung dengan kata berikutnya jika lolos whitelist
NEGATION_WORDS = {'tidak', 'belum', 'bukan', 'jangan', 'tanpa', 'tak', 'tiada'}

# Alasan (A.5): Whitelist kata bermakna agar negasi hanya menempel ke kata kerja/sifat
# yang relevan dan TIDAK membentuk token semu dengan stopword (mis. jangan_di, tidak_pernah).
# Daftar ini sengaja dipisah di atas agar mudah diedit sesuai kebutuhan domain.
NEGATION_TARGET_WHITELIST = {
    'bisa', 'bagus', 'jelas', 'stabil', 'sesuai', 'lancar', 'puas', 'mau', 'ada',
    'masuk', 'guna', 'berguna', 'suka', 'paham', 'benar', 'wajar', 'layak', 'normal',
    'aman', 'berfungsi', 'fungsi', 'muncul', 'kirim', 'terkirim', 'buka', 'terbuka',
    'jalan', 'berjalan', 'connect', 'konek', 'nyambung', 'terhubung', 'respon',
    'responsif', 'cepat', 'murah', 'mahal', 'worth', 'rekomen', 'ramah', 'peduli',
    'tahu', 'mengerti', 'sebanding', 'berubah', 'hilang', 'rusak', 'lemot', 'lambat',
    'error', 'gangguan', 'ribet', 'susah', 'sulit', 'enak', 'nyaman', 'mudah',
    'berhasil', 'mampu', 'sempat', 'sanggup', 'terima', 'beli', 'pakai', 'bayar',
}

# Alasan (A.1 & A.2): Kata kontras dipakai sebagai penanda batas klausa pada n-gram
# agar tidak terbentuk bigram semu lintas klausa (mis. "mahal sinyal" dari "mahal tapi sinyal jelek"),
# dan wajib dipertahankan (tidak boleh dihapus sebagai stopword).
CONTRAST_WORDS = {'tapi', 'tetapi', 'padahal', 'namun', 'sedangkan', 'kecuali', 'melainkan'}

# Alasan (A.2): Kata penguat/pelemah membawa intensitas sentimen sehingga harus dipertahankan.
INTENSIFIER_WORDS = {
    'sangat', 'banget', 'kurang', 'sekali', 'terlalu', 'paling',
    'agak', 'cukup', 'lebih', 'malah', 'justru', 'amat',
}

CONTRAST_AND_INTENSIFIER_WORDS = CONTRAST_WORDS.union(INTENSIFIER_WORDS)

# Alasan (B.2): Entri berspasi (multi-kata) dipindahkan dari kamus_slang.json ke daftar
# penggantian frasa berbasis regex agar tertangani utuh SEBELUM tokenizing.
PHRASE_REPLACEMENTS = {
    'kayak': 'seperti',
    'force close': 'keluar_paksa',
    'blank screen': 'layar_putih',
    'customer service': 'cs',
    'nge lag': 'lag',
    'nge-lag': 'lag',
    'muter muter': 'loading',
    'terima kasih': 'terima_kasih',
    'no signal': 'tidak ada sinyal',
    'bad network': 'jaringan buruk',
    'slow connection': 'koneksi lambat',
    'worth it': 'sebanding',
    'pulsa safe': 'pulsa_safe',
}

# Alasan (A.6): Konversi emoji menjadi kata sentimen bahasa Indonesia sebelum
# pembersihan karakter non-ASCII, agar sinyal emosi dari emoji tidak hilang.
EMOJI_SENTIMENT_MAP = {
    '😡': ' marah ', '🤬': ' marah ', '😠': ' marah ', '😤': ' marah ',
    '👎': ' buruk ', '💩': ' buruk ', '🤮': ' muak ', '🤢': ' muak ',
    '😭': ' sedih ', '😢': ' sedih ', '😔': ' sedih ', '😞': ' sedih ', '🥺': ' sedih ', '🥲': ' sedih ',
    '😑': ' kecewa ', '😒': ' kecewa ', '🤦': ' kecewa ', '🙄': ' kecewa ', '💔': ' kecewa ',
    '👍': ' bagus ', '⭐': ' bagus ', '🌟': ' bagus ', '🔥': ' mantap ', '💯': ' mantap ',
    '😁': ' senang ', '😄': ' senang ', '😊': ' senang ', '🙂': ' senang ', '😇': ' senang ',
    '🥰': ' suka ', '😍': ' suka ', '❤️': ' suka ', '❤': ' suka ', '💕': ' suka ', '💖': ' suka ',
    '😂': ' ketawa ', '🤣': ' ketawa ', '🙏': ' mohon ',
}

# Alasan (A.7): Normalisasi kata bahasa Inggris bermakna ke padanan Indonesia
# supaya frekuensi kata sentimen/domain tidak terpecah antara EN dan ID.
ENGLISH_TO_INDONESIAN = {
    'good': 'bagus', 'best': 'terbaik', 'nice': 'bagus', 'great': 'bagus',
    'awesome': 'mantap', 'cool': 'keren', 'fast': 'cepat', 'slow': 'lambat',
    'bad': 'buruk', 'worst': 'terburuk', 'poor': 'buruk', 'trash': 'buruk',
    'garbage': 'buruk', 'useless': 'tidak berguna', 'worthless': 'tidak berguna',
    'helpful': 'membantu', 'easy': 'mudah', 'hard': 'sulit', 'expensive': 'mahal',
    'cheap': 'murah', 'love': 'suka', 'like': 'suka', 'hate': 'benci',
    'happy': 'senang', 'disappointed': 'kecewa', 'fix': 'perbaiki', 'fixed': 'perbaiki',
    'broken': 'rusak', 'crash': 'keluar paksa', 'freeze': 'macet', 'stuck': 'macet',
    'smooth': 'lancar', 'stable': 'stabil', 'connection': 'koneksi', 'network': 'jaringan',
    'price': 'harga', 'service': 'layanan', 'customer': 'pelanggan', 'quota': 'kuota',
    'money': 'uang', 'free': 'gratis', 'please': 'tolong', 'sorry': 'maaf',
    'thank': 'terima_kasih', 'recommend': 'rekomendasi', 'recommended': 'rekomendasi',
    'not': 'tidak', 'very': 'sangat', 'but': 'tapi',
}

# Alasan (A.7): Stopword dasar bahasa Inggris & sapaan agar tidak mengotori token.
# Catatan: 'no' sengaja TIDAK dimasukkan karena di ulasan Indonesia sering berarti 'nomor'.
ENGLISH_BASIC_STOPWORDS = {
    'the', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
    'as', 'it', 'its', 'this', 'that', 'these', 'those', 'my', 'your',
    'you', 'me', 'we', 'our', 'they', 'their', 'he', 'she', 'and', 'or',
    'so', 'if', 'when', 'while', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'can', 'could', 'should', 'boss', 'bos', 'bro', 'guys',
    'sir', 'dear', 'gan', 'cuy', 'broo',
}

# Alasan (A.4): Kata berimbuhan/majemuk bertanda hubung yang harus dipertahankan utuh
# sebagai satu token agar tidak rusak menjadi kata gabungan (mis. "tibatiba").
LEXICALIZED_HYPHEN_WORDS = {
    'tiba-tiba', 'gara-gara', 'sia-sia', 'rata-rata', 'masing-masing',
    'sehari-hari', 'm-banking', 'e-wallet', 'e-commerce', 'non-paket',
    'anti-sedot', 'pulsa-safe',
}

# Alasan (A.2): Penanda kata di daftar stopword yang masih memiliki nuansa sentimen/pragmatik
# untuk dilaporkan saat menampilkan daftar stopword final.
SENTIMENT_FLAGGED_STOPS = {
    'terima', 'semoga', 'moga', 'maaf', 'ampun', 'tolong', 'mohon', 'bantu',
    'baik', 'benar', 'salah', 'masalah', 'lama', 'butuh', 'mending', 'hadehh',
    'hadeh', 'jangan_mahal', 'tidak_ngotak', 'tidak_masuk', 'cukup', 'biasa',
}

# Kata domain yang TIDAK boleh di-stem oleh Sastrawi (agar tetap intuitif di word cloud)
# Alasan (A.4): Tambahkan kata bertanda hubung & token gabungan agar Sastrawi tidak memecah tanda hubung/underscore.
STEM_EXCEPTIONS = {
    # Domain telekomunikasi
    'paket', 'jaringan', 'sinyal', 'kuota', 'pulsa', 'internet',
    'aplikasi', 'login', 'loading', 'verifikasi', 'notifikasi',
    'transaksi', 'pembayaran', 'pelayanan', 'layanan', 'berlangganan', 'langganan', 'registrasi',
    'promo', 'bonus', 'diskon', 'tagihan', 'pascabayar', 'prabayar',
    'provider', 'operator', 'server', 'error', 'bug', 'lag', 'hang',
    'otp', 'restart', 'download', 'upload', 'streaming',
    # Kata bertanda hubung & majemuk yang harus tetap utuh
    'tiba-tiba', 'gara-gara', 'sia-sia', 'rata-rata', 'masing-masing',
    'sehari-hari', 'm-banking', 'e-wallet', 'e-commerce', 'non-paket',
    'terima_kasih', 'keluar_paksa', 'pulsa_safe',
    # Adjektiva/kata umum yang rawan over-stem oleh Sastrawi
    'jelas', 'kualitas', 'gratis', 'stabil', 'malas', 'puas',
    'habis', 'bagus', 'cepat', 'lambat', 'murah', 'mahal',
    'aktif', 'lancar', 'sesuai', 'guna', 'masuk',
    'boros', 'parah', 'susah', 'sulit', 'gagal', 'hilang',
    'kecewa', 'rugi', 'busuk', 'potong', 'hangus',
    'pelanggan', 'dipakai', 'telepon',
}

# Alasan (D.1): Sensor kata kasar HANYA pada gambar word cloud agar tetap etis untuk publikasi,
# namun TIDAK menghapusnya dari dataset/teks_final agar analisis & pemodelan tetap objektif.
PROFANITY_FILTER = {
    'tolol', 'babi', 'anjing', 'anj', 'ajg', 'asu', 'asuu', 'bangsat', 'bangke',
    'bangkee', 'kontol', 'memek', 'ngentot', 'ngentod', 'tai', 'taik', 'thai',
    'goblok', 'goblog', 'bodoh', 'idiot', 'bajingan', 'brengsek', 'setan', 'iblis',
    'jancok', 'cuk', 'cok', 'bgst', 'kntl', 'mmk', 'ngtd', 'pantek', 'puki',
    'kimak', 'sialan', 'keparat', 'bacot', 'cot', 'bejat', 'ngotak',
}

# Kata-kata berkonotasi negatif — dipakai untuk word cloud Positif & validasi leksikon (C.2)
NEGATIVE_LEXICON = {
    'mahal', 'error', 'bug', 'lag', 'hang', 'lambat', 'lelet', 'berat',
    'susah', 'sulit', 'kecewa', 'buruk', 'jelek', 'parah', 'payah',
    'gagal', 'rusak', 'hancur', 'mati', 'hilang', 'hangus', 'potong',
    'sedot', 'boros', 'rugi', 'sayang', 'busuk', 'ampas', 'sampah',
    'loading', 'aneh', 'ribet', 'rumit', 'ganggu', 'gangguan',
    'salah', 'turun', 'keluh', 'kendala', 'ilang',
    'kikir', 'habis', 'turunin', 'naikin', 'sumpah', 'gara', 'iklan',
    'kalah', 'pindah', 'kacau', 'bobrok', 'burik', 'bapuk', 'muak',
    'emosi', 'marah', 'sedih', 'sesal', 'korup', 'korupsi', 'kemahalan',
    'terburuk', 'merugikan', 'mengecewakan', 'bermasalah', 'macet', 'benci',
    'pelit', 'sia-sia', 'keluar_paksa',
    # Token negasi (negatif di konteks positif)
    'tidak_bisa', 'tidak_bagus', 'tidak_jelas', 'tidak_stabil', 'tidak_sesuai',
    'tidak_guna', 'tidak_berguna', 'tidak_masuk', 'tidak_lancar', 'tidak_puas',
    'tidak_mau', 'tidak_ada', 'tidak_normal', 'tidak_aman', 'tidak_berfungsi',
    'belum_bisa', 'belum_stabil', 'belum_sesuai', 'belum_lancar',
    # Kata harapan/keluhan tersirat
    'moga', 'tolong', 'mohon',
    # Kata netral tapi noisy di konteks positif
    'sisa', 'cek', 'jaring', 'ngeleg',
    # Keluhan/negatif tambahan yang sering muncul di ulasan rating 4-5
    'kesal', 'kesel', 'bangkrut', 'spam', 'restart', 'paksa',
    'hapus', 'darurat', 'putus', 'gila', 'lupa',
}

# Kata-kata berkonotasi positif — dipakai untuk word cloud Negatif & validasi leksikon (C.2)
POSITIVE_LEXICON = {
    'bagus', 'mantap', 'keren', 'lancar', 'cepat', 'mudah', 'murah',
    'puas', 'senang', 'suka', 'best', 'gratis', 'stabil', 'praktis',
    'nyaman', 'setia', 'percaya', 'manfaat', 'bermanfaat', 'reward', 'hadiah',
    'terima_kasih', 'terimakasih', 'terbaik', 'membantu', 'hemat', 'terjangkau',
    'lumayan', 'good', 'top', 'mantep', 'aman', 'untung', 'sesuai', 'super',
    'rekomendasi', 'responsif', 'ramah', 'berhasil', 'sebanding',
}

# Kata topik netral — dibiarkan di Word Cloud Keseluruhan, tapi difilter
# dari Word Cloud Positif & Negatif agar kata sentimen lebih menonjol
NEUTRAL_TOPIC_WORDS = {
    'paket', 'kuota', 'jaringan', 'sinyal', 'internet', 'pulsa', 'data',
    'kartu', 'harga', 'nomor', 'provider', 'operator', 'simpati', 'indihome',
    'wifi', 'youtube', 'game', 'sms', 'daerah', 'akun', 'sistem', 'fitur',
    'beli', 'pakai', 'isi', 'buka', 'bayar', 'pembayaran', 'transaksi',
    'update', 'login', 'akses', 'aktif', 'ganti', 'tukar', 'cek', 'sisa',
    'kirim', 'ambil', 'jual', 'poin', 'saldo', 'tagihan', 'registrasi',
    'layanan', 'langganan', 'langgan', 'layan', 'koneksi', 'metode',
}

# Alasan (D.3): Kamus kata kunci aspek untuk analisis aspek berbasis leksikon.
ASPECT_KEYWORDS = {
    'Aplikasi': {
        'login', 'error', 'update', 'bug', 'loading', 'otp', 'keluar_paksa',
        'keluar', 'masuk', 'daftar', 'registrasi', 'buka', 'verifikasi',
        'hang', 'macet', 'restart', 'layar', 'crash', 'fitur', 'menu',
        'notifikasi', 'akun', 'aplikasi',
    },
    'Jaringan': {
        'sinyal', 'jaringan', 'lambat', 'lag', 'koneksi', 'internet', 'lemot',
        'lelet', 'putus', 'gangguan', 'hilang', 'stabil', 'cepat', 'lancar',
        'tower', '4g', '5g',
    },
    'Harga/Paket': {
        'mahal', 'kuota', 'pulsa', 'paket', 'harga', 'murah', 'promo',
        'diskon', 'potong', 'sedot', 'hangus', 'boros', 'bayar', 'saldo',
        'tagihan', 'beli', 'transaksi', 'poin', 'voucher', 'gratis',
    },
    'Layanan': {
        'cs', 'tolong', 'bantu', 'respon', 'layanan', 'pelayanan', 'admin',
        'keluhan', 'komplain', 'solusi', 'perbaiki', 'perbaikan', 'ramah',
        'peduli', 'lapor', 'laporan',
    },
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
    """Muat kamus slang dari file JSON eksternal dan pastikan bebas entri berspasi/self-mapping."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, path)
    with open(full_path, 'r', encoding='utf-8') as f:
        raw_dict = json.load(f)
    # Alasan (B.1 & B.2): Validasi agar kunci berspasi dan self-mapping tidak lolos
    return {k: v for k, v in raw_dict.items() if ' ' not in k and k != v}


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


def detect_near_duplicates(df, text_col='content', cosine_threshold=0.90, block_size=500):
    """Deteksi near-duplicate memakai hash teks ternormalisasi lalu TF-IDF cosine > 0.90."""
    # Alasan (A.8): Menangkap spam/ulasan hampir identik yang hanya berbeda kapitalisasi,
    # tanda baca, atau 1 kata sisipan agar tidak membias distribusi sentimen & train/test split.
    def _norm_hash(text):
        s = str(text).lower()
        s = re.sub(r'[^\w\s]', '', s)
        return re.sub(r'\s+', ' ', s).strip()

    norm_texts = df[text_col].apply(_norm_hash)
    seen_hashes = set()
    keep_after_hash = []
    for nh in norm_texts:
        if nh in seen_hashes:
            keep_after_hash.append(False)
        else:
            seen_hashes.add(nh)
            keep_after_hash.append(True)

    df_hash_clean = df[keep_after_hash].reset_index(drop=True)
    hash_dropped = len(df) - len(df_hash_clean)

    # Tahap kedua: TF-IDF cosine similarity > 0.90 per blok agar hemat memori
    norm_for_tfidf = df_hash_clean[text_col].apply(_norm_hash).tolist()
    tfidf_dropped = 0
    if len(norm_for_tfidf) > 1:
        vec = TfidfVectorizer(min_df=1)
        X = vec.fit_transform(norm_for_tfidf)
        n = X.shape[0]
        to_drop = set()
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            sim_block = (X[start:end] @ X.T).toarray()
            for i_rel in range(end - start):
                i = start + i_rel
                if i in to_drop:
                    continue
                matches = np.where(sim_block[i_rel, i + 1:] > cosine_threshold)[0]
                for m in matches:
                    to_drop.add(i + 1 + int(m))
        keep_mask = [idx not in to_drop for idx in range(n)]
        tfidf_dropped = len(to_drop)
        df_result = df_hash_clean[keep_mask].reset_index(drop=True)
    else:
        df_result = df_hash_clean

    return df_result, hash_dropped, tfidf_dropped


def step_2_screening(df_raw):
    print(f"\n{'='*70}\n[TAHAP 2] DATA SCREENING & PELABELAN SENTIMEN\n{'='*70}")
    raw_count = len(df_raw)

    # 1. Hapus null / kosong
    df_clean = df_raw.dropna(subset=['content']).copy()
    df_clean = df_clean[df_clean['content'].astype(str).str.strip() != ''].copy()
    null_dropped = raw_count - len(df_clean)
    after_null = len(df_clean)

    # 2. Hapus duplikat persis
    df_clean = df_clean.drop_duplicates(subset=['content']).copy()
    dedup_dropped = after_null - len(df_clean)
    after_dedup = len(df_clean)

    # 3. Hapus ulasan < 3 kata
    df_clean['word_count'] = df_clean['content'].apply(lambda x: len(str(x).strip().split()))
    df_clean = df_clean[df_clean['word_count'] >= 3].reset_index(drop=True)
    short_dropped = after_dedup - len(df_clean)
    after_short = len(df_clean)

    # 4. Hapus near-duplicate (A.8: hash ternormalisasi + TF-IDF cosine > 0.90)
    df_clean, hash_dup_dropped, tfidf_dup_dropped = detect_near_duplicates(df_clean)
    near_dup_dropped = hash_dup_dropped + tfidf_dup_dropped
    after_near_dup = len(df_clean)

    # Alasan (A.9): Pastikan kolom tanggal ulasan ('at') dan 'appVersion' selalu tersimpan
    # dengan rapi pada seluruh keluaran untuk mendukung analisis tren waktu & versi.
    if 'at' in df_clean.columns:
        df_clean['at'] = pd.to_datetime(df_clean['at'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        df_clean['tanggal_ulasan'] = df_clean['at'].astype(str).str.slice(0, 10)
    else:
        df_clean['at'] = ''
        df_clean['tanggal_ulasan'] = ''

    if 'appVersion' in df_clean.columns:
        df_clean['appVersion'] = df_clean['appVersion'].fillna('Tidak Diketahui').replace('', 'Tidak Diketahui')
    else:
        df_clean['appVersion'] = 'Tidak Diketahui'

    # 5. Pelabelan Sentimen
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

    # Alasan (A.10): Catat dan simpan jumlah data pada setiap tahap screening dalam tabel & CSV.
    screening_rows = [
        {'Tahap': '1. Data Mentah (Raw)', 'Dibuang': 0, 'Sisa': raw_count,
         'Persentase_Sisa': 100.0, 'Keterangan': 'Hasil scraping awal'},
        {'Tahap': '2. Hapus Kosong (Null/Empty)', 'Dibuang': null_dropped, 'Sisa': after_null,
         'Persentase_Sisa': round(after_null / raw_count * 100, 2), 'Keterangan': 'Teks kosong/NaN'},
        {'Tahap': '3. Hapus Duplikat Persis', 'Dibuang': dedup_dropped, 'Sisa': after_dedup,
         'Persentase_Sisa': round(after_dedup / raw_count * 100, 2), 'Keterangan': 'Teks identik persis'},
        {'Tahap': '4. Hapus Terlalu Pendek (<3 kata)', 'Dibuang': short_dropped, 'Sisa': after_short,
         'Persentase_Sisa': round(after_short / raw_count * 100, 2), 'Keterangan': 'Minim konteks keluhan/pujian'},
        {'Tahap': '5. Hapus Near-Duplicate (Cosine > 0.9)', 'Dibuang': near_dup_dropped, 'Sisa': after_near_dup,
         'Persentase_Sisa': round(after_near_dup / raw_count * 100, 2),
         'Keterangan': f'Hash norm: {hash_dup_dropped}, TF-IDF cosine > 0.9: {tfidf_dup_dropped}'},
    ]
    df_screening_summary = pd.DataFrame(screening_rows)
    df_screening_summary.to_csv('ringkasan_screening.csv', index=False, encoding='utf-8')

    print("\n📋 TABEL RINGKASAN TAHAP SCREENING:")
    print(df_screening_summary.to_string(index=False))
    print(f"✅ Data screening disimpan ke '{output_csv}' & 'ringkasan_screening.csv'.")

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


def convert_emojis_to_sentiment(text):
    """Konversi emoji ke kata sentimen Indonesia sebelum karakter non-ASCII dihapus."""
    # Alasan (A.6): Emoji sering mengekspresikan emosi langsung (😡, 👍, 👎) sehingga
    # diubah menjadi token sentimen daripada dibuang mentah.
    for emj, sentiment_word in EMOJI_SENTIMENT_MAP.items():
        if emj in text:
            text = text.replace(emj, sentiment_word)

    if HAS_EMOJI_LIB:
        demojized = emoji.demojize(text, delimiters=(' __EMO_', '__ '))
        def _map_demoji(match):
            name = match.group(1).lower()
            if any(k in name for k in ('angry', 'rage', 'pouting', 'steam')):
                return ' marah '
            if any(k in name for k in ('cry', 'sob', 'sad', 'pensive', 'disappointed', 'pleading')):
                return ' sedih '
            if any(k in name for k in ('thumbs_down', 'pile_of_poo', 'nauseated', 'vomiting')):
                return ' buruk '
            if any(k in name for k in ('thumbs_up', 'star', 'check')):
                return ' bagus '
            if any(k in name for k in ('smile', 'grin', 'happy', 'joy', 'laugh')):
                return ' senang '
            if any(k in name for k in ('heart', 'love')):
                return ' suka '
            if any(k in name for k in ('folded_hands', 'pray')):
                return ' mohon '
            return ' '
        text = re.sub(r'__EMO_([a-zA-Z0-9_&-]+)__', _map_demoji, demojized)
    return text


def normalize_reduplication_and_hyphens(text):
    """Tangani kata ulang ('bagus-bagus', 'bagus2') & lindungi kata berhubung ('tiba-tiba')."""
    # Alasan (A.4): Superscript '²' (mis. mahal², tiba²) sering dipakai di ulasan Indonesia;
    # samakan ke '2' sebelum pembersihan non-ASCII agar pola reduplikasi tertangani konsisten.
    text = text.replace('\u00b2', '2')

    # Alasan (A.4): Normalisasi variasi penulisan kata ulang semu ke bentuk baku bertanda hubung
    # agar "tibatiba" / "tiba2" tidak kehilangan maknanya menjadi token rusak.
    idiom_redup_map = {
        r'\btiba\s*2(?:nya)?\b': 'tiba-tiba',
        r'\btibatiba\b': 'tiba-tiba',
        r'\bgara\s*2(?:nya)?\b': 'gara-gara',
        r'\bgaragara\b': 'gara-gara',
        r'\bsia\s*2\b': 'sia-sia',
        r'\bsiasia\b': 'sia-sia',
        r'\brata\s*2(?:nya)?\b': 'rata-rata',
        r'\bratarata\b': 'rata-rata',
        r'\bmasing\s*2\b': 'masing-masing',
        r'\bsehari\s*2\b': 'sehari-hari',
    }
    for pat, repl in idiom_redup_map.items():
        text = re.sub(pat, repl, text)

    # Lindungi kata bertanda hubung leksikal (tiba-tiba, gara-gara, sia-sia, rata-rata, m-banking, dll.)
    # SEBELUM penyederhanaan kata ulang 'X-X -> X' agar 'tiba-tiba' tetap utuh menjadi satu token.
    protected = {}
    for idx, hw in enumerate(LEXICALIZED_HYPHEN_WORDS):
        if hw in text:
            token_key = f'__HYPH{idx}__'
            protected[token_key] = hw
            text = re.sub(rf'\b{re.escape(hw)}\b', token_key, text)

    # Kata ulang angka 'bagus2' / 'mahal2' / 'ujung2nya' -> bentuk tunggal 'bagus' / 'mahal' / 'ujung'
    text = re.sub(r'\b([a-z]{2,})2(?:nya|an)?\b', r'\1', text)

    # Kata ulang identik bertanda hubung 'bagus-bagus' / 'mahal-mahal' -> 'bagus' / 'mahal'
    text = re.sub(r'\b([a-z]{2,})-\1\b', r'\1', text)

    # Alasan (A.4): Tanda hubung lain (mis. 'di-update', 'se-indonesia') diubah jadi spasi,
    # BUKAN dihapus sehingga kata menempel menjadi 'diupdate' / 'tibatiba'.
    text = re.sub(r'(?<=[a-z])-(?=[a-z])', ' ', text)

    # Hapus tanda baca lain selain underscore dan placeholder
    punct_no_hyphen = string.punctuation.replace('_', '')
    text = text.translate(str.maketrans('', '', punct_no_hyphen))

    for token_key, hw in protected.items():
        text = text.replace(token_key, hw)

    return text


def step_3_preprocessing(df_clean):
    print(f"\n{'='*70}\n[TAHAP 3] PIPELINE PRE-PROCESSING DATA TEKS\n{'='*70}")

    slang_dict = load_slang_dict()
    print(f"📖 Kamus slang dimuat: {len(slang_dict):,} entri tunggal dari 'kamus_slang.json'.")

    # Jalur (b) D.4: Cleaning ringan untuk IndoBERT (tanpa stemming & tanpa stopword removal)
    print("⏳ Menyiapkan Jalur (b) Cleaning Ringan untuk IndoBERT...")
    def clean_for_indobert(text):
        # Alasan (D.4): IndoBERT membutuhkan struktur kalimat alami (imbuhan, stopword, kata kontras),
        # sehingga hanya dilakukan lowercase ringan, hapus URL/@/#, normalisasi huruf berulang & emoji.
        t = str(text).lower().replace('\u00b2', '2')
        t = convert_emojis_to_sentiment(t)
        t = re.sub(r'http\S+|www\S+|https\S+', '', t, flags=re.MULTILINE)
        t = re.sub(r'[@#]\w+', '', t)
        t = re.sub(r'(.)\1{2,}', r'\1', t)
        t = re.sub(r'[^\x00-\x7F]+', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    df_clean['teks_indobert'] = df_clean['content'].apply(clean_for_indobert)

    # 1. Case Folding
    print("⏳ Menjalankan Case Folding...")
    df_clean['case_folded'] = df_clean['content'].apply(lambda x: str(x).lower())

    # 2. Text Cleaning (Emoji -> Sentimen, Huruf Berulang, Kata Ulang, Tanda Hubung)
    print("⏳ Menjalankan Text Cleaning (Emoji, Huruf Berulang (.)\\1{2,}->\\1, Kata Ulang)...")
    def clean_text(text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[@#]\w+', '', text)
        # Alasan (A.6): Konversi emoji menjadi kata sentimen sebelum menghapus non-ASCII
        text = convert_emojis_to_sentiment(text)
        # Alasan (A.4): Normalisasi reduplikasi & tanda hubung sebelum hapus tanda baca
        text = normalize_reduplication_and_hyphens(text)
        # Alasan (A.3): Reduksi >=3 huruf berulang menjadi 1 huruf ('bagusss'->'bagus',
        # 'lemooott'->'lemot') SEBELUM pencocokan slang.
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df_clean['cleaned'] = df_clean['case_folded'].apply(clean_text)

    # 3. Penggantian Frasa (Regex sebelum tokenizing) & Normalisasi Slang + Inggris (B.2, A.7)
    print("⏳ Menjalankan Penggantian Frasa (Regex) & Normalisasi Slang + Kata Inggris...")
    compiled_phrases = [
        (re.compile(rf'\b{re.escape(phrase)}\b'), replacement)
        for phrase, replacement in PHRASE_REPLACEMENTS.items()
    ]

    def normalize_slang_and_phrases(text):
        # Alasan (B.2): Ganti frasa multi-kata seperti 'force close' dengan regex sebelum split token
        for pat, repl in compiled_phrases:
            text = pat.sub(repl, text)
        words = text.split()
        normalized = []
        for w in words:
            mapped = slang_dict.get(w, ENGLISH_TO_INDONESIAN.get(w, w))
            # Alasan (B.3): Token yang dipetakan ke "" (mis. 'nihh') langsung diabaikan
            if mapped:
                normalized.append(mapped)
        return ' '.join(normalized)

    df_clean['slang_normalized'] = df_clean['cleaned'].apply(normalize_slang_and_phrases)

    # 4. Negation Handling Berbasis Whitelist (A.5)
    print("⏳ Menjalankan Negation Handling Berbasis Whitelist...")
    def handle_negation(text):
        # Alasan (A.5): Gabungkan 'negasi_kata' HANYA jika kata berikutnya ada di
        # NEGATION_TARGET_WHITELIST agar tidak menempel ke stopword (jangan_di, tidak_pernah, dll.).
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            if (
                words[i] in NEGATION_WORDS
                and i + 1 < len(words)
                and words[i + 1] in NEGATION_TARGET_WHITELIST
            ):
                result.append(f"{words[i]}_{words[i + 1]}")
                i += 2
            else:
                result.append(words[i])
                i += 1
        return ' '.join(result)

    df_clean['negation_handled'] = df_clean['slang_normalized'].apply(handle_negation)

    # 5. Tokenizing & Stopword Removal (mempertahankan kata kontras, penguat, dan negasi)
    print("⏳ Menjalankan Tokenizing (nltk) & Stopword Removal...")
    nltk_stops = set(stopwords.words('indonesian'))
    sastrawi_stops = set(StopWordRemoverFactory().get_stop_words())
    custom_stops = {
        # Identitas aplikasi (noise di semua word cloud)
        'telkomsel', 'mytelkomsel', 'aplikasi', 'app', 'apk', 'telkom',
        # Partikel dan kata seru
        'nya', 'nih', 'nihh', 'sih', 'deh', 'dong', 'kan', 'tuh', 'lah', 'loh', 'woi',
        'kak', 'min', 'admin', 'bintang', 'kasih',
        # Kata penghubung/fungsional (Alasan A.2: 'banget', 'sangat', 'sekali' DIHAPUS dari sini)
        'kali', 'versi', 'bisa', 'ada', 'dari', 'yang', 'untuk',
        'dengan', 'ini', 'itu', 'di', 'ke', 'pada', 'dan', 'atau',
        'saya', 'kamu', 'kami', 'kita', 'mereka', 'dia', 'kau', 'nge',
        # Filler / kata sehari-hari tanpa makna analitis
        'dah', 'udah', 'udh', 'gitu', 'gito', 'cuman', 'cuma', 'doang',
        'mending', 'kayak', 'kaya', 'biar', 'biarin', 'pas', 'gin', 'begini',
        'ampun', 'mulu', 'terus', 'kesini', 'dikit', 'sedikit',
        'gimana', 'bngt', 'yaa', 'yah', 'ngak', 'nggak',
        'kadang', 'kadangan', 'emang', 'memang', 'aja', 'saja',
        'juga', 'masih', 'sama', 'lagi', 'pernah', 'sekarang',
        'coba', 'pilih', 'bikin', 'main', 'mah', 'suruh', 'maaf',
        'ribu', 'jam', 'minggu', 'kota', 'terima', 'semoga', 'moga',
        'the', 'menit', 'tanggal', 'malam', 'kemarin', 'belah', 'gilir',
        'pokok', 'ulang', 'ubah', 'tinggal', 'bilang', 'beda', 'atur',
        'minimal', 'full', 'sengaja', 'utama', 'butuh', 'alih', 'lihat',
        'dapat', 'hadehh', 'bos', 'mas', 'kah', 'tidak_seperti',
        'jangan_mahal', 'info', 'hasil', 'proses', 'tawar', 'batas',
        'usaha', 'muncul', 'ekonomi', 'indonesia', 'konsumen',
        # Alasan (A.5): 'tidak_ada' & 'tidak_masuk' dihapus dari daftar buang karena bermakna
        'belum_ada', 'tidak_punya', 'tidak_di', 'tidak_ngotak', 'dll',
        'gb', 'rb', 'hp', 'ni', 'ku', 'eh',
    }
    all_stops = nltk_stops.union(sastrawi_stops).union(ENGLISH_BASIC_STOPWORDS).union(custom_stops)
    # Alasan (A.2): Pastikan negasi dan kata kontras/penguat TIDAK masuk daftar stopword final
    all_stops -= NEGATION_WORDS
    all_stops -= CONTRAST_AND_INTENSIFIER_WORDS

    # Alasan (A.2): Tampilkan daftar stopword final dan tandai kata bernuansa sentimen + simpan ke CSV
    sorted_stops = sorted(all_stops)
    stop_records = []
    for w in sorted_stops:
        sumber = []
        if w in nltk_stops:
            sumber.append('NLTK')
        if w in sastrawi_stops:
            sumber.append('Sastrawi')
        if w in ENGLISH_BASIC_STOPWORDS:
            sumber.append('English')
        if w in custom_stops:
            sumber.append('Custom')
        has_sent = w in SENTIMENT_FLAGGED_STOPS
        stop_records.append({
            'kata': w,
            'sumber': '+'.join(sumber),
            'bermakna_sentimen': has_sent,
            'penanda': '[SENTIMEN]' if has_sent else '',
        })
    df_stops = pd.DataFrame(stop_records)
    df_stops.to_csv('daftar_stopword_final.csv', index=False, encoding='utf-8')
    flagged_list = [r['kata'] for r in stop_records if r['bermakna_sentimen']]
    print(f"📋 Daftar Stopword Final: {len(sorted_stops)} kata (disimpan ke 'daftar_stopword_final.csv').")
    print("   Daftar lengkap:", ', '.join(
        f"{r['kata']}{r['penanda']}" for r in stop_records
    ))
    print(f"🏷️  Kata bernuansa sentimen/pragmatik yang ditandai ({len(flagged_list)}): {flagged_list}")

    def tokenize_preserve_hyphen(text):
        # Tokenisasi menjaga token bertanda hubung ('tiba-tiba') dan underscore ('tidak_bisa')
        return [tok for tok in word_tokenize(text) if re.search(r'[a-z]', tok)]

    df_clean['pre_stop_tokens'] = df_clean['negation_handled'].apply(tokenize_preserve_hyphen)
    df_clean['tokens'] = df_clean['pre_stop_tokens'].apply(
        lambda toks: [w for w in toks if w not in all_stops and len(w) > 2]
    )
    df_clean['stopwords_removed'] = df_clean['tokens'].apply(lambda x: ' '.join(x))

    # 6. Optimasi Stemming Sastrawi (Unique Word Caching + Exception List)
    print("⏳ Menjalankan Stemming Sastrawi dengan Unique Word Caching & Exception List...")
    stemmer = StemmerFactory().create_stemmer()

    # Alasan (A.1): Stem juga token pra-stopword agar n-gram bisa dibangun SEBELUM stopword removal
    unique_tokens = set()
    for t_list in df_clean['pre_stop_tokens']:
        unique_tokens.update(t_list)

    print(f"💡 Jumlah kata unik (vocabulary) yang di-stem: {len(unique_tokens):,} kata unik")
    print(f"🛡️ Exception list: {len(STEM_EXCEPTIONS)} kata tidak akan di-stem.")
    start_stem = time.time()
    stem_cache = {}
    for word in tqdm(unique_tokens, desc="Stemming"):
        if word in STEM_EXCEPTIONS:
            stem_cache[word] = word
        elif '_' in word:
            parts = word.split('_', 1)
            base = parts[1]
            if base in STEM_EXCEPTIONS:
                stem_cache[word] = word
            else:
                stem_cache[word] = f"{parts[0]}_{stemmer.stem(base)}"
        else:
            stem_cache[word] = stemmer.stem(word)
    print(f"⚡ Stemming selesai dalam {time.time() - start_stem:.1f} detik!")

    def apply_cache_post_stop(tokens):
        stemmed = []
        for w in tokens:
            root = stem_cache.get(w, w)
            if root not in all_stops and len(root) > 2:
                stemmed.append(root)
        return stemmed

    def build_clause_marked_pre_stop(pre_tokens):
        # Alasan (A.1): Bangun teks pra-stopword yang dipecah pada kata kontras ('tapi', 'padahal', dll.)
        # dengan pemisah ' | ' agar CountVectorizer tidak membentuk bigram semu lintas klausa.
        clauses = []
        current = []
        for w in pre_tokens:
            root = stem_cache.get(w, w)
            if root in CONTRAST_WORDS:
                if current:
                    clauses.append(' '.join(current))
                    current = []
            elif len(root) > 1:
                current.append(root)
        if current:
            clauses.append(' '.join(current))
        return ' | '.join(clauses)

    df_clean['stemmed_tokens'] = df_clean['tokens'].apply(apply_cache_post_stop)
    df_clean['teks_final'] = df_clean['stemmed_tokens'].apply(lambda x: ' '.join(x))
    df_clean['teks_klasik'] = df_clean['teks_final']
    df_clean['teks_ngram_pre_stopword'] = df_clean['pre_stop_tokens'].apply(build_clause_marked_pre_stop)

    # 7. B.4: Pencarian kandidat slang baru secara sistematis terhadap kamus Sastrawi
    print("⏳ Mencari 100 kandidat slang/OOV teratas (di luar kamus kata baku Sastrawi)...")
    sastrawi_vocab = getattr(getattr(stemmer, 'dictionary', None), 'words', set())
    known_vocab = set(sastrawi_vocab).union(slang_dict.keys()).union(slang_dict.values()).union(all_stops)
    oov_counter = Counter()
    oov_examples = {}
    for raw_text, pre_toks in zip(df_clean['content'], df_clean['pre_stop_tokens']):
        for tok in pre_toks:
            root_tok = stem_cache.get(tok, tok)
            if (
                len(tok) >= 2
                and '_' not in tok
                and '-' not in tok
                and tok not in known_vocab
                and root_tok not in sastrawi_vocab
            ):
                oov_counter[tok] += 1
                if tok not in oov_examples:
                    oov_examples[tok] = str(raw_text).replace('\n', ' ')[:100]

    top100_oov = oov_counter.most_common(100)
    df_oov = pd.DataFrame([
        {'Rank': i + 1, 'Token_OOV': tok, 'Frekuensi': cnt, 'Contoh_Konteks': oov_examples.get(tok, '')}
        for i, (tok, cnt) in enumerate(top100_oov)
    ])
    df_oov.to_csv('kandidat_slang_top100.csv', index=False, encoding='utf-8')
    print("📋 Top 100 kandidat slang/OOV disimpan ke 'kandidat_slang_top100.csv'.")

    df_final = df_clean[df_clean['teks_final'].str.strip() != ''].copy().reset_index(drop=True)
    df_final.to_csv('ulasan_terproses.csv', index=False, encoding='utf-8')

    # Alasan (D.4): Simpan dua jalur keluaran secara terpisah (Model Klasik/TF-IDF vs IndoBERT)
    cols_base = [c for c in ['at', 'tanggal_ulasan', 'appVersion', 'score', 'sentiment'] if c in df_final.columns]
    df_final[cols_base + ['teks_klasik']].to_csv('dataset_klasik_tfidf.csv', index=False, encoding='utf-8')
    df_final[cols_base + ['teks_indobert']].to_csv('dataset_indobert.csv', index=False, encoding='utf-8')

    print(f"✅ Pre-processing selesai! {len(df_final):,} ulasan bersih tersimpan di 'ulasan_terproses.csv'.")
    print("📁 Output dua jalur disimpan ke 'dataset_klasik_tfidf.csv' & 'dataset_indobert.csv'.")
    return df_final


def step_3b_label_validation(df_final):
    """Validasi kualitas label rating (C.1 sampel audit 150 ulasan & C.2 sentimen leksikon)."""
    print(f"\n{'='*70}\n[TAHAP 3B] VALIDASI KUALITAS LABEL RATING\n{'='*70}")

    # C.1: Sampel acak 150 ulasan rating 4-5 yang mengandung kata negatif
    neg_seed_words = {
        'mahal', 'lambat', 'lemot', 'lelet', 'buruk', 'jelek', 'tidak_bisa',
        'error', 'bug', 'lag', 'gagal', 'kecewa', 'parah', 'payah', 'susah',
        'sulit', 'sedot', 'hilang', 'rusak', 'boros', 'rugi', 'keluar_paksa',
    }
    raw_neg_pattern = re.compile(
        r'\b(?:' + '|'.join(re.escape(w.replace('_', ' ')) for w in neg_seed_words) + r')\b',
        re.IGNORECASE,
    )

    def _has_neg_signal(row):
        toks = set(str(row.get('negation_handled', '')).split()) | set(str(row.get('teks_final', '')).split())
        if toks & neg_seed_words:
            return True
        return bool(raw_neg_pattern.search(str(row.get('content', ''))))

    mask_r45 = df_final['score'].isin([4, 5])
    df_r45_neg = df_final[mask_r45 & df_final.apply(_has_neg_signal, axis=1)].copy()
    sample_n = min(150, len(df_r45_neg))
    df_audit = df_r45_neg.sample(n=sample_n, random_state=42)[['content', 'score', 'at', 'appVersion']].reset_index(drop=True)
    df_audit.to_csv('audit_rating45_negatif_150.csv', index=False, encoding='utf-8')
    print(f"🔎 Ditemukan {len(df_r45_neg):,} ulasan rating 4-5 bermuatan kata negatif.")
    print(f"📋 Sampel acak {sample_n} ulasan disimpan ke 'audit_rating45_negatif_150.csv' (10 contoh pertama):")
    print(df_audit[['content', 'score']].head(10).to_string(index=True))

    # C.2: Kolom sentimen_leksikon & tingkat ketidaksesuaian dengan rating
    def compute_lexicon_sentiment(text):
        words = str(text).split()
        score_lex = 0
        i = 0
        while i < len(words):
            w = words[i]
            # Alasan (C.2): Tangani negasi yang belum bergabung agar "tidak bagus" dihitung negatif
            if w in NEGATION_WORDS and i + 1 < len(words):
                nxt = words[i + 1]
                if nxt in POSITIVE_LEXICON:
                    score_lex -= 1
                    i += 2
                    continue
                if nxt in NEGATIVE_LEXICON:
                    score_lex += 1
                    i += 2
                    continue
            if w in POSITIVE_LEXICON:
                score_lex += 1
            elif w in NEGATIVE_LEXICON:
                score_lex -= 1
            i += 1
        if score_lex > 0:
            return 'Positif'
        if score_lex < 0:
            return 'Negatif'
        return 'Netral'

    df_final['sentimen_leksikon'] = df_final['negation_handled'].apply(compute_lexicon_sentiment)
    mismatch_rate = (df_final['sentiment'] != df_final['sentimen_leksikon']).mean() * 100
    r45_mask = df_final['score'].isin([4, 5])
    r45_mismatch = (df_final.loc[r45_mask, 'sentimen_leksikon'] == 'Negatif').mean() * 100
    r12_mask = df_final['score'].isin([1, 2])
    r12_mismatch = (df_final.loc[r12_mask, 'sentimen_leksikon'] == 'Positif').mean() * 100

    print(f"\n📊 Tingkat Ketidaksesuaian Rating vs Leksikon (keseluruhan): {mismatch_rate:.2f}%")
    print(f"   - Rating 4-5 (Positif) diprediksi Negatif oleh leksikon : {r45_mismatch:.2f}%")
    print(f"   - Rating 1-2 (Negatif) diprediksi Positif oleh leksikon : {r12_mismatch:.2f}%")
    print("\n📋 Crosstab Rating (baris) vs Sentimen Leksikon (kolom):")
    print(pd.crosstab(df_final['sentiment'], df_final['sentimen_leksikon'], margins=True))
    print(
        "\n💡 Catatan Keterbatasan Proxy Rating & Kelas Netral (Rating 3):\n"
        "   - Rating bintang tidak selalu selaras dengan isi teks (mis. user memberi ★5 agar keluhan dibaca,\n"
        "     atau ★4 berisi pujian sekaligus keluhan harga mahal/sinyal lemot).\n"
        "   - Kelas Netral (rating 3, ~5,8%) sering mencampur pujian + kritik sehingga sangat ambigu.\n"
        "     Untuk klasifikasi biner yang lebih bersih, kelas Netral dapat dipertimbangkan untuk dihapus\n"
        "     atau dievaluasi terpisah dengan metrik macro-F1."
    )

    # Perbarui file output utama dengan kolom sentimen_leksikon
    df_final.to_csv('ulasan_terproses.csv', index=False, encoding='utf-8')
    cols_base = [c for c in ['at', 'tanggal_ulasan', 'appVersion', 'score', 'sentiment', 'sentimen_leksikon'] if c in df_final.columns]
    df_final[cols_base + ['teks_klasik']].to_csv('dataset_klasik_tfidf.csv', index=False, encoding='utf-8')
    df_final[cols_base + ['teks_indobert']].to_csv('dataset_indobert.csv', index=False, encoding='utf-8')
    return df_final


def compute_log_odds_dirichlet(corpus_target, corpus_other, min_total_freq=5, prior_alpha=0.01):
    """Hitung Log-Odds Ratio dengan Informative Dirichlet Prior (Monroe et al., 2008)."""
    # Alasan (D.2): Log-odds ratio berprior Dirichlet menonjolkan kata yang paling membedakan
    # suatu kelas terhadap kelas pembanding tanpa bias terhadap kata berfrekuensi kecil/typo.
    c1 = Counter()
    c2 = Counter()
    for doc in corpus_target.dropna():
        c1.update(str(doc).split())
    for doc in corpus_other.dropna():
        c2.update(str(doc).split())

    total = c1 + c2
    vocab = [w for w, cnt in total.items() if cnt >= min_total_freq]
    if not vocab:
        return []

    n1 = sum(c1[w] for w in vocab)
    n2 = sum(c2[w] for w in vocab)
    # Background prior berbasis frekuensi korpus gabungan + smoothing ringan
    alpha_map = {w: max(prior_alpha, total[w] * 0.1) for w in vocab}
    alpha_0 = sum(alpha_map.values())

    scores = []
    for w in vocab:
        a_w = alpha_map[w]
        y1 = c1[w] + a_w
        y2 = c2[w] + a_w
        denom1 = max(1e-9, (n1 + alpha_0) - y1)
        denom2 = max(1e-9, (n2 + alpha_0) - y2)
        delta = np.log(y1 / denom1) - np.log(y2 / denom2)
        sigma = np.sqrt((1.0 / y1) + (1.0 / y2))
        z_score = float(delta / sigma)
        scores.append((w, z_score, c1[w], c2[w]))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def make_sequential_color_func(cmap_name, min_val=0.45, max_val=0.92):
    """Buat color_func WordCloud agar kata berfrekuensi tinggi memakai warna sedang-gelap."""
    # Alasan (D.1): Colormap bawaan WordCloud mengacak nilai [0.0, 1.0] sehingga kata penting
    # bisa mendapat warna terlalu terang/pudar di latar putih.
    cmap = plt.colormaps.get_cmap(cmap_name)

    def _color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        # Ambil ukuran font relatif (WordCloud mengatur max_font_size saat render)
        norm = min(1.0, max(0.0, (font_size - 10) / 110.0))
        rgba = cmap(min_val + norm * (max_val - min_val))
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        return f"rgb({r}, {g}, {b})"

    return _color_func


def step_4_analysis_and_visualization(df_final):
    print(f"\n{'='*70}\n[TAHAP 4] ANALISIS FREKUENSI, LOG-ODDS, ASPEK & WORD CLOUD\n{'='*70}")

    # ── Unigram Analysis (manual Counter) ────────────────────────────────────
    def get_top_unigrams(corpus, top_n=20):
        words = []
        for doc in corpus:
            words.extend(str(doc).split())
        return Counter(words).most_common(top_n)

    corpus_all = df_final['teks_final']
    corpus_pos = df_final[df_final['sentiment'] == 'Positif']['teks_final']
    corpus_neg = df_final[df_final['sentiment'] == 'Negatif']['teks_final']
    corpus_net = df_final[df_final['sentiment'] == 'Netral']['teks_final']

    top20_all = get_top_unigrams(corpus_all, 20)
    top20_pos = get_top_unigrams(corpus_pos, 20)
    top20_neg = get_top_unigrams(corpus_neg, 20)

    df_top20 = pd.DataFrame({
        'Rank': range(1, 21),
        'Kata (Semua)': [w[0] for w in top20_all],
        'Frekuensi (Semua)': [w[1] for w in top20_all],
        'Kata (Positif)': [w[0] for w in top20_pos],
        'Frekuensi (Positif)': [w[1] for w in top20_pos],
        'Kata (Negatif)': [w[0] for w in top20_neg],
        'Frekuensi (Negatif)': [w[1] for w in top20_neg],
    })
    df_top20.to_csv('top20_unigrams_per_sentimen.csv', index=False, encoding='utf-8')
    print("📁 Tabel Top 20 Unigram disimpan ke 'top20_unigrams_per_sentimen.csv'.")

    # ── Bigram & Trigram Analysis (DARI TEKS SEBELUM STOPWORD + BATAS KLAUSA KONTRAS) ──
    print("⏳ Menjalankan analisis Bigram & Trigram sebelum stopword removal (dengan batas klausa)...")

    def get_top_ngrams_from_clauses(corpus_clauses, ngram_range, top_n=20):
        """Hitung n-gram per klausa (dipisah di ' | ') agar tidak muncul bigram semu lintas kontras."""
        # Alasan (A.1): Memecah setiap ulasan berdasarkan pemisah klausa ' | ' sebelum masuk
        # ke CountVectorizer mencegah pasangan lintas kata kontras seperti 'mahal sinyal'.
        clauses = []
        for doc in corpus_clauses.dropna():
            for seg in str(doc).split(' | '):
                seg = seg.strip()
                if len(seg.split()) >= ngram_range[0]:
                    clauses.append(seg)
        if not clauses:
            return []
        vec = CountVectorizer(ngram_range=ngram_range, token_pattern=r'(?u)\b[a-zA-Z0-9_-]{2,}\b')
        bag = vec.fit_transform(clauses)
        sum_words = bag.sum(axis=0)
        words_freq = [(word, int(sum_words[0, idx])) for word, idx in vec.vocabulary_.items()]
        words_freq.sort(key=lambda x: x[1], reverse=True)
        return words_freq[:top_n]

    ngram_all = df_final['teks_ngram_pre_stopword']
    ngram_pos = df_final[df_final['sentiment'] == 'Positif']['teks_ngram_pre_stopword']
    ngram_neg = df_final[df_final['sentiment'] == 'Negatif']['teks_ngram_pre_stopword']

    top20_bigram_all = get_top_ngrams_from_clauses(ngram_all, (2, 2), 20)
    top20_bigram_pos = get_top_ngrams_from_clauses(ngram_pos, (2, 2), 20)
    top20_bigram_neg = get_top_ngrams_from_clauses(ngram_neg, (2, 2), 20)

    top20_trigram_all = get_top_ngrams_from_clauses(ngram_all, (3, 3), 20)
    top20_trigram_pos = get_top_ngrams_from_clauses(ngram_pos, (3, 3), 20)
    top20_trigram_neg = get_top_ngrams_from_clauses(ngram_neg, (3, 3), 20)

    def pad_ngrams(ngram_list, length):
        padded = ngram_list + [('', 0)] * (length - len(ngram_list))
        return padded[:length]

    max_len_bi = max(len(top20_bigram_all), len(top20_bigram_pos), len(top20_bigram_neg), 1)
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
    def plot_bars(ax, item_list, title, palette, xlabel='Frekuensi'):
        words = [item[0] for item in item_list][::-1]
        vals = [item[1] for item in item_list][::-1]
        sns.barplot(x=vals, y=words, ax=ax, palette=palette)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(xlabel)
        max_v = max(vals) if vals else 1
        for i, v in enumerate(vals):
            lbl = f"{v:,}" if isinstance(v, int) else f"{v:.2f}"
            ax.text(v + (max_v * 0.015), i, lbl, va='center', fontsize=9)

    plot_bars(axes[0], top20_all, 'Top 20 Kata: Keseluruhan Ulasan', 'Blues_d')
    plot_bars(axes[1], top20_pos, 'Top 20 Kata: Ulasan Positif (Rating 4-5)', 'Greens_d')
    plot_bars(axes[2], top20_neg, 'Top 20 Kata: Ulasan Negatif (Rating 1-2)', 'Reds_d')
    plt.tight_layout()
    plt.savefig('barchart_top20_kata.png', dpi=300)
    plt.close()
    print("📁 Bar chart Unigram disimpan ke 'barchart_top20_kata.png'.")

    # ── Bar Chart: Bigram Top 15 ─────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(22, 8), sharey=False)
    plot_bars(axes2[0], top20_bigram_all[:15], 'Top 15 Bigram: Keseluruhan', 'Blues_d')
    plot_bars(axes2[1], top20_bigram_pos[:15], 'Top 15 Bigram: Positif (Rating 4-5)', 'Greens_d')
    plot_bars(axes2[2], top20_bigram_neg[:15], 'Top 15 Bigram: Negatif (Rating 1-2)', 'Reds_d')
    plt.tight_layout()
    plt.savefig('barchart_top15_bigram.png', dpi=300)
    plt.close()
    print("📁 Bar chart Bigram disimpan ke 'barchart_top15_bigram.png'.")

    # ── D.2: Kata Diskriminatif per Kelas (Log-Odds Ratio Informative Dirichlet Prior) ──
    print("⏳ Menghitung Log-Odds Ratio (Informative Dirichlet Prior) per kelas...")
    logodds_pos = compute_log_odds_dirichlet(corpus_pos, df_final[df_final['sentiment'] != 'Positif']['teks_final'])
    logodds_neg = compute_log_odds_dirichlet(corpus_neg, df_final[df_final['sentiment'] != 'Negatif']['teks_final'])
    logodds_net = compute_log_odds_dirichlet(corpus_net, df_final[df_final['sentiment'] != 'Netral']['teks_final'])

    top20_lo_pos = [(w, round(z, 3)) for w, z, _, _ in logodds_pos[:20]]
    top20_lo_neg = [(w, round(z, 3)) for w, z, _, _ in logodds_neg[:20]]
    top20_lo_net = [(w, round(z, 3)) for w, z, _, _ in logodds_net[:20]]

    df_logodds = pd.DataFrame({
        'Rank': range(1, 21),
        'Kata_Positif': [w for w, _ in pad_ngrams(top20_lo_pos, 20)],
        'Z_LogOdds_Positif': [z for _, z in pad_ngrams(top20_lo_pos, 20)],
        'Kata_Netral': [w for w, _ in pad_ngrams(top20_lo_net, 20)],
        'Z_LogOdds_Netral': [z for _, z in pad_ngrams(top20_lo_net, 20)],
        'Kata_Negatif': [w for w, _ in pad_ngrams(top20_lo_neg, 20)],
        'Z_LogOdds_Negatif': [z for _, z in pad_ngrams(top20_lo_neg, 20)],
    })
    df_logodds.to_csv('top20_kata_diskriminatif_logodds.csv', index=False, encoding='utf-8')
    print("📁 Tabel Top 20 Kata Diskriminatif (Log-Odds) disimpan ke 'top20_kata_diskriminatif_logodds.csv'.")

    fig_lo, axes_lo = plt.subplots(1, 3, figsize=(22, 8), sharey=False)
    plot_bars(axes_lo[0], top20_lo_pos, 'Top 20 Kata Diskriminatif: Positif', 'Greens_d', xlabel='Z-score Log-Odds')
    plot_bars(axes_lo[1], top20_lo_net, 'Top 20 Kata Diskriminatif: Netral', 'Oranges_d', xlabel='Z-score Log-Odds')
    plot_bars(axes_lo[2], top20_lo_neg, 'Top 20 Kata Diskriminatif: Negatif', 'Reds_d', xlabel='Z-score Log-Odds')
    plt.tight_layout()
    plt.savefig('barchart_top20_diskriminatif_logodds.png', dpi=300)
    plt.close()
    print("📁 Bar chart Log-Odds disimpan ke 'barchart_top20_diskriminatif_logodds.png'.")

    # ── D.1: Word Clouds (color_func sedang-gelap + sensor kata kasar + versi log-odds) ──
    def build_clean_wc_frequencies(corpus_text, exclude_words):
        freq = Counter()
        # Alasan (D.1): Sensor kata kasar hanya saat menyusun frekuensi Word Cloud
        full_exclude = set(exclude_words).union(PROFANITY_FILTER)
        for doc in corpus_text.dropna():
            for w in str(doc).split():
                if w not in full_exclude:
                    freq[w] += 1
        return freq

    def make_wordcloud_from_freq(freq_dict, title, cmap_name, output_png):
        if not freq_dict:
            print(f"⚠️ Frekuensi kosong, word cloud '{output_png}' tidak dihasilkan.")
            return
        wc = WordCloud(
            width=1200, height=600,
            background_color='white',
            color_func=make_sequential_color_func(cmap_name),
            max_words=120,
            random_state=42,
            collocations=False,
            normalize_plurals=False,
        ).generate_from_frequencies(freq_dict)

        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=15, fontweight='bold', pad=18)
        plt.tight_layout(pad=0)
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🖼️ Word Cloud disimpan ke '{output_png}'.")

    # 1. Word Cloud Keseluruhan (frekuensi, disensor kata kasar)
    make_wordcloud_from_freq(
        build_clean_wc_frequencies(corpus_all, set()),
        'Word Cloud: Keseluruhan Ulasan MyTelkomsel',
        'Blues', 'wordcloud_keseluruhan.png'
    )

    # 2. Word Cloud Positif & Negatif (frekuensi + sentiment-aware filtering + sensor kata kasar)
    pos_exclude = NEGATIVE_LEXICON.union(NEUTRAL_TOPIC_WORDS).union(NEGATION_WORDS).union(CONTRAST_WORDS).union({
        'kurang', 'agak', 'terlalu', 'repot', 'tidak_tahu',
    })
    make_wordcloud_from_freq(
        build_clean_wc_frequencies(corpus_pos, pos_exclude),
        'Word Cloud: Ulasan Positif MyTelkomsel (Rating 4 - 5)',
        'Greens', 'wordcloud_positif.png'
    )

    neg_exclude = POSITIVE_LEXICON.union(NEUTRAL_TOPIC_WORDS)
    make_wordcloud_from_freq(
        build_clean_wc_frequencies(corpus_neg, neg_exclude),
        'Word Cloud: Ulasan Negatif MyTelkomsel (Rating 1 - 2)',
        'Reds', 'wordcloud_negatif.png'
    )

    # 3. Word Cloud Berbobot Log-Odds per Kelas (D.1)
    lo_pos_weights = {w: z for w, z, _, _ in logodds_pos if z > 0 and w not in PROFANITY_FILTER}
    lo_neg_weights = {w: z for w, z, _, _ in logodds_neg if z > 0 and w not in PROFANITY_FILTER}
    make_wordcloud_from_freq(
        lo_pos_weights,
        'Word Cloud Berbobot Log-Odds: Ulasan Positif',
        'Greens', 'wordcloud_positif_logodds.png'
    )
    make_wordcloud_from_freq(
        lo_neg_weights,
        'Word Cloud Berbobot Log-Odds: Ulasan Negatif',
        'Reds', 'wordcloud_negatif_logodds.png'
    )

    # ── D.3: Analisis Aspek Berbasis Kamus Kata Kunci ─────────────────────────
    print("⏳ Menjalankan Analisis Aspek Berbasis Kamus Kata Kunci...")
    aspect_rows = []
    total_docs = len(df_final)
    for aspect_name, kw_set in ASPECT_KEYWORDS.items():
        # Cek pada token terproses + pra-stopword ternormalisasi agar kata kunci seperti
        # 'tolong', 'bantu', 'cs' tetap tertangkap akurat.
        has_aspect = df_final.apply(
            lambda r: bool(
                (set(str(r.get('teks_final', '')).split()) | set(str(r.get('negation_handled', '')).split()))
                & kw_set
            ),
            axis=1,
        )
        df_final[f'aspek_{aspect_name}'] = has_aspect
        sub = df_final[has_aspect]
        cnt = len(sub)
        sent_dist = sub['sentiment'].value_counts().to_dict()
        aspect_rows.append({
            'Aspek': aspect_name,
            'Jumlah_Ulasan': cnt,
            'Proporsi_Total_Pct': round(cnt / total_docs * 100, 2) if total_docs else 0.0,
            'Negatif': sent_dist.get('Negatif', 0),
            'Netral': sent_dist.get('Netral', 0),
            'Positif': sent_dist.get('Positif', 0),
            'Negatif_Pct': round(sent_dist.get('Negatif', 0) / cnt * 100, 2) if cnt else 0.0,
            'Netral_Pct': round(sent_dist.get('Netral', 0) / cnt * 100, 2) if cnt else 0.0,
            'Positif_Pct': round(sent_dist.get('Positif', 0) / cnt * 100, 2) if cnt else 0.0,
        })

    df_aspect = pd.DataFrame(aspect_rows)
    df_aspect.to_csv('analisis_aspek_sentimen.csv', index=False, encoding='utf-8')
    print("\n📋 TABEL ANALISIS ASPEK VS SENTIMEN:")
    print(df_aspect.to_string(index=False))

    fig_asp, axes_asp = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x='Aspek', y='Proporsi_Total_Pct', data=df_aspect, ax=axes_asp[0], palette='Blues_d')
    axes_asp[0].set_title('Proporsi Ulasan yang Membahas Tiap Aspek (%)', fontsize=13, fontweight='bold')
    axes_asp[0].set_ylabel('Persentase Ulasan (%)')
    for i, row in df_aspect.iterrows():
        axes_asp[0].text(i, row['Proporsi_Total_Pct'] + 1.0,
                         f"{row['Proporsi_Total_Pct']:.1f}%\n({row['Jumlah_Ulasan']:,})",
                         ha='center', fontsize=9)

    x_idx = np.arange(len(df_aspect))
    neg_p = df_aspect['Negatif_Pct'].values
    net_p = df_aspect['Netral_Pct'].values
    pos_p = df_aspect['Positif_Pct'].values
    axes_asp[1].bar(x_idx, neg_p, label='Negatif', color='#e74c3c')
    axes_asp[1].bar(x_idx, net_p, bottom=neg_p, label='Netral', color='#f39c12')
    axes_asp[1].bar(x_idx, pos_p, bottom=neg_p + net_p, label='Positif', color='#2ecc71')
    axes_asp[1].set_xticks(x_idx)
    axes_asp[1].set_xticklabels(df_aspect['Aspek'])
    axes_asp[1].set_ylim(0, 108)
    axes_asp[1].set_ylabel('Proporsi Sentimen (%)')
    axes_asp[1].set_title('Distribusi Sentimen per Aspek (%)', fontsize=13, fontweight='bold')
    axes_asp[1].legend(loc='upper right')
    for i in range(len(df_aspect)):
        axes_asp[1].text(i, neg_p[i] / 2, f"{neg_p[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        if pos_p[i] >= 8:
            axes_asp[1].text(i, neg_p[i] + net_p[i] + pos_p[i] / 2, f"{pos_p[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('grafik_analisis_aspek.png', dpi=300)
    plt.close()
    print("📁 Grafik aspek disimpan ke 'grafik_analisis_aspek.png'.")


def step_5_modeling_prep(df_final):
    """Siapkan split train/test stratified 80/20, class_weight, contoh oversampling train, & penjelasan macro-F1."""
    print(f"\n{'='*70}\n[TAHAP 5] PERSIAPAN DATASET MODELING (SPLIT, CLASS WEIGHT, MACRO-F1)\n{'='*70}")

    # Alasan (D.5): Encoding label numerik standar untuk scikit-learn & IndoBERT
    label_map = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
    df_model = df_final.copy()
    df_model['label'] = df_model['sentiment'].map(label_map)

    keep_cols = [c for c in [
        'at', 'tanggal_ulasan', 'appVersion', 'score', 'sentiment',
        'label', 'sentimen_leksikon', 'teks_klasik', 'teks_indobert'
    ] if c in df_model.columns]
    df_model = df_model[keep_cols]

    # Stratified train/test split 80/20 dengan random_state=42
    df_train, df_test = train_test_split(
        df_model,
        test_size=0.20,
        stratify=df_model['label'],
        random_state=42,
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train.to_csv('train_data.csv', index=False, encoding='utf-8')
    df_test.to_csv('test_data.csv', index=False, encoding='utf-8')

    # Ekspor subset konsisten (bebas kontradiksi langsung rating vs sentimen leksikon)
    mask_contradict = (
        ((df_model['sentiment'] == 'Positif') & (df_model['sentimen_leksikon'] == 'Negatif')) |
        ((df_model['sentiment'] == 'Negatif') & (df_model['sentimen_leksikon'] == 'Positif'))
    )
    df_consistent = df_model[~mask_contradict].reset_index(drop=True)
    df_consistent.to_csv('ulasan_bersih_konsisten.csv', index=False, encoding='utf-8')
    print(f"🧹 Subset konsisten (tanpa kontradiksi rating vs teks): {len(df_consistent):,}/{len(df_model):,} baris "
          f"(dibuang {int(mask_contradict.sum()):,} baris kontradiktif) -> 'ulasan_bersih_konsisten.csv'.")
    print(f"✅ Split stratified 80/20 selesai: Train = {len(df_train):,}, Test = {len(df_test):,}")
    print("   Distribusi Train:", df_train['sentiment'].value_counts().to_dict())
    print("   Distribusi Test :", df_test['sentiment'].value_counts().to_dict())

    # Hitung class_weight seimbang hanya dari data train
    classes = np.array([0, 1, 2])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=df_train['label'].values)
    class_weight_dict = {int(c): round(float(w), 4) for c, w in zip(classes, weights)}
    inv_map = {v: k for k, v in label_map.items()}
    readable_weights = {inv_map[k]: v for k, v in class_weight_dict.items()}
    print(f"⚖️  Contoh class_weight ('balanced') untuk model klasik/PyTorch: {class_weight_dict} -> {readable_weights}")

    # Alasan (D.5): Oversampling HANYA boleh dilakukan pada data TRAIN setelah split,
    # agar data TEST tetap mencerminkan distribusi asli dan terhindar dari data leakage.
    max_train_cnt = df_train['label'].value_counts().max()
    train_oversampled = pd.concat([
        grp.sample(n=max_train_cnt, replace=True, random_state=42)
        for _, grp in df_train.groupby('label')
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"🔁 Contoh oversampling (HANYA Train): {len(df_train):,} -> {len(train_oversampled):,} baris "
          f"(Test tetap {len(df_test):,} baris tanpa duplikasi).")
    print(
        "\n📏 Metrik Evaluasi Utama — Macro-F1:\n"
        "   Karena distribusi kelas sangat timpang (Negatif ~78%, Netral ~6%, Positif ~16%),\n"
        "   Accuracy mudah menipu (menebak semua Negatif sudah memberi ~78% akurasi).\n"
        "   Macro-F1 merata-ratakan skor F1 tiap kelas secara setara sehingga kesalahan pada\n"
        "   kelas minoritas (Positif & Netral) langsung terukur."
    )


def main():
    print("="*70)
    print("  PIPELINE ANALISIS ULASAN APLIKASI MYTELKOMSEL (LOCAL RUNNER)")
    print("="*70)
    setup_nltk()
    df_raw = step_1_scraping(target_reviews=10000)
    df_clean = step_2_screening(df_raw)
    df_final = step_3_preprocessing(df_clean)
    df_final = step_3b_label_validation(df_final)
    step_4_analysis_and_visualization(df_final)
    step_5_modeling_prep(df_final)
    print(f"\n{'='*70}\n🎉 SELURUH TAHAPAN SELESAI DENGAN SUKSES!\n{'='*70}")


if __name__ == '__main__':
    main()
