# Proyek Akhir: Klasifikasi Gambar

## Deskripsi Proyek

Proyek ini merupakan implementasi dari klasifikasi gambar menggunakan dataset "Rock Paper Scissors". Tujuan utamanya adalah untuk mengembangkan model machine learning yang dapat membedakan antara gambar batu, kertas, dan gunting.

## Persyaratan

- Python 3.x
- TensorFlow
- Split Folders
- Matplotlib

## Langkah-langkah

1. **Persiapan Dataset:**
   - Unduh dataset dari [tautan ini](https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip) dan ekstrak ke dalam direktori proyek.

2. **Augmentasi dan Pembagian Dataset:**
   - Dataset akan dibagi menjadi set pelatihan dan validasi dengan ukuran validasi sebesar 40% dari total dataset.
   - Augmentasi gambar dilakukan untuk meningkatkan variasi data.

3. **Pemodelan dan Pelatihan:**
   - Model klasifikasi gambar menggunakan arsitektur Convolutional Neural Network (CNN) dengan bantuan TensorFlow.
   - Model akan dilatih menggunakan data pelatihan dengan target akurasi minimal 85%.

4. **Evaluasi Model:**
   - Kurva pelatihan dan validasi akan ditampilkan untuk memantau kinerja model.
   - Model juga dapat diprediksi dengan mengunggah gambar baru.

## Cara Menjalankan

1. Pastikan semua persyaratan terpenuhi dengan menginstal paket-paket yang diperlukan.
2. Unduh dan ekstrak dataset ke dalam direktori proyek.
3. Jalankan script Python pada lingkungan yang mendukung TensorFlow.
4. Ikuti langkah-langkah yang dijelaskan dalam script untuk melatih dan mengevaluasi model, serta melakukan prediksi dengan gambar baru.

## Kontribusi

Kontribusi terhadap proyek ini sangat dianjurkan. Silakan buka *issue* atau ajukan *pull request* untuk saran perbaikan atau penambahan fitur.

## Lisensi

Proyek ini dilisensikan di bawah [nama lisensi/link].
