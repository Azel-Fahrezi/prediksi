# Di Sarankan Agar Membuat Virtual Environment

## **[Linux]**

1. Pastikan kamu sudah menginstal python3 dan python3-venv (untuk membuat virtual environment). Jika belum, instal dengan perintah berikut:
    ```bash
    sudo apt update
    sudo apt install python3 python3-venv
    ```

2. Setelah itu, buat virtual environment bernama `myenv` di dalam direktori proyek:
    ```bash
    python3 -m venv myenv
    ```

## **[Windows]**

1. **Instalasi Python**  
   Pastikan Python sudah terinstal di komputer Windows kamu. Jika belum, download dan instal Python dari situs resmi [Python for Windows](https://www.python.org/downloads/windows/). Pastikan untuk mencentang pilihan **Add Python to PATH** saat menginstal.

2. **Buat Virtual Environment**  
   Buka Command Prompt atau PowerShell, lalu navigasikan ke direktori proyek di mana kamu ingin membuat virtual environment. Misalnya, jika proyekmu berada di `C:\Users\Azel\Documents\Projek`, gunakan perintah berikut:
    ```bash
    cd C:\Users\Azel\Documents\Projek
    ```

    Setelah itu, buat virtual environment menggunakan perintah ini:
    ```bash
    python -m venv myenv
    ```

    Perintah ini akan membuat sebuah folder baru bernama `myenv` di dalam direktori proyek yang berisi lingkungan Python terisolasi.

## **Menggunakan Virtual Environment di VSCode**

Jika kamu menggunakan VSCode, pastikan interpreter Python diatur untuk menggunakan virtual environment yang telah dibuat. Pilih interpreter dengan menekan `Ctrl+Shift+P`, ketik "Python: Select Interpreter", dan pilih `myenv/bin/python`.

## Library yang Digunakan

1. **Flask**  
   Flask adalah framework micro web untuk Python yang digunakan untuk membangun aplikasi web.
   
2. **Pandas**  
   Pustaka untuk manipulasi dan analisis data, terutama digunakan untuk memanipulasi data dalam format tabel (DataFrame).

3. **OS**  
   Modul untuk berinteraksi dengan sistem operasi, digunakan untuk menangani path dan operasi file.

4. **NumPy**  
   Pustaka untuk komputasi numerik dan manipulasi array multidimensi.

5. **Scikit-learn**  
   Pustaka untuk machine learning. Digunakan untuk standar skala data dan model Support Vector Regression (SVR).

6. **TensorFlow**  
   Framework open-source untuk deep learning, digunakan untuk membuat dan melatih model neural network.

7. **Keras**  
   API deep learning tingkat tinggi yang berjalan di atas TensorFlow untuk membangun dan melatih model ANN.

## Install Library dan Menjalankan Code
Pastikan virtual environment sudah aktif. Kemudian, jalankan perintah berikut untuk menginstal semua pustaka yang dibutuhkan:

```bash
pip install flask pandas numpy scikit-learn tensorflow
```
Kemudian untuk menjalankannya
```bash
python app.py
```


