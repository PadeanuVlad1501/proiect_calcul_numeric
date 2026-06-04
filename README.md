# Atac Diferențial de Putere (DPA) folosind SVD Redus

Acest proiect implementează un atac de canal secundar de tip DPA asupra algoritmului de criptare AES-128, utilizând o abordare de filtrare a semnalului prin Descompunerea la Valori Singulare Redusă (Truncated SVD / SVD Redus). 

## 📂 Fişierele Proiectului

Proiectul este modularizat și conține următoarele fișiere principale:

* **`svd_core.py`**: Este nucleul numeric al proiectului. Acesta conține implementarea algoritmului SVD "de la zero", folosind reflexii Householder pentru tridiagonalizare și iterația QR pentru diagonalizare.
* **`ascad_handler.py`**: Reprezintă modulul de gestionare a datelor. Extrage "Fereastra de Interes" (Point of Interest) din baza de date hardware sau generează date sintetice de rezervă în cazul în care fișierul de date real lipsește.
* **`dpa_pipeline.py`**: Este orchestratorul proiectului. Acesta reunește modulele anterioare pentru a executa calculul corelației Pearson (CPA), a aplica filtrarea SVD și a genera graficele vizuale cu rezultatele atacului.
* **`ASCAD.h5`** *(Fișier de intrare)*: Baza de date publică ce conține urmele reale de consum de putere capturate de pe un microcontroler ATmega8515 în timpul execuției AES-128.
* **`raport_dpa.png`** *(Fișier generat)*: Raportul grafic final care afișează efectul filtrării SVD și rezultatul corelației Pearson asupra cheilor candidate.

## 🧠 Concepte de Bază: AES și CPA

**AES (Advanced Encryption Standard):**
Este un algoritm standard de criptare, considerat extrem de sigur împotriva atacurilor de tip forță brută matematică, oferind $2^{128}$ chei posibile. Cu toate acestea, implementarea fizică a algoritmului pe un cip introduce "canale secundare". Atunci când procesorul hardware manipulează date (cum ar fi trecerea prin S-Box-ul AES), consumul său de curent variază în funcție de numărul de biți de `1` din datele procesate, concept cunoscut sub numele de Greutate Hamming (Hamming Weight).

**CPA (Correlation Power Analysis):**
CPA este o metodă statistică din categoria atacurilor de canal secundar. Funcționează astfel:
* Atacatorul generează un model "ipotetic" al consumului de curent pentru toate cele 256 de valori posibile ale unui octet din cheie. 
* Se calculează corelația Pearson între acest consum teoretic (ipotetic) și consumul real, măsurat fizic cu un osciloscop, pe parcursul rulării algoritmului AES.
* Ipoteza (cheia) care obține cel mai mare scor absolut de corelație cu datele reale este considerată a fi cheia secretă corectă.

## 🔍 De ce este SVD util în DPA?

Urmele de putere capturate de pe un dispozitiv fizic sunt pline de zgomot care maschează scurgerea reală de informație criptografică. Zgomotul provine în principal din:
1. **Macro-zgomotul de ceas**: Componenta dominantă care acoperă aproape total semnalul util.
2. **Zgomot termic și electronic**: Variații aleatorii.

Aici intervine SVD Trunchiat (Truncated SVD), care acționează ca un filtru matematic extrem de puternic:
* Prin descompunerea SVD a matricei urmelor de curent, macro-zgomotul de ceas este concentrat în prima componentă principală (valoarea singulară dominantă, $\sigma_1$) deoarece este perfect corelat pe toate urmele.
* Prin eliminarea acestei prime componente și păstrarea doar a componentelor de rang mic (care conțin variațiile fine de putere), SVD reconstruiește o matrice "curată" a consumului de curent.
* Această filtrare îmbunătățește semnificativ raportul semnal/zgomot, permițând atacului CPA să identifice corect cheia secretă cu mult mai puține erori.
