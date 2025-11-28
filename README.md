# $${\color{blue}\space📘\space README\space –\space Etapa\space 3:\space Analiza\space si\space Pregatirea\space Setului\space de\space Date\space pentru\space Retele\space Neuronale}$$

**$${\color{green}Disciplina:}$$** Rețele Neuronale  
**$${\color{green}Institutie:}$$** POLITEHNICA București – FIIR  
**$${\color{green}Student:}$$:** Balu Adrian-David  
**$${\color{green}Data:}$$** 21.11.2025  

---

## $${\color{red}INTRODUCERE}$$

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  $${\color{red}1.\space STRUCTURA\space REPOSITORY-ULUI\space GITHUB\space (VERSIUNEA\space ETAPEI\space 3)}$$
```
UnderMyAISkin-RN/
├── README.md
├── docs/
│   └── datasets/          # descrierea setului UCI Dermatology, surse
├── data/
│   ├── raw/               # dermatology.data (brut)
│   ├── processed/         # date curățate, scalate, cu doar 12 atribute
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # cod Python pentru curățarea și scalarea datelor
│   ├── data_acquisition/  # script pentru citirea datelor UCI
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # configurație preprocesare/model
└── requirements.txt       # dependențe Python (Pandas, Numpy)
```
---

## $${\color{red}2.\space DESCRIEREA\space SETULUI\space DE\space DATE:\space UCI\space Dermatology}$$

### $${\color{green}2.1. \space Sursa\space Datelor}$$

* **Origine:** UCI Machine Learning Repository – Dermatology Data Set
* **Modul de achiziție:** ☒ Fișier extern (dermatology.data)
* **Perioada / condițiile colectării:** Date colectate inițial pentru clasificarea bolilor eritremato-scuamoase, colectate pe durata a mai mulți ani și donate ca set de date în data de 31.12.1997.

### $${\color{green}2.2. \space Caracteristicile\space Dataset-ului}$$

* **Număr total de observații:** 366
* **Număr de caracteristici (features) în setul original:** 34
* **Număr de caracteristici (features) utilizate (Input RN):** 12 (doar atribute clinice ce se pot determina de la distanță, non-invaziv)
* **Tipuri de date:** ☒ Numerice (majoritatea discrete 0-3) / ☐ Categoriale (Clasa de ieșire)
* **Format fișiere:** ☒ TXT / ☒ Altele: Format tabular fără antet (necesită preprocesare Pandas)

### $${\color{green}2.3. \space Descrierea\space Fiecarei\space Caracteristici}$$

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
| :--- | :--- | :--- | :--- | :--- |
| `erythema` | numeric (discret) | – | Intensitatea roșeții/eritemului. | 0–3 |
| `scaling` | numeric (discret) | – | Intensitatea descuamării (scuame). | 0–3 |
| `definite borders` | numeric (discret) | – | Claritatea marginilor leziunii. | 0–3 |
| `itching` | numeric (discret) | – | Intensitatea pruritului (mâncărime). | 0–3 |
| `koebner phenomenon` | numeric (discret) | – | Prezența fenomenului Koebner. | 0–3 |
| `polygonal papules` | numeric (discret) | – | Prezența papulelor poligonale. | 0–3 |
| `follicular papules` | numeric (discret) | – | Prezența papulelor foliculare. | 0–3 |
| `oral mucosal involvement` | numeric (discret) | – | Afectarea mucoasei orale. | 0–3 |
| `knee and elbow involvement` | numeric (discret) | – | Afectarea zonelor tipice (genunchi/cot). | 0–3 |
| `scalp involvement` | numeric (discret) | – | Afectarea scalpului. | 0–3 |
| `family history` | binar | – | Istoric familial de boală. | 0 sau 1 |
| `Age` (Vârsta) | numeric (liniar) | ani | Vârsta pacientului. | >0 (continuu) |
| **Clasa de Ieșire** | categorial | – | Diagnosticul final (Cele 6 boli). | {1, 2, 3, 4, 5, 6} |

**Fișier recomandat:**  `data/README.md`

---

##  $${\color{red}3.\space ANALIZA\space EXPLORATORIE\space A\space DATELOR\space (EDA)\space -\space SINTETIC}$$

### $${\color{green}3.1. \space Statistici\space Descriptive\space Aplicate}$$

* **Medie, mediană, deviație standard:** Calculul acestor indicatori, în special pentru atributul continuu `Age`, pentru a determina valoarea optimă de imputare pentru datele lipsă.
* **Min–max și quartile:** Determinarea domeniului de valori și a quartilelor (Q1, Q3) pentru toate caracteristicile, esențială pentru a pregăti etapa de scalare a datelor și pentru a înțelege distribuția.
* **Distribuții pe caracteristici:** Analiza vizuală a distribuției atributelor discrete (0-3) pentru a înțelege frecvența simptomelor, și a distribuției claselor de ieșire (dezechilibrul de clasă).
* **Identificarea outlierilor:** Aplicarea metodelor statistice pentru detectarea valorilor extreme, în special în coloana `Age`, înainte de scalare și antrenarea Rețelei Neuronale.

### $${\color{green}3.2. \space Analiza\space Calitatii\space Datelor}$$

* **Detectarea valorilor lipsă:** Coloana `Age` conține valori lipsă notate cu `?` 
* **Detectarea valorilor inconsistente sau eronate:** Nu există valori eronate (non-numerice) în atributele (0-3), dar este imperios necesară gestionarea simbolului `?`
* **Identificarea caracteristicilor neutilizate din setul de date:** Cele 22 de atribute histopatologice au fost excluse (Feature Selection) pentru a menține modelul non-invaziv.

### $${\color{green}3.3. \space Probleme\space Identificate}$$

* **Tratarea valorilor lipsă în `Age`:** Aproximativ 2% dintre valori (8 din 366) 
* **Dezechilibru între clase:** Deși setul are 6 clase, repartizarea cazurilor este inegală (Clasa 1 - Psoriazis este dominantă), ceea ce necesită atenție în faza de antrenare și în cea de evaluare.

---

##  $${\color{red}4.\space PREPROCESAREA\space DATELOR}$$

### $${\color{green}4.1. \space Curatarea\space Datelor}$$

* **Eliminare duplicatelor:** Se verifică integritatea datelor, dar folosind un set deja prestabilit și consacrat, șansele de duplicate sunt minime.
* **Tratarea valorilor lipsă:**
  * Feature A - imputare artificială `Age`:  Valoarea `?` va fi înlocuită cu mediana vârstelor cunoscute.
* **Selecția de Caracteristici:** Se rețin exclusiv cele 12 atribute clinice și se elimină restul de 22 de atribute histopatologice.

### $${\color{green}4.2. \space Transformarea\space Caracteristicilor}$$

* **Normalizare:** Se aplică MinMaxScaler pe cele 12 atribute de intrare. Aceasta este esențială, în special pentru variabila `Age`, care are un domeniu mult mai mare (liniar) decât celelalte atribute (0-3).
* **Encoding pentru variabile categoriale:** Coloana de Clasă (1-6) va fi transformată folosind `One-Hot Encoding` (sau `to_categorical` în Keras) pentru a fi compatibilă cu funcția de pierdere (cuantificarea predicțiilor greșite) a rețelei neuronale.

### $${\color{green}4.3. \space Structurarea\space Seturilor\space De \space Date}$$

**Împărțire recomandată:**
* 80% – Antrenare
* 10% – Validare
* 10% – Testare

**Principii respectate:**
* Stratificare pentru clasificare: Seturile vor fi împărțite folosind Stratified Sampling (pe baza clasei de ieșire) pentru a menține proporția inegală a claselor în fiecare subset de date.
* Fără scurgere de informație (data leakage): Scalarea datelor va fi calculată DOAR pe setul de Train și apoi aplicată pe seturile Validation și Test.

### $${\color{green}4.4. \space Salvarea\space Rezultatelor\space Preprocesării}$$

* Datele preprocesate (scalate și encodate) vor fi salvate ca fișiere .`csv` sau `.pkl`.

---

##  $${\color{red}5.\space FISIERE\space GENERATE\space IN\space ACEASTA\space ETAPA}$$

* `data/raw/dermatology.data` – Date brute
* `data/processed/dermatology_12features.csv` – Date curățate & scalate (X)
* `data/processed/dermatology_labels_onehot.csv` – Etichete `One-Hot` (Y)
* `data/train/X_train.csv`, `data/validation/X_val.csv`, etc. – Seturi finale
* `src/preprocessing/preprocessing_script.py` - Codul Python care stă la baza proiectului
* `requirements.txt` - Dependențe (Pandas, Numpy)
* `README.md` – Descrierea dataset-ului

---

##  $${\color{red}6.\space STARE\space ETAPA}$$

- [X] Structură repository configurată
- [X] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Documentație actualizată în README + `data/README.md`

---
