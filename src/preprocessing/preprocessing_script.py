# Importare librarii utile
import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Definirea căilor și structurii de directoare
DATA_ROOT = 'data'
PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed')
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VALIDATION_DIR = os.path.join(DATA_ROOT, 'validation')
TEST_DIR = os.path.join(DATA_ROOT, 'test')
DATA_PATH = os.path.join(DATA_ROOT, 'raw', 'dermatology.data')

# ----------------- FUNCTIE DE INIȚIALIZARE DIRECTOARE -----------------
def create_required_directories():
    """Creează directoarele necesare daca nu exista, pentru a preveni OSError."""
    for dir_path in [
        os.path.join(DATA_ROOT, 'raw'),
        PROCESSED_DIR, 
        TRAIN_DIR, 
        VALIDATION_DIR, 
        TEST_DIR
    ]:
        os.makedirs(dir_path, exist_ok=True)
    print("Structura de directoare a fost verificată/creată.")

# -------------------------- RULARE LOGICĂ --------------------------
create_required_directories() # APEL NOU

# 1. Definirea Coloanelor/Antet
# Setul de date nu are antet, dar in documentatie (dermatology.names)
# am 34 de atribute + Clasa de Ieșire (a 35-a coloană).

# Vom folosi 35 de nume generice pentru a citi datele
column_names = [f'A{i}' for i in range(1, 35)] + ['Class'] 

# 2. Încărcarea Datelor
# Specificăm '?' de la Age unde nu stim varsta ca valoare NaN pentru ca Pandas sa o recunoasca
try:
    df = pd.read_csv(DATA_PATH, header=None, names=column_names, na_values='?')
    print("DataFrame încărcat cu succes.")
except FileNotFoundError:
    print(f"Eroare: Fisierul nu a fost găsit la calea specificată ({DATA_PATH}). Asigurați-vă că este în directorul corect.")
    exit()

# 3. Implementarea Modelului Triage)
# Folosim doar cele 12 atribute clinice non-invazive(plus Clasa de Ieșire)
# Bazat pe documentația din dermatology.names:
# A1-A10 (clinice 0-3), A11 (family history 0/1), A34 (Age - linear)
input_features_to_keep = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A34'
]

# Clasa de Ieșire pentru selecția finală
selected_df = df[input_features_to_keep + ['Class']].copy()

# 4. Curățarea Datelor: Tratarea Valorilor Lipsă
# Valori lipsă sunt doar în coloana 'A34' (Age) [cite: 21]
# Tratăm '?' (care acum sunt NaN) prin imputare cu mediana, conform README.

# Calcul mediană pe coloana 'Age' (A34)
median_age = selected_df['A34'].median()

# Înlocuirea dinamică a valorilor lipsă cu mediana (folosind .loc pentru a evita FutureWarning)
selected_df.loc[:, 'A34'] = selected_df['A34'].fillna(median_age)

# Verific dacă mai sunt valori lipsă
print("\nVerificare Valori Lipsă după imputare:")
print(selected_df.isnull().sum())
print(f"Valoarea medianei folosită pentru imputarea Age: {median_age}")

# 5. Afișarea Datelor Curățate (Sintetic)
print("\nPrimele 5 rânduri din DataFrame-ul curățat și selectat (12 atribute):")
print(selected_df.head())

# 6. Stocarea Datelor Curățate
cleaned_path = os.path.join(PROCESSED_DIR, 'dermatology_12features_cleaned.csv')
selected_df.to_csv(cleaned_path, index=False)
print(f"\nDatele curățate au fost salvate în: {cleaned_path}")


# 7. Separare X (Input) si y (Output)
# X - cele 12 atribute clinice (A1-A11, A34)
X = selected_df.drop('Class', axis=1)

# y - diagnosticul de la 1 la 6, in functie de ce boala de piele e cea mai probabila
# Scădem 1 din clasă pentru a o aduce în domeniul 0-5 (necesar pentru One-Hot Encoding/Keras)
# Deci Clasele vor fi: 0 (psoriazis), 1 (seboreic), 2 (lichen plan), 3 (pityriasis rosea), 4 (cronică eczemă), 5 (pityriasis rubra pilaris)
y = selected_df['Class'] - 1 

print("\nForma (shape) inițială a datelor:")
print(f"X: {X.shape}, y: {y.shape}")


# 8. One-Hot Encoding (OHE) al Etichetelor de Ieșire (y)
# Folosim OneHotEncoder pentru a transforma etichetele (0-5) in vectori binari, 6 coloane (pentru Keras/TensorFlow).
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

print(f"Forma y după One-Hot Encoding: {y_encoded.shape}")


# 9. Împărțirea Seturilor de Date (Stratified Sampling) ---
# Implementăm împărțirea 80% Train, 10% Validation, 10% Test cu stratificare (cea din documentul README.md de pe GitHub).
# Am nevoie de stratificare deoarece am dezechilibru de clase (unele boli sunt mai preponderente in setul meu de date)
# Pasul A: Split inițial (Train vs. Temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y
)

# Pasul B: Split Temp (Validation vs. Test)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp.argmax(axis=1)
)

print("\nFormele seturilor după split (aprox. 80/10/10):")
print(f"Train: {X_train.shape[0]} ({round(X_train.shape[0]/X.shape[0]*100)}%)")
print(f"Validation: {X_val.shape[0]} ({round(X_val.shape[0]/X.shape[0]*100)}%)")
print(f"Test: {X_test.shape[0]} ({round(X_test.shape[0]/X.shape[0]*100)}%)")


# 10. Normalizarea (Scalarea Min-Max) a Atributelor de Intrare (X) ---
# Scalarea se face DOAR pe X_train pentru a preveni Data Leakage si implicit incorectitudinea setului de date
scaler = MinMaxScaler()

# Aplică scalarea (fit_transform) doar pe setul de antrenare
X_train_scaled = scaler.fit_transform(X_train)

# Aplică scalarea (transform) pe Validation și Test (folosind parametrii învățați din Train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nDatele sunt acum curățate, scalate și împărțite!")


# 11. Salvarea Seturilor Preprocesate
# Salvăm seturile în folderele dedicate

np.savetxt(os.path.join(TRAIN_DIR, 'X_train.csv'), X_train_scaled, delimiter=",")
np.savetxt(os.path.join(TRAIN_DIR, 'y_train.csv'), y_train, delimiter=",")

np.savetxt(os.path.join(VALIDATION_DIR, 'X_val.csv'), X_val_scaled, delimiter=",")
np.savetxt(os.path.join(VALIDATION_DIR, 'y_val.csv'), y_val, delimiter=",")

np.savetxt(os.path.join(TEST_DIR, 'X_test.csv'), X_test_scaled, delimiter=",")
np.savetxt(os.path.join(TEST_DIR, 'y_test.csv'), y_test, delimiter=",")

print("\nSeturile finale (train/val/test) au fost salvate cu succes.")