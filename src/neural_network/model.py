# Importare librarii necesare
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os # Import NOU: necesar pentru lucrul cu cai absolute

# Definire cai catre seturile de date preprocesate din preprocessing_script.py
def load_processed_data():
    # Definim calea bazei de date relativ la directorul radacina (care e cu 2 nivele mai sus)
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    
    # Incărcam datele preprocesate din Etapa 3
    X_train = np.loadtxt(os.path.join(data_dir, 'train', 'X_train.csv'), delimiter=",")
    y_train = np.loadtxt(os.path.join(data_dir, 'train', 'y_train.csv'), delimiter=",")
    X_val = np.loadtxt(os.path.join(data_dir, 'validation', 'X_val.csv'), delimiter=",")
    y_val = np.loadtxt(os.path.join(data_dir, 'validation', 'y_val.csv'), delimiter=",")
    X_test = np.loadtxt(os.path.join(data_dir, 'test', 'X_test.csv'), delimiter=",")
    y_test = np.loadtxt(os.path.join(data_dir, 'test', 'y_test.csv'), delimiter=",")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_mlp_model(input_dim, output_dim):
    # Modelul secvential (MLP)
    model = Sequential([
        # Stratul de Intrare (1st HL):
        # 12 neuroni de intrare (input_dim=12)
        Dense(64, activation='relu', input_shape=(input_dim,)),
        
        # Al Doilea Strat Ascuns (2nd HL):
        # Asigura complexitatea sporita
        Dense(32, activation='relu'),
        
        # Stratul de Ieșire:
        # 6 neuroni (output_dim=6), folosind Softmax pentru probabilități
        Dense(output_dim, activation='softmax')
    ])
    
    # ReLU vine de la Rectified Linear Unit care este o instructiune populara de activare

    # Compilarea Modelului
    # Optimizator: Adam este standard si eficient (conform discutiilor gasite pe StackOverflow, deci am mers cu el mai departe)
    # Loss: Categorical Crossentropy este necesară pentru etichetele One-Hot
    # Metrica: Vrem să urmărim acuratețea datelor generate
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

    # Categorical Crossentropy este o matrice utilizata pentru clasificarea multiclase eficienta

def train_and_save_model():
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data()

    # Dimensiunile (I/O)
    input_dim = X_train.shape[1] # 12 atribute (cele non-invazive)
    output_dim = y_train.shape[1] # 6 clase (bolile posibile)
    
    # 1. Crearea Modelului
    model = create_mlp_model(input_dim, output_dim)
    print("Arhitectura Modelului MLP:")
    model.summary() # Afisare sumar arhitectura

    # 2. Antrenarea Modelului
    print("\n--- Începe Antrenarea Modelului ---")
    history = model.fit(
        X_train, y_train,
        epochs=100, # Numar de epoci (am ales 100 deoarece am vazut ca este un numar standard)
        batch_size=32, # Dimensiunea batch-ului
        validation_data=(X_val, y_val),
        verbose=1
    )

    # 3. Evaluare Finală pe Setul de Test
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- Evaluare Finală pe Setul de Test (10% date) ---")
    print(f"Loss (Pierdere): {loss:.4f}")
    print(f"Accuracy (Acuratețe): {accuracy*100:.2f}% (Asteptat: pana la 90% pentru modelul Triage)")

    # 4. Salvarea Modelului Antrenat
    # Calea absolută de salvare, asigurând că este in src/neural_network
    model_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dermotriage_mlp_model.h5'))
    model.save(model_save_path)
    print(f"\nModelul a fost salvat cu succes ca {model_save_path}")

    return model_save_path

if __name__ == "__main__":
    trained_model_path = train_and_save_model()