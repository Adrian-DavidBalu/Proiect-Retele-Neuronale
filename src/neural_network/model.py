# Importare librarii necesare
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os 
import json # Import NOU: necesar pentru lucrul cu fisierul de configurare JSON

# Definire cai catre seturile de date preprocesate din preprocessing_script.py
def load_processed_data():
    # Definim calea bazei de date relativ la directorul radacina (care e cu 2 nivele mai jos)
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    
    # Incărcam datele preprocesate din Etapa 3
    X_train = np.loadtxt(os.path.join(data_dir, 'train', 'X_train.csv'), delimiter=",")
    y_train = np.loadtxt(os.path.join(data_dir, 'train', 'y_train.csv'), delimiter=",")
    X_val = np.loadtxt(os.path.join(data_dir, 'validation', 'X_val.csv'), delimiter=",")
    y_val = np.loadtxt(os.path.join(data_dir, 'validation', 'y_val.csv'), delimiter=",")
    X_test = np.loadtxt(os.path.join(data_dir, 'test', 'X_test.csv'), delimiter=",")
    y_test = np.loadtxt(os.path.join(data_dir, 'test', 'y_test.csv'), delimiter=",")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_model_config():
    # Definim calea catre fisierul de configurare
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'model_params.json'))
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Eroare: Fisierul de configurare nu a fost gasit la calea: {config_path}")
        exit()
    except json.JSONDecodeError:
        print("Eroare: Fisierul de configurare nu este un JSON valid.")
        exit()

def create_mlp_model(config):
    # Modelul secvential (MLP)
    model = Sequential()

    # Adaugare straturi ascunse din configurare
    for i, layer in enumerate(config['layer_structure']):
        # Stratul de Intrare (1st HL):
        if i == 0:
             # Utilizam layer['neurons'] si layer['activation'] si config['input_dim'] din JSON
             model.add(Dense(layer['neurons'], activation=layer['activation'], input_shape=(config['input_dim'],)))
        else:
            # Al Doilea Strat Ascuns (2nd HL):
            model.add(Dense(layer['neurons'], activation=layer['activation']))
    
    # Stratul de Ieșire:
    # 6 neuroni (output_dim=6), folosind Softmax pentru probabilități
    # Utilizam config['output_dim'] din JSON
    model.add(Dense(config['output_dim'], activation='softmax'))
    
    # Compilarea Modelului
    # Optimizator: Adam este standard si eficient 
    # Loss: Categorical Crossentropy este necesară pentru etichetele One-Hot
    # Metrica: Vrem să urmărim acuratețea datelor generate
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

    # Categorical Crossentropy este o matrice utilizata pentru clasificarea multiclase eficienta

def train_and_save_model():
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data()
    config = load_model_config() # Citire configurare JSON

    # Dimensiunile (I/O) - Citite din JSON
    input_dim = config['input_dim'] # 12 atribute (cele non-invazive)
    output_dim = config['output_dim'] # 6 clase (bolile posibile)
    
    # 1. Crearea Modelului
    model = create_mlp_model(config)
    print("Arhitectura Modelului MLP:")
    model.summary() # Afisare sumar arhitectura

    # 2. Antrenarea Modelului
    print("\n--- Începe Antrenarea Modelului ---")
    history = model.fit(
        X_train, y_train,
        epochs=config['training_epochs'], # Numar de epoci citit din JSON
        batch_size=config['batch_size'], # Dimensiunea batch-ului citita din JSON
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
    # Numele fisierului este preluat din config['model_name']
    model_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"{config['model_name']}.h5"))
    model.save(model_save_path)
    print(f"\nModelul a fost salvat cu succes ca {model_save_path}")

    return model_save_path

if __name__ == "__main__":
    # Blocul de rulare principal este corect, doar apelam functia
    trained_model_path = train_and_save_model()