import sys
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QGridLayout, QLabel, QCheckBox, 
    QPushButton, QGroupBox, QTextEdit, QMessageBox,
    QLineEdit, QRadioButton
)
from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import Qt
import os

# Importam logica de baza de date si alertă
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.database_manager import initialize_database, record_case, check_for_epidemic_alert
except ImportError:
    print("EROARE: Modulul src.database_manager nu a fost gasit.")
    print("Va rugam sa va asigurati ca ati creat fisierul src/database_manager.py")
    sys.exit(1)


# --- Configuratii ---
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'neural_network', 'dermotriage_mlp_model.h5'))
# Vom folosi numele bolilor de la clasa 1 la 6
DIAGNOSES = {
    1: "Psoriazis", 
    2: "Dermatită seboreică", 
    3: "Lichen plan", 
    4: "Pityriasis rosea", 
    5: "Dermatită cronică", 
    6: "Pityriasis rubra pilaris"
}

# --- Recomandări Clinice (Text) ---
RECOMMENDATIONS = {
    1: "Necesită neapărat vizita la dermatolog. Această afecțiune necesită o schemă de tratament complexă.",
    2: "Este nevoie de medicamente topice. Consultați medicul de familie pentru un tratament inițial.",
    3: "Necesită neapărat vizita la dermatolog pentru confirmare și tratament sistemic.",
    4: "Trece de la sine în 6-8 săptămâni. Monitorizați simptomele; consultați un medic doar dacă se agravează.",
    5: "Este nevoie de medicamente. Evitați factorii iritanți și consultați un medic pentru prescripții de unguente.",
    6: "Necesită neapărat vizita la dermatolog. Terapia este adesea complexă și de lungă durată."
}

# --- Scaler Parameters ---
# Aceste valori sunt necesare pentru a scala 'Age' (A34) la fel ca în pre-procesare
AGE_MIN = 1.0
AGE_MAX = 90.0
AGE_RANGE = AGE_MAX - AGE_MIN


class UnderMyAISkinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UnderMyAISkin: Sistem AI de Pre-Diagnosticare Dermatologică")
        self.setGeometry(100, 100, 1000, 750)

        # 1. Inițializare BD și Model
        self.init_dependencies()

        # 2. Setare UI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # 3. Construire Layout-uri
        self.create_input_panel()
        self.create_output_panel()

        self.main_layout.addWidget(self.input_group, 4) 
        self.main_layout.addWidget(self.output_group, 3) 
    
    def init_dependencies(self):
        # Încărcare Model (și suprimarea mesajelor de la TensorFlow)
        tf.get_logger().setLevel('ERROR')
        print("Încărc Modelul AI...")
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("Model AI încărcat cu succes.")
        except Exception as e:
            QMessageBox.critical(self, "Eroare AI", f"Nu am putut încărca modelul MLP: {e}. Asigurați-vă că model.py a rulat și ca fisierul .h5 exista la calea: {MODEL_PATH}")
            self.model = None

        # Inițializare Bază de Date
        initialize_database()
        print("Bază de date SQLite inițializată.")

        # Atributele (A1-A10, A11=Family History, A34=Age)
        self.attribute_map = [
            ("A1: Eritem (Roșeață)"), ("A2: Scuame (Coajă)"),
            ("A3: Margini definite"), ("A4: Prurit (Mâncărime)"),
            ("A5: Fenomen Koebner"), ("A6: Papule poligonale"),
            ("A7: Papule foliculare"), ("A8: Afectare mucoasă orală"),
            ("A9: Afectare genunchi/cot"), ("A10: Afectare scalp"),
            ("A11: Istoric familial"), ("A34: Vârsta pacientului") 
        ]
        self.input_widgets = {}
        self.current_age = 35 
        
    def create_input_panel(self):
        """Creează panoul de input cu checkbox-uri și slider-e."""
        self.input_group = QGroupBox("1. Introducere Simptome Clinice (12 Atribute)")
        layout = QGridLayout()
        
        # Titlurile coloanelor
        layout.addWidget(QLabel("Simptom"), 0, 0, 1, 2)
        layout.addWidget(QLabel("Nivel"), 0, 2)
        
        row = 1
        for i, label_text in enumerate(self.attribute_map):
            attribute_id = label_text.split(':')[0]
            
            # --- Vârsta (A34) ---
            if attribute_id == 'A34':
                self.age_label = QLabel(f"A34: Vârsta: {self.current_age} ani")
                self.age_input = QLineEdit(str(self.current_age))
                self.age_input.setValidator(QIntValidator(1, 100))
                self.age_input.textChanged.connect(self.update_age)
                self.input_widgets[attribute_id] = self.age_input
                
                layout.addWidget(self.age_label, row, 0)
                layout.addWidget(self.age_input, row, 1)

            # --- Istoric Familial (A11) ---
            elif attribute_id == 'A11':
                group = QGroupBox(label_text)
                hbox = QHBoxLayout(group)
                self.input_widgets[attribute_id] = []
                
                radio_no = QRadioButton("0 (NU)")
                radio_yes = QRadioButton("1 (DA)")
                radio_no.setChecked(True)
                
                hbox.addWidget(radio_no)
                hbox.addWidget(radio_yes)
                self.input_widgets[attribute_id].extend([radio_no, radio_yes])
                
                layout.addWidget(group, row, 0, 1, 3)

            # --- Simptome Discrete (A1-A10, Nivel 0-3) ---
            else:
                group = QGroupBox(label_text)
                hbox = QHBoxLayout(group)
                self.input_widgets[attribute_id] = []
                
                # Creăm 4 RadioButtons pentru 0, 1, 2, 3
                for val in range(4):
                    radio_button = QRadioButton(str(val))
                    if val == 0:
                        radio_button.setChecked(True) 
                    hbox.addWidget(radio_button)
                    self.input_widgets[attribute_id].append(radio_button)
                    
                layout.addWidget(group, row, 0, 1, 3)

            row += 1

        # Buton de Clasificare
        self.predict_button = QPushButton("Clasificare cu UnderMyAISkin")
        self.predict_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 15px; font-size: 16px; font-weight: bold; border-radius: 8px;")
        self.predict_button.clicked.connect(self.run_prediction)
        layout.addWidget(self.predict_button, row, 0, 1, 3)
        
        self.input_group.setLayout(layout)

    def create_output_panel(self):
        """Creează panoul de rezultate și alertă."""
        self.output_group = QGroupBox("2. Rezultate AI & Recomandări")
        layout = QVBoxLayout()
        self.output_group.setStyleSheet("QGroupBox {border: 2px solid gray; border-radius: 5px; margin-top: 1ex;} QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px;}")


        # Display Alert Epidemie
        self.alert_label = QLabel("STATUS EPIDEMIC: Nu sunt focare active detectate.")
        self.alert_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_label.setStyleSheet("font-weight: bold; color: green; padding: 10px; border: 2px solid green; border-radius: 5px; background-color: #e6ffe6;")
        layout.addWidget(self.alert_label)

        # Output Top 3
        layout.addWidget(QLabel("--- Toate Cele 6 Clasificări Probabile (Sortate) ---")) # TITLU MODIFICAT
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setFontPointSize(10)
        self.result_display.setText("Așteaptă inputul...")
        layout.addWidget(self.result_display)
        
        # Recomandare
        layout.addWidget(QLabel("\n--- Recomandare de Acțiune Clinică ---"))
        self.recommendation_display = QTextEdit()
        self.recommendation_display.setReadOnly(True)
        self.recommendation_display.setFontPointSize(12)
        self.recommendation_display.setStyleSheet("background-color: #f0f0f0;")
        layout.addWidget(self.recommendation_display)

        self.output_group.setLayout(layout)

    def update_age(self, text):
        """Actualizează vârsta în interfață."""
        try:
            age = int(text)
            if 1 <= age <= 100:
                self.current_age = age
                self.age_label.setText(f"A34: Vârsta: {self.current_age} ani")
        except ValueError:
            pass
        
    def collect_input_vector(self):
        """Colectează cele 12 atribute clinice de la utilizator."""
        input_vector = []
        symptoms_str = []
        
        for label_text in self.attribute_map:
            attribute_id = label_text.split(':')[0]
            
            if attribute_id == 'A34':
                value = self.current_age
                
            elif attribute_id == 'A11': 
                value = 1 if self.input_widgets[attribute_id][1].isChecked() else 0
            
            else: 
                value = 0
                for idx, rb in enumerate(self.input_widgets[attribute_id]):
                    if rb.isChecked():
                        value = idx
                        break
            
            input_vector.append(value)
            symptoms_str.append(str(value))
        
        # --- SCALARE INPUT (CRITIC: Trebuie să folosească aceeași logică ca în pre-procesare) ---
        input_array = np.array(input_vector).astype(np.float32)
        
        # A1-A10 (index 0-9): Scalare 0-3 -> 0-1 (împărțire la 3)
        input_array[0:10] = input_array[0:10] / 3.0
        
        # A11 (index 10): Istoric Familial este deja 0 sau 1
        
        # A34 (index 11): Vârsta, Scalare Min-Max
        if input_array[11] > 0:
             input_array[11] = (input_array[11] - AGE_MIN) / AGE_RANGE
        
        
        return input_array.reshape(1, -1), ",".join(symptoms_str)

    def run_prediction(self):
        """Execută clasificarea, înregistrează cazul și verifică alerte."""
        if not self.model:
            QMessageBox.warning(self, "Avertisment", "Modelul AI nu este încărcat.")
            return

        # 1. Colectare și Pregătire Input
        scaled_input, raw_symptoms_str = self.collect_input_vector()

        # 2. Executare Predicție (Inferență)
        probabilities = self.model.predict(scaled_input, verbose=0)[0] 
        
        # 3. Procesare Rezultate (TOATE CELE 6 CLASE)
        # Obține indicii (clasele 0-5) sortate după probabilitate descrescător
        # Luăm toate cele 6 indici ([:6])
        top_indices = np.argsort(probabilities)[::-1][:6] 
        
        # Obține cele mai probabile clase (1-6)
        top_classes = top_indices + 1 
        
        # Obține cel mai probabil diagnostic (primul din lista sortată)
        max_class_index = top_indices[0]
        max_class = top_classes[0]
        max_prob = probabilities[max_class_index]
        
        # 4. Înregistrare Caz în Baza de Date
        record_case(int(max_class), raw_symptoms_str)
        print(f"Caz înregistrat: Clasa {max_class} ({DIAGNOSES[max_class]}), Prob: {max_prob:.2f}")

        # 5. Verificare Alerte Epidemice (Logica Sănătății Publice)
        focare_active = check_for_epidemic_alert()
        
        is_epidemic = False
        if focare_active:
            self.alert_label.setStyleSheet("font-weight: bold; color: white; padding: 10px; border: 2px solid red; border-radius: 5px; background-color: #ff3333;")
            self.alert_label.setText(f"!!! ALERTĂ EPIDEMICĂ CRITICĂ: FOCAR ACTIV - VIZITĂ LA CLINICĂ OBLIGATORIE !!!\nClase afectate: {focare_active}")
            is_epidemic = True
        else:
            self.alert_label.setStyleSheet("font-weight: bold; color: green; padding: 10px; border: 2px solid green; border-radius: 5px; background-color: #e6ffe6;")
            self.alert_label.setText("STATUS EPIDEMIC: Nu sunt focare active detectate. Urmați recomandarea.")

        # 6. Generare Output (TOATE CELE 6 CLASIFICĂRI)
        output_text = "Probabilitate | Diagnostic\n"
        output_text += "---------------------------------\n"
        for rank, index in enumerate(top_indices):
            class_code = index + 1
            prob = probabilities[index] * 100
            # Afisam toate cele 6, dar evidentiem primele 3
            prefix = "⭐ " if rank < 1 else "   " 
            output_text += f"{prefix} {prob:.2f}% | {DIAGNOSES[class_code]}\n"
        
        self.result_display.setText(output_text)
        
        # 7. Generare Recomandare Finală
        if is_epidemic:
            final_recommendation = (
                f"*** ALERTA SUPRASCRIE DIAGNOSTICUL INDIVIDUAL ***\n"
                f"Sistemul detectează o creștere anormală a cazurilor în comunitate (focar). "
                f"REGULĂ OBLIGATORIE: VĂ PREZENTAȚI IMEDIAT LA CLINICĂ sau la medicul dermatolog pentru evaluare și măsuri de control."
            )
        else:
            final_recommendation = (
                f"Diagnostic Probabil (Top 1): {DIAGNOSES[max_class]} ({max_prob*100:.2f}%)\n\n"
                f"ACȚIUNE RECOMANDATĂ:\n"
                f"{RECOMMENDATIONS.get(max_class, 'Consultați un medic specialist.')}"
            )

        self.recommendation_display.setText(final_recommendation)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    try:
        QIntValidator(1, 100)
        QLineEdit()
    except NameError:
        print("\nEROARE: Componentele PyQt necesare (QIntValidator, QLineEdit) nu sunt importate corect.")
        print("Va rugam sa va asigurati ca ati instalat PyQt6 si ca codul este complet.")
        sys.exit(1)

    window = UnderMyAISkinApp()
    window.show()
    sys.exit(app.exec())