import sqlite3
from datetime import datetime, timedelta
import os

# Nume baza de date
DB_NAME = 'undermyaiskin_cases.db'

# Pragul de alerta (am hotarat ca 5 pacienti intr-o saptamana sa porneasca alerta)
ALERT_THRESHOLD = 5 
ALERT_PERIOD_DAYS = 7 

def initialize_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY,
            diagnosis_class INTEGER NOT NULL,
            prediction_date TEXT NOT NULL,
            symptoms_vector TEXT NOT NULL,
            quiz_responses TEXT  
        )
    ''')
    conn.commit()
    conn.close()

def record_case(diagnosis_class, symptoms_vector, quiz_responses=""):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    prediction_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Eroarea de la utilizarea 'diagnosis_date' a fost corectată aici, folosind 'diagnosis_class'
    cursor.execute('''
        INSERT INTO cases (diagnosis_class, prediction_date, symptoms_vector, quiz_responses)
        VALUES (?, ?, ?, ?)
    ''', (diagnosis_class, prediction_date, symptoms_vector, quiz_responses))
    
    conn.commit()
    conn.close()
    
def check_for_epidemic_alert():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    start_date = datetime.now() - timedelta(days=ALERT_PERIOD_DAYS)
    start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute('''
        SELECT diagnosis_class, COUNT(*) 
        FROM cases
        WHERE prediction_date >= ?
        GROUP BY diagnosis_class
        HAVING COUNT(*) >= ?
    ''', (start_date_str, ALERT_THRESHOLD))
    
    focare = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    if focare:
        print(f"!!! ALERTĂ EPIDEMICĂ: Focare detectate pentru clasele: {focare}")
        return focare
    else:
        return []

if __name__ == "__main__":
    # La rularea directă a scriptului, doar inițializăm DB pentru utilizare
    initialize_database()
    print(f"Managerul de baze de date este gata. Fișier DB: {DB_NAME}")