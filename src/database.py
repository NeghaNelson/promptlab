import sqlite3
import pandas as pd

DB_NAME = "promptlab.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model TEXT,
            tone TEXT,
            original_prompt TEXT,
            optimized_prompt TEXT,
            original_response TEXT,
            optimized_response TEXT,
            avg_original_score REAL,
            avg_optimized_score REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_experiment(timestamp, model, tone, original_prompt, optimized_prompt, orig_resp, opt_resp, orig_score, opt_score):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO experiments (timestamp, model, tone, original_prompt, optimized_prompt, original_response, optimized_response, avg_original_score, avg_optimized_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, model, tone, original_prompt, optimized_prompt, orig_resp, opt_resp, orig_score, opt_score))
    conn.commit()
    conn.close()

def get_all_experiments():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM experiments", conn)
    conn.close()
    return df