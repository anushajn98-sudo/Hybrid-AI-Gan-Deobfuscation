import sqlite3
import os

def init_db():
    conn = sqlite3.connect('malware_detection.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL,
                 email TEXT,
                 created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create history table to store user queries
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER NOT NULL,
                 input_code TEXT NOT NULL,
                 deobfuscated_code TEXT,
                 malware_type TEXT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                 FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect('malware_detection.db')
    conn.row_factory = sqlite3.Row
    return conn

