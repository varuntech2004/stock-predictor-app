import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import bcrypt
import bcrypt
import sqlite3

load_dotenv()

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "stock_app_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                username VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        """)
        
        # Create watchlist table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                stock_symbol VARCHAR(50) NOT NULL,
                UNIQUE(user_id, stock_symbol)
            )
        """)
        
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"Database initialization failed: {e}")
        return False
    finally:
        conn.close()

# --- User Management ---

def create_user(name, username, password):
    conn = get_db_connection()
    if not conn: return False
    
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (name, username, password) VALUES (%s, %s, %s)",
            (name, username, hashed_password)
        )
        conn.commit()
        cursor.close()
        return True



        # Username already exists
        return False
    except Exception as e:
        print(f"Error creating user: {e}")
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_db_connection()
    if not conn: return None
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return user
        return None
    except Exception as e:
        print(f"Error authenticating: {e}")
        return None
    finally:
        conn.close()

def update_password(username, new_password):
    conn = get_db_connection()
    if not conn: return False
    
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password = %s WHERE username = %s",
            (hashed_password, username)
        )
        conn.commit()
        row_count = cursor.rowcount
        cursor.close()
        return row_count > 0
    except Exception as e:
        print(f"Error updating password: {e}")
        return False
    finally:
        conn.close()

# --- Watchlist Management ---

def add_to_watchlist(user_id, stock_symbol):
    conn = get_db_connection()
    if not conn: return False
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO watchlist (user_id, stock_symbol) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (user_id, stock_symbol)
        )
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"Error adding to watchlist: {e}")
        return False
    finally:
        conn.close()

def get_watchlist(user_id):
    conn = get_db_connection()
    if not conn: return []
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT stock_symbol FROM watchlist WHERE user_id = %s", (user_id,))
        watchlist = cursor.fetchall()
        cursor.close()
        return [item['stock_symbol'] for item in watchlist]
    except Exception as e:
        print(f"Error getting watchlist: {e}")
        return []
    finally:
        conn.close()

def remove_from_watchlist(user_id, stock_symbol):
    conn = get_db_connection()
    if not conn: return False
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM watchlist WHERE user_id = %s AND stock_symbol = %s",
            (user_id, stock_symbol)
        )
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"Error removing from watchlist: {e}")
        return False
    finally:
        conn.close()
