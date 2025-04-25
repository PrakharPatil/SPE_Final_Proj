import pymysql
import os
import time
from dotenv import load_dotenv

load_dotenv()


# Function to get MySQL connection
def get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        charset='utf8mb4',  # important for Unicode and large content
        connect_timeout=10,  # Timeout for connection attempts
        client_flag=pymysql.constants.CLIENT.MULTI_STATEMENTS  # Allow multiple queries
    )


# Function to insert file into the database
def insert_file(level, filepath):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        query = "INSERT INTO text_data (level, filename, content) VALUES (%s, %s, %s)"

        # Attempt to execute the query with retries
        retries = 3
        for attempt in range(retries):
            try:
                cursor.execute(query, (level, filepath, content))
                conn.commit()
                print(f"Successfully inserted {filepath} at level {level}")
                break
            except pymysql.MySQLError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2)  # Wait before retrying
                else:
                    raise e
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error inserting {filepath}: {e}")


# Insert files
insert_file("L1", "../L1_ChildrenStories.txt")
insert_file("L2", "../L2_BookCorpus.txt")
insert_file("L3", "../L3_CNN_DailyMail.txt")
insert_file("L4", "../L4_S2ORC.txt")
