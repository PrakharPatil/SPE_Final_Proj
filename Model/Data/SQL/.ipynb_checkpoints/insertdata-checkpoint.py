import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

conn = pymysql.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    charset='utf8mb4'  # important for Unicode and large content
)

cursor = conn.cursor()

def insert_file(level, filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    query = "INSERT INTO text_data (level, filename, content) VALUES (%s, %s, %s)"
    cursor.execute(query, (level, filepath, content))
    conn.commit()

insert_file("L1", "../L1_ChildrenStories.txt")
insert_file("L2", "../L2_BookCorpus.txt")
insert_file("L3", "../L3_CNN_DailyMail.txt")
insert_file("L4", "../L4_S2ORC.txt")

cursor.close()
conn.close()
