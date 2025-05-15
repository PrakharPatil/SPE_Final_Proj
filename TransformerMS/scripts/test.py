import os
# BASE_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

raw_dir = os.path.join(BASE_DIR, "Data", "Raw")
print(raw_dir)