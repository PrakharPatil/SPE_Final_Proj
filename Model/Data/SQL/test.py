import pymysql

# Database connection details
db_host = '127.0.0.1'  # or 'localhost'
db_user = 'root'
db_password = 'MySQL31#'  # Update with your actual password
db_name = 'text_dataset_db'

# Establish the connection
try:
    connection = pymysql.connect(host=db_host,
                                 user=db_user,
                                 password=db_password,
                                 database=db_name)
    print("Connection successful!")

    # Close the connection after testing
    connection.close()

except pymysql.MySQLError as e:
    print(f"Error connecting to MySQL: {e}")
