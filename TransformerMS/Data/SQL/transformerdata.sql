USE text_dataset_db;
CREATE TABLE IF NOT EXISTS text_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    level VARCHAR(10),
    filename VARCHAR(255),
    content LONGTEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
