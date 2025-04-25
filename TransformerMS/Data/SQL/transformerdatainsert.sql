USE text_dataset_db;

-- Inserting L1 data
LOAD DATA LOCAL INFILE '/Users/prakhar_patil/Desktop/SPE_Proj/TransformerMS/Data/L1_ChildrenStories.txt'
INTO TABLE text_data
LINES TERMINATED BY '\n'
(@line)
SET level = 'L1',
    filename = 'L1_ChildrenStories.txt',
    content = @line;



-- Inserting L2 data
LOAD DATA LOCAL INFILE '/Users/prakhar_patil/Desktop/SPE_Proj/TransformerMS/Data/L2_BookCorpus.txt'
INTO TABLE text_data
LINES TERMINATED BY '\n'
(@line)
SET level = 'L2',
    filename = 'L2_BookCorpus.txt',
    content = @line;




-- Inserting L3 data
LOAD DATA LOCAL INFILE '/Users/prakhar_patil/Desktop/SPE_Proj/TransformerMS/Data/L3_CNN_DailyMail.txt'
INTO TABLE text_data
LINES TERMINATED BY '\n'
(@line)
SET level = 'L3',
    filename = 'L3_CNN_DailyMail.txt',
    content = @line;



-- Inserting L4 data
LOAD DATA LOCAL INFILE '/Users/prakhar_patil/Desktop/SPE_Proj/TransformerMS/Data/L4_S2ORC.txt'
INTO TABLE text_data
LINES TERMINATED BY '\n'
(@line)
SET level = 'L4',
    filename = 'L4_S2ORC.txt',
    content = @line;

