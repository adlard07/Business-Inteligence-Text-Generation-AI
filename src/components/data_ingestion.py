import os
import sys
from dataclasses import dataclass

from src.utils import read_pdf, save_text
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    def data_path(self, dir_path):
        try:
            data_paths = []
            for book in os.listdir(dir_path):
                data = os.path.join(dir_path, book)
                data_paths.append(data) 
                logging.info('Content Path Extracted!')       
            return data_paths
        
        except Exception as e:
                raise CustomException(e, sys)
        
 
class DataIngestion: 
    def __init__(self):
        self.pdf_file_path = "data\\books"
        self.data_ingestion = DataIngestionConfig()
        self.paths = self.data_ingestion.data_path(self.pdf_file_path)
        logging.info('Content Path Extracted!')
    
    def get_data(self):
        try:
            start_pages = [32, 2, 11, 7]
            end_pages = [369, 16, 208, 218]
            for i in range(len(start_pages)):
                text = read_pdf(self.paths[i], start_pages[i], end_pages[i])
            logging.info('Data Read!') 
            train_data = text.split('\n')
            return train_data
                
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__=='__main__':
    data = DataIngestion()
    train_data = data.get_data()
    save_text('data\\content\\context.txt', train_data)
    logging.info("Content Saved as 'context.txt'")
    print('Content Saved as "context.txt"') 