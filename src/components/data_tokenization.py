import sys
from dataclasses import dataclass
from transformers import GPT2Tokenizer

from src.components.data_ingestion import DataIngestion
from src.components.get_tokenizer_model import GetModels

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTokenizer:
    def __init__(self):
        self.data_path = 'data/content/context.txt'
        self.tokenizer = 'artifacts/tokenizer'

            
    def initiate_data_tokenization(self):
        try:
            data = DataIngestion()
            train_data = data.get_data()
            get_tokenizer = GetModels()
            tokenizer, tokenizer_path = get_tokenizer.get_data_tokenizer_object()
            train_encodings = tokenizer(train_data, truncation=True, return_tensors='tf')
            logging.info('Data Encoded!')
            print('Data Encoded!')
            return (
                train_encodings
                )
            
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=='__main__':
    transform = DataTokenizer()
    train_encodings = transform.initiate_data_tokenization()
    print(type(train_encodings))
    print("Data Tokenization Complete!")