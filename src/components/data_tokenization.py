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
        self.data_path = 'data/content/context.csv'
        self.tokenizer = 'artifacts/tokenizer'

            
    def initiate_data_tokenization(self):
        try:
            data = DataIngestion()
            text_data = data.get_data()
            
            get_tokenizer = GetModels()
            tokenizer, _ = get_tokenizer.get_data_tokenizer_object()
            
            train_data = []
            for text in text_data:
                token = tokenizer(text)
                train_data.append(token)
                
            logging.info('Data Encoded!')
            print('Data Encoded!')
            
            return (
                train_data,
                tokenizer
                )
            
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=='__main__':
    transform = DataTokenizer()
    train_encodings, tokenizer = transform.initiate_data_tokenization()
    print(f'Encodings : {train_encodings[:5]}')
    print(f'Decodings : {tokenizer.decode(train_encodings[1]["input_ids"])}')
    print("Data Tokenization Complete!")
    print('\n')