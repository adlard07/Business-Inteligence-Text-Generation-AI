import os
import sys
import tensorflow as tf
from dataclasses import dataclass
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, GPT2Tokenizer, GPT2Model, TextStreamer, GPT2LMHeadModel

from src.logger import logging
from src.utils import save_model
from src.exception import CustomException


class GetModels:
    def __init__(self):
        self.context_path = 'data/content/context.txt'
        self.tokenizer_path = 'artifacts/tokenizer'
        self.model_path = 'artifacts/model'
        
    
    def get_data_tokenizer_object(self):
        try:
            if os.path.exists(self.tokenizer_path):
                tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_path)
                logging.info('Model available!')
                print('Tokenizer available!') 
            else:
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                save_model(tokenizer, self.tokenizer_path)
                logging.info('Downloaded Tokenizer!')
                print('Models Saved!')
            return (
                tokenizer, 
                self.tokenizer_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)
                
                
    def get_model_object(self):
        try:
            if os.path.exists(self.model_path):
                model = GPT2LMHeadModel.from_pretrained(self.model_path)
                logging.info('Model available!')
                print('Model available!')
            else:
                model = GPT2LMHeadModel.from_pretrained('gpt2')
                save_model(model, self.model_path)
                logging.info('Downloaded Model!')
                print('Models Saved!')
            return (
                model, 
                self.model_path
                )
             
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=='__main__':
    get_models = GetModels()
    print('Processing...')
    tokenizer, tokenizer_path = get_models.get_data_tokenizer_object()
    model, model_path = get_models.get_model_object()
    print('Process Complete!')