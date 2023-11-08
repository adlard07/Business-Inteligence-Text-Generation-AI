import sys
from dataclasses import dataclass
from transformers import GPT2Tokenizer, TextStreamer

from src.components.data_tokenization import DataTokenizer
from src.components.get_tokenizer_model import GetModels
from src.exception import CustomException
from src.logger import logging

        
class GenerateText:
    def __init__(self):
        get_model = GetModels()
        self.tokenizer, _ = get_model.get_data_tokenizer_object()
        self.model, _ = get_model.get_model_object()
            
    def generate_text(self, input_text):
        try:
            inputs = self.tokenizer([input_text], return_tensors="pt")
            streamer = TextStreamer(self.tokenizer)
            
            output = self.model.generate(**inputs, streamer=streamer, max_new_tokens=50)
            generated_text = self.tokenizer.decode(output[-1], skip_special_tokens=True)
            
            return (
                self.model, 
                self.tokenizer, 
                generated_text
                    )
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
        
if __name__=='__main__':
    
    generate = GenerateText()
    # input_text = input('Enter text: ')
    input_text = 'To increase the revenue of a company you should do the following '
    model, tokenizer, generated_text = generate.generate_text(input_text)
    print(generated_text)
        
