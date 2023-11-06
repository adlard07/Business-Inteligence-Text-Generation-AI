import sys
from dataclasses import dataclass
from transformers import  AutoTokenizer, AutoModelForCausalLM, TextStreamer

from src.logger import logging
from src.utils import save_model
from src.exception import CustomException


class GenerateText:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("artifacts/tokenizer")
        self.model = AutoModelForCausalLM.from_pretrained("artifacts/model")
        
    def generate_text(self, input_text):
        try:
            inputs = self.tokenizer([input_text], return_tensors="pt")
            streamer = TextStreamer(self.tokenizer)
            
            output = self.model.generate(**inputs, streamer=streamer, max_new_tokens=25)
            generated_text = self.tokenizer.decode(output[-1], skip_special_tokens=True)
            
            return (
                self.model, 
                self.tokenizer, 
                generated_text
                    )
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
        
if __name__=='__main__':
    input_text = input('Enter text: ')
    
    generate = GenerateText()
    model, tokenizer, generated_text = generate.generate_text(input_text)
    print(generated_text)