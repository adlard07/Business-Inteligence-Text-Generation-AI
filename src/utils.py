import os
import sys
import PyPDF2
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


# reads pdfs
def read_pdf(pdf_file_path, start_page, end_page):
    pdf_reader = PyPDF2.PdfReader(pdf_file_path)
    extracted_text = ""
    for page_num in range(start_page, end_page):
        page = pdf_reader.pages[page_num]
        extracted_text += page.extract_text()
    pdf_reader.stream.close()
    return extracted_text


# saves pdf text into a .txt file     
def save_text(file_path, text):
    if os.path.exists(file_path):
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(text)
    else:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
        
        
# saves model and tokenizer
def save_model(model, model_path):
    try:
        if os.path.exists(model_path):
            pass
        else:    
            model.save_pretrained(model_path)
            logging.info('Model Saved!')
            print('Saved!')
            
    except Exception as e:
        raise CustomException(e, sys)