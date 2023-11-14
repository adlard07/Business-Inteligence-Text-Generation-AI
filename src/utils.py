import os
import pandas as pd
import sys
from csv import writer
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


# saves pdf text into a .csv file     
def save_text(file_path, text_data):
    if os.path.exists(file_path):
        with open(file_path, 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(text_data)
            f_object.close()
    else:
        text_data.to_csv(file_path, header=False, index=False)
        
        
# saves model and tokenizer
def save_model(model, model_path):
    try:
        if os.path.exists(model_path):
            pass
        else:    
            model.save_pretrained(model_path)
            logging.info('Model Saved!')
            print('Model Saved!')
            
    except Exception as e:
        raise CustomException(e, sys)
