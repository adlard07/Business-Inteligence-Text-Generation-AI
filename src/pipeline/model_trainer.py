import sys
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from src.logger import logging
from src.utils import save_model
from src.exception import CustomException
from src.components.data_tokenization import GetDataTokenization

access_token = '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'

@dataclass
class ModelTrainer:
    def group_texts(self, examples):
        try:        
            block_size = 32
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            
            grouped_text = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            
            grouped_text["labels"] = grouped_text["input_ids"].copy()
        
            return grouped_text
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
    
    def model_training(self,  model, tokenizer, training_dataset):
        try:
            training_args = TrainingArguments(
                output_dir="artifacts/output",
                evaluation_strategy = "epoch",
                learning_rate=2e-5,
                weight_decay=0.01,
                push_to_hub=True,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=training_dataset,
                tokenizer=tokenizer
            )
            return (
                trainer
            )
                    
        except Exception as e:
            raise CustomException(e, sys)


if __name__=='__main__':
    
    tokenizer = AutoTokenizer.from_pretrained("artifacts/tokenizer")
    model = AutoModelForCausalLM.from_pretrained("artifacts/model")
    print("Models Read!")
    
    initiate_tokenizer = GetDataTokenization()
    tokenized_datasets = initiate_tokenizer.initiate_data_tokenization(tokenizer, model)
    print("Data Tokenization Complete!")
    
    trainer = ModelTrainer()
    training_dataset = trainer.group_texts(tokenized_datasets)
    
    new_model = trainer.model_training(model, tokenizer, training_dataset)
            
    print("Model training initiating...")
    
    new_model.train() 
    
    print("Model Training Complete")
    
    save_model(new_model)
    
    print("Model Saved!")