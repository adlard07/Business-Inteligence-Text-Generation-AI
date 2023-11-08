import sys
import torch
from torch.nn.functional import pad
import tensorflow as tf
from dataclasses import dataclass
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer

from src.logger import logging
from src.utils import save_model
from src.exception import CustomException
from src.components.data_tokenization import DataTokenizer
from huggingface_hub import notebook_login
notebook_login()


@dataclass
class ModelTrainer:
    def __init__(self):
        self.model_path = 'artifacts/model'
        self.tokenizer_path = 'artifacts/tokenizer'
    
    def model_initialize(self, training_dataset):
        try:
            model = GPT2LMHeadModel.from_pretrained(self.model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_path)
            
            tokenizer.pad_token = tokenizer.eos_token
            
            max_seq_length = max(len(seq['input_ids']) for seq in training_dataset)
            padded_sequences = tf.keras.utils.pad_sequences(training_dataset, padding="pre", maxlen=max_seq_length)
                        
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
                train_dataset=padded_sequences
            )
            return (
                padded_sequences,
                trainer
            )
                    
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__=='__main__':
    transform = DataTokenizer()
    train_encodings, _ = transform.initiate_data_tokenization()

    print("Model training initiating...")
    model_trainer = ModelTrainer()
    padded_sequences, trainer = model_trainer.model_initialize(train_encodings)
    # trainer.train() 
    print("Model Training Complete")
    print(padded_sequences)
    
    # save_model(model)
    # print("Model Saved!")