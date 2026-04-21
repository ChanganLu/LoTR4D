import transformers
transformers.logging.set_verbosity_error()

from transformers import DataCollatorWithPadding, RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader

from typing import Tuple
import os


GLUE_TASK_TO_KEYS = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
}

CACHE_DIR = './local_cache'

def load_and_preprocess_data(batch_size, task_name) -> Tuple[DataLoader, DataLoader, int]:
    print(f'Loading task: {task_name}')
    actual_task = 'mnli' if task_name == 'mnli-mm' else task_name

    cache_dir_task = os.path.join(CACHE_DIR, actual_task)
    if os.path.isdir(cache_dir_task):
        raw_data = load_from_disk(cache_dir_task)
    else:
        raw_data = load_dataset('glue', actual_task)
        raw_data.save_to_disk(cache_dir_task)

    cache_dir_tokenizer = os.path.join(CACHE_DIR, 'roberta-base-tokenizer')
    if os.path.isdir(cache_dir_tokenizer):
        tokenizer = RobertaTokenizer.from_pretrained(cache_dir_tokenizer)
    else:
        tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer.save_pretrained(cache_dir_tokenizer)
        
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]

    def tokenize_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None 
            else (examples[sentence1_key], examples[sentence2_key])
        )
        return tokenizer(*args, truncation=True, return_overflowing_tokens=False, max_length=128, padding=False)
    
    tokenized_data = raw_data.map(tokenize_function, batched=True)

    raw_columns = raw_data['train'].column_names
    columns_to_remove = [c for c in raw_columns if c != 'label']
    
    tokenized_data = tokenized_data.remove_columns(columns_to_remove)
    tokenized_data = tokenized_data.rename_column('label', 'labels')
    tokenized_data.set_format('torch')

    validation_key = 'validation_mismatched' if task_name == 'mnli-mm' else \
                     'validation_matched' if task_name == 'mnli' else \
                     'validation'
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(tokenized_data['train'], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_data[validation_key], batch_size=batch_size, collate_fn=data_collator)
    num_labels = raw_data['train'].features['label'].num_classes

    return train_dataloader, eval_dataloader, num_labels

def load_roberta(num_labels: int) -> RobertaForSequenceClassification:
    cache_dir_model = os.path.join(CACHE_DIR, f'roberta-base-model({num_labels})')
    if os.path.isdir(cache_dir_model):
        model = RobertaForSequenceClassification.from_pretrained(cache_dir_model, num_labels=num_labels)
    else:
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
        model.save_pretrained(cache_dir_model)
    model.requires_grad_(False)
    return model
