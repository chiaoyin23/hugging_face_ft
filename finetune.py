import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import evaluate
import os
os.environ["OMP_NUM_THREADS"] = "1"

data_path = 'train.csv'
data = pd.read_csv(data_path)

conversations = []
for i in range(len(data)):
    conversations.append({"input_text": data.iloc[i]['instruction'], "response_text": data.iloc[i]['output']})

dataset = Dataset.from_pandas(pd.DataFrame(conversations))

train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

model_name = "taide/TAIDE-LX-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


local_dir = "./saved_taide_base_model"
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

# print(f"Model and tokenizer saved to {local_dir}")
# tokenizer = AutoTokenizer.from_pretrained(local_dir)
# model = AutoModelForCausalLM.from_pretrained(local_dir)


def tokenize_function(examples):
    combined_texts = [
        (input_text if input_text is not None else "") + " " + (response_text if response_text is not None else "")
        for input_text, response_text in zip(examples['input_text'], examples['response_text'])
    ]
    return tokenizer(combined_texts, padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  
)


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    eval_strategy="epoch", 
)
