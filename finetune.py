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


batch_size = 64
model_name = "poem_model"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=40,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_dir="./logs",
                                )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()


model.save_pretrained("./finetuned-model")
tokenizer.save_pretrained("./finetuned-model")
