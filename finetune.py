import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import evaluate
import os
import json
import traceback

os.environ["OMP_NUM_THREADS"] = "1"
try:
    print("Loading data...")
    data_path = 'D:/hugging_face_ft/chat_data_copy.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Processing data...")
    conversations = []
    for item in data:
        instruction = item['instruction'] if item['instruction'] is not None else ""
        input_text = item['input'] if item['input'] is not None else ""
        output_text = item['output'] if item['output'] is not None else ""
        
        combined_text = instruction + input_text  # 保留原始資料中的 \r 符號
        conversations.append({"input_text": combined_text, "response_text": output_text})

    print("Converting data to dataset...")
    dataset = Dataset.from_pandas(pd.DataFrame(conversations))

    print("Splitting dataset...")
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    local_dir = "./saved_taide_base_model"
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForCausalLM.from_pretrained(local_dir, device_map='cpu')

    print("Tokenizing datasets...")
    def tokenize_function(examples):
        combined_texts = [
            input_text + " " + response_text
            for input_text, response_text in zip(examples['input_text'], examples['response_text'])
        ]
        return tokenizer(combined_texts, padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    print("Preparing data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  
    )

    print("Loading metric...")
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    print("Setting up training arguments...")
    batch_size = 2
    model_name = "finetuned-model"
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=3,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    eval_strategy = 'no', 
                                    save_strategy = 'epoch',
                                    use_cpu=True
                                    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained("./finetuned-model")
    tokenizer.save_pretrained("./finetuned-model")

except Exception as e:
    print("An error occurred:")
    print(str(e))
    traceback.print_exc()
