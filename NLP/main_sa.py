from auxiliar_functions import generate_noisy_data
import os
import numpy as np

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_dataset, load_metric

if __name__ == '__main__':
    name='canine'
    language='english'
    noise_level=0.15

    persistent_storage = 'code/working/'
    os.makedirs(persistent_storage, exist_ok=True)

    if name=='bert':
        checkpoint = "bert-base-cased"
        model_name = "bert-cased-tweet-noise"
    else:
        checkpoint = "google/canine-c"
        model_name = "canine-c-tweet-noise"

    def tokenize_function(example):
        return tokenizer(example["text"],
                         truncation=True,
                         padding=True)

    output_path = os.path.join(persistent_storage, model_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = load_dataset("tyqiangz/multilingual-sentiments",language).map(tokenize_function, batched=True)

    if noise_level>0:
        if language=='english':
            df_train_noisy, df_val_noisy, df_test_noisy = generate_noisy_data(dataset["train"], dataset["validation"],
                                                                      dataset["test"], noise_level=noise_level, language='eng',
                                                                      name='text')
        else:
            df_train_noisy, df_val_noisy, df_test_noisy = generate_noisy_data(dataset["train"], dataset["validation"],
                                                                          dataset["test"], noise_level=noise_level,
                                                                          language='eng', name='text')




    def compute_metrics(eval_preds):
        f1 = load_metric("f1")
        accuracy = load_metric("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return {**f1.compute(predictions=predictions, references=labels, average="macro"),
                **accuracy.compute(predictions=predictions, references=labels)}


    def train_evaluate(checkpoint, tokenizer, metrics_function, dataset_train, training_args, model_name, dataset_val,
                       num_labels=None):
        if num_labels == None:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model,
            training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_val,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=metrics_function,
        )

        trainer.train()

        trainer.save_model("./best_model")

        print(trainer.evaluate())


    training_args = TrainingArguments(output_path,
                                      num_train_epochs=20,
                                      learning_rate=1e-5,
                                      per_device_train_batch_size=16,
                                      evaluation_strategy="epoch",
                                      logging_steps=1,
                                      )

    train_evaluate(checkpoint, tokenizer, compute_metrics, df_train_noisy, training_args, model_name, df_test_noisy,
                   num_labels=3)

