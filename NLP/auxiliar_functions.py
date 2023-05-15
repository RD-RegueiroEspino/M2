import numpy as np
import random
import copy
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_dataset, load_metric

def generate_noisy_data(df_train, df_val, df_test, noise_level=0.15, language='es', name='sentence'):
    """
    Create new data to check the adversarial robustness by including splits of words, insert, delete and replace letters
     or transpose two letters with a fixed probability
    :param df_train: train dataset
    :param df_val: validation dataset
    :param df_test: test dataset
    :param noise_level: probability of changing a word
    :param language: 'es' to include spanish accents, otherwise without accents
    :param name: name of the string to modify in the dataframe
    :return: modified datasets
    """
    def editions(word, language='es'):
        if language == 'es':
            letters = 'abcdefghijklmnñopqrstuvwxyzáéíóú'
        else:
            letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def noise_sentence(sentence, level=0.15, language='es'):
        L = sentence.strip().split(' ')
        L = [word if random.random() >= level else np.random.choice(list(editions(word, language))) for word in L]
        return ' '.join(L)

    def apply_to_dataset(df, noise_level, language='es'):
        df_new = copy.deepcopy(df)
        new_rows = []
        for it in range(len(df[name])):
            row = noise_sentence(df[name][it], noise_level, language)
            new_rows.append(row)
        df_new = df_new.remove_columns([name])
        df_new = df_new.add_column(name, new_rows)
        return df_new

    df_train_noisy = apply_to_dataset(df_train, noise_level, language)
    df_val_noisy = apply_to_dataset(df_val, noise_level, language)
    df_test_noisy = apply_to_dataset(df_test, noise_level, language)

    return df_train_noisy, df_val_noisy, df_test_noisy


def train_evaluate(checkpoint, tokenizer, metrics_function, dataset, training_args, model_name, num_labels=None,
                   eval_subset="test"):
    if num_labels == None:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[eval_subset],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metrics_function,
    )

    trainer.train()

    trainer.save_model("./best_model")

    print(trainer.evaluate())


