import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Step 1: Load and Prepare CSV Data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['text'] = df.apply(lambda row: f"User: {row['Prompt']}\nAssistant: {row['Response']}\nTone: {row['Tone']}", axis=1)
    dataset = Dataset.from_pandas(df[['text']])

    return dataset

# Step 2: Tokenize the Data with Labels
def tokenize_data(dataset, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        encoding = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
        encoding["labels"] = encoding["input_ids"].copy()  # Use input_ids as labels for causal LM
        return encoding

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Main Training Function
def train_model(file_path, model_path, output_dir="./results", num_train_epochs=1):
    dataset = load_and_prepare_data(file_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    # Split into training and evaluation datasets
    train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.1).values()

    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # Updated from evaluation_strategy to eval_strategy
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Run the training
if __name__ == "__main__":
    csv_file_path = r"C:\Users\alanl\LapplandBot\output_with_tone.csv"
    llama_model_path = r"C:\Users\alanl\LapplandBot\Llama-3.2-1b"
    train_model(csv_file_path, llama_model_path)

