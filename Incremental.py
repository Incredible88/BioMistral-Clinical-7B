from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoTokenizer, DataCollatorForLanguageModeling,TrainingArguments,Trainer
import pandas as pd
from transformers import AutoModelForCausalLM
from datasets import Dataset
import transformers
from transformers import TrainerCallback
from datetime import datetime
import wandb
import os
import torch

local_model_dir = "AI-ModelScope/BioMistral-7B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the tokenizer and model from the local path
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
model = AutoModelForCausalLM.from_pretrained(local_model_dir)

directory = "PE/"

# Get all CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv') and not f.startswith('.')]

# Initialize an empty DataFrame to store the merged data
all_data = pd.DataFrame()

# Iterate through the file list, read each file and merge it into all_data
for file in csv_files:
    file_path = os.path.join(directory, file)
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        all_data = pd.concat([all_data, df], ignore_index=True)
    except Exception as e:
        print(f"Error reading file {file}: {e}")

# Extract text data, assuming the text is in the second column
data = all_data.iloc[:, 0].reset_index(drop=True)
data_list = data.tolist()

# Convert data to Hugging Face Dataset format
from datasets import Dataset
dataset = Dataset.from_dict({"text": data_list})

from datasets import load_dataset
from sklearn.model_selection import train_test_split

data = dataset.to_pandas()

# Use sklearn's train_test_split to split the dataset
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=42)

# Convert the segmented data back to a Dataset object
train_dataset = dataset.from_pandas(train_dataset)
test_dataset = dataset.from_pandas(test_dataset)

# Select Freeze First 24 Layers
num_layers_to_freeze = 24
for i in range(num_layers_to_freeze):
    for param in model.model.layers[i].parameters():
        param.requires_grad = False

# Ensure that the parameters of the embedding layer and output layer are not frozen
for param in model.model.embed_tokens.parameters():
    param.requires_grad = True
for param in model.lm_head.parameters():
    param.requires_grad = True

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""

{data_point['text']}

"""
    return tokenize(full_prompt)

tokenizer = AutoTokenizer.from_pretrained(
    local_model_dir,
    add_eos_token=True,
    add_bos_token=True, 
)

tokenizer.pad_token = tokenizer.eos_token


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result
    
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_test_dataset = test_dataset.map(generate_and_tokenize_prompt)

# Set the project and run name
project = "biomistral-7B-incremental-trained"
base_model_name = "biomistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

# Set the W&B API key
wandb_api_key = "Your_Key"
wandb.login(key=wandb_api_key)

# Initialize wandb to run
wandb.init(project=project, name=run_name)

# Configure Trainer
training_args = transformers.TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    num_train_epochs=5, 
    learning_rate=2.5e-5,  # Want about 10x smaller than the Mistral learning rate
    bf16=True,
    optim="paged_adamw_8bit",
    evaluation_strategy="steps",
    eval_steps=200,  
    logging_steps=500, 
    save_steps=5000, 
    report_to="wandb",
    run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

class WandbEvaluationCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Record metrics at the end of each epoch
        if metrics:
            wandb.log(metrics)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[WandbEvaluationCallback()]
)

# Set model configuration
model.config.use_cache = False  # Silence the warnings. Please re-enable for inference!

# Start training and tracking
trainer.train()

# End wandb run
wandb.finish()
