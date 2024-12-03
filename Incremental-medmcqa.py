from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


local_model_dir = "/ai/Chenzhirui/czw/biomistral-biomistral-7B-incremental-trained1.2/checkpoint-10000"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(local_model_dir, quantization_config=bnb_config)

original_model_dir = "AI-ModelScope/BioMistral-7B"

tokenizer = AutoTokenizer.from_pretrained(original_model_dir)

import pandas as pd
from datasets import Dataset

train_df = pd.read_csv('medmcqa/train_dataset.csv')
validation_df = pd.read_csv('medmcqa/valid_dataset.csv')
test_df = pd.read_csv('medmcqa/test_dataset.csv')

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

import ast
import re
tokenizer.pad_token = tokenizer.eos_token
def generate_and_tokenize_prompt(examples):
    # 修复 options 字段字符串格式（添加缺失的逗号）
    options = {
        'A': examples['A'],
        'B': examples['B'],
        'C': examples['C'],
        'D': examples['D']
    }
    # 将选项字典转换为格式化的字符串，每个选项换行
    option_str = '\n'.join([f"{k}: {v}" for k, v in options.items()])
    
    # 构建完整的提示文本
    prompt_str = f"""From the MedmcQA Dataset: Given the medical question, provide an accurate answer.

    ###Question:
    {examples['question']} ?

    ###Options:\n{option_str}
    
    ###Answers:
    {examples['cop']}: {examples['Answer']}"""
    
    # 标记化提示
    result = tokenizer(
        prompt_str,
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


# 应用到训练集和验证集
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt, batched=False)
tokenized_val_dataset = validation_dataset.map(generate_and_tokenize_prompt, batched=False)

from peft import prepare_model_for_kbit_training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)


model = get_peft_model(model, config)

import transformers
from datetime import datetime
import wandb

# 设置项目和运行名称
project = "Incremental-Medmcqa-SFT"
base_model_name = "biomistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

# 设置 W&B API key
wandb_api_key = "5f5f94d3de9157cf146ad88ecc4e0518a7a7549e"
wandb.login(key=wandb_api_key)

# 初始化 wandb 运行
wandb.init(project=project, name=run_name)


# 配置 Trainer
training_args = transformers.TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    num_train_epochs=5, 
    learning_rate=2.5e-5,  # Want about 10x smaller than the Mistral learning rate
    bf16=True,
    optim="paged_adamw_8bit",
    evaluation_strategy="steps",
    eval_steps=200,  # 根据你的需要调整评估的步长
    logging_steps=500,  # 记录日志的步长
    save_steps=1000,  # 保存模型的步长
    report_to="wandb",
    run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

class WandbEvaluationCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 在每个epoch结束时记录指标
        if metrics:
            wandb.log(metrics)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[WandbEvaluationCallback()]
)

# 设置模型配置
model.config.use_cache = False  # Silence the warnings. Please re-enable for inference!

# 开始训练并跟踪
trainer.train()

# 结束 wandb 运行
wandb.finish()