from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


# 指定模型和分词器的本地路径
local_model_dir = "AI-ModelScope/BioMistral-7B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 从本地路径加载分词器和模型
# tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
# model = AutoModelForCausalLM.from_pretrained(local_model_dir, quantization_config=bnb_config)
model = AutoModelForCausalLM.from_pretrained(local_model_dir)

from transformers import AutoTokenizer, DataCollatorForLanguageModeling,TrainingArguments,Trainer
import pandas as pd
from transformers import AutoModelForCausalLM
from datasets import Dataset

import os
directory = "PE/"

# 获取目录下所有的 CSV 文件
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv') and not f.startswith('.')]

# 初始化一个空的 DataFrame 来存储合并后的数据
all_data = pd.DataFrame()

# 遍历文件列表，读取每个文件并合并到 all_data 中
for file in csv_files:
    file_path = os.path.join(directory, file)
    try:
        # 读取 CSV 文件
        df = pd.read_csv(file_path, on_bad_lines='skip')
        # 将当前文件的数据追加到 all_data 中
        all_data = pd.concat([all_data, df], ignore_index=True)
    except Exception as e:
        print(f"读取文件 {file} 时发生错误：{e}")

# 提取文本数据，假设文本在第二列
data = all_data.iloc[:, 0].reset_index(drop=True)
data_list = data.tolist()

# 将数据转换为 Hugging Face Dataset 格式
from datasets import Dataset
dataset = Dataset.from_dict({"text": data_list})

from datasets import load_dataset
from sklearn.model_selection import train_test_split

data = dataset.to_pandas()

# 使用 sklearn 的 train_test_split 来分割数据集
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=42)

# 将分割后的数据转换回 Dataset 对象
train_dataset = dataset.from_pandas(train_dataset)
test_dataset = dataset.from_pandas(test_dataset)


# 现在你有了训练集、测试集和评估集
print(train_dataset)
print(test_dataset)


num_layers_to_freeze = 24
for i in range(num_layers_to_freeze):
    for param in model.model.layers[i].parameters():
        param.requires_grad = False

# 确保嵌入层和输出层的参数不会被冻结
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

import transformers
from transformers import TrainerCallback
from datetime import datetime
import wandb

# 设置项目和运行名称
project = "biomistral-7B-incremental-trained1.2"
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
    save_steps=5000,  # 保存模型的步长
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
    eval_dataset=tokenized_test_dataset,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[WandbEvaluationCallback()]
)

# 设置模型配置
model.config.use_cache = False  # Silence the warnings. Please re-enable for inference!

# 开始训练并跟踪
trainer.train()

# 结束 wandb 运行
wandb.finish()