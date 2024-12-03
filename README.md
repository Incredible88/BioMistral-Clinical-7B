# BioMistral-Clinical 7B

BioMistral-Clinical 7B, a new LLM specifically designed for clinical applications, built upon the foundation of the BioMistral-7B model.

This model now is avaliable at:  https://huggingface.co/ZiweiChen/BioMistral-Clinical-7B 

## How to use

Loading the model from Hunggingface:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ZiweiChen/BioMistral-Clinical-7B")
model = AutoModelForCausalLM.from_pretrained("ZiweiChen/BioMistral-Clinical-7B")
```
Lightweight model loading can be used - using 4-bit quantization!
```python
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git

from transformers import  AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("ZiweiChen/BioMistral-Clinical-7B")
model = AutoModelForCausalLM.from_pretrained("ZiweiChen/BioMistral-Clinical-7B", quantization_config=bnb_config)

```
How to Generate text:
```python
model_device = next(model.parameters()).device

prompt = """
### Question:

How to treat severe obesity?

### Answer:
"""
model_input = tokenizer(prompt, return_tensors="pt").to(model_device)

with torch.no_grad():
    output = model.generate(**model_input, max_new_tokens=100)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print(answer)
```
## Incremental learning
The process of  incremental learning:    
![image/png](https://cdn-uploads.huggingface.co/production/uploads/64850ea733e82e2c99337143/7i75kNDOR08WU4SXIhdb5.png)  
The  training process records:    
![image/png](https://cdn-uploads.huggingface.co/production/uploads/64850ea733e82e2c99337143/E1ES03zUNw8-mZ98Tz9lh.png)
## Clinical Scenario Analysis
More informative answer:  
![image](https://github.com/user-attachments/assets/21a21e75-6014-43cf-8a40-0f296c0974a6)

##  Supervised Fine-tuning Benchmark

![image](https://github.com/user-attachments/assets/8c0f08e7-cad0-4203-8b8c-fcf648831fb5)

**CAUTION!** Both direct and downstream users need to be informed about the risks, biases, and constraints inherent in the model. While the model can produce natural language text, our exploration of its capabilities and limitations is just beginning. In fields such as medicine, comprehending these limitations is crucial. Hence, we strongly advise against deploying this model for natural language generation in production or for professional tasks in the realm of health and medicine.
