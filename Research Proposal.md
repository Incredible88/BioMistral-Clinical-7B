# Research Proposal: Utilizing Generative Large Language Models for Summarizing Medical Records

**Introduction:**
The exponential growth of electronic health records (EHRs) has led to a vast accumulation of clinical data. Extracting actionable insights from these records is crucial for enhancing patient care, guiding clinical decisions, and supporting medical research. Traditional methods of manual chart review are time-consuming and prone to human error. The advent of large language models (LLMs) has opened new avenues for automating the summarization of medical narratives, potentially revolutionizing the way healthcare data is processed and understood.

**Dataset:**
The dataset chosen for this study is the 'augmented-clinical-notes' dataset available on Hugging Face. This dataset comprises 30,000 real clinical notes, which offer a rich source for training and evaluating the performance of generative models in the medical domain. The notes are diverse and cover a wide range of medical conditions and treatments, making it an ideal dataset for developing a robust summarization model.

**Prompt Engineering:**
To ensure the model generates structured and informative summaries, we will use the following prompt format:

```json
{
  "Conclusion statements": {
    "Core": "Summarize the medical condition diagnosed, based on clinical findings and diagnostic tests.",
    "Context": [
      "List the symptoms and signs that were observed and contributed to the diagnosis."
    ],
    "Disc. Context": [
      "List additional details that provide context or further explanation of the diagnosis."
    ]
  },
  "Second statements": {
    "Core": "Describe the outcome of the treatment and the patient's subsequent health status.",
    "Context": [
      "Describe the main treatment methods that were administered."
    ],
    "Disc. Context": [
      "Describe the results of the treatment, including any complications or improvements."
    ]
  },
  "Conclusion": {
    "Disease": "Name the diagnosed disease.",
    "Treatment": "List the treatments performed.",
    "Findings": "Summarize key diagnostic findings.",
    "Outcome": "State the overall outcome after treatment."
  }
}
```

**Example Output:**
```json
{
  "Conclusion statements": {
    "Core": "The patient was diagnosed with sclerosing xanthofibroma of the thoracic wall, a benign tumor with distinct clinical and radiological features.",
    "Context": [
      "Complaints of pain and swelling in the right back for several weeks",
      "No significant health problems except a thoracic trauma one year prior"
    ],
    "Disc. Context": [
      "X-ray showed a shadow in the lower part of the right hemithorax",
      "CT-scan revealed a tumor with heterogeneous density and destruction of the 9th rib"
    ]
  },
  "Second statements": {
    "Core": "The patient underwent successful surgical removal of the tumor and showed no signs of recurrence two years post-surgery.",
    "Context": [
      "Oncologic resection of the tumor involving three ribs"
    ],
    "Disc. Context": [
      "Postoperative recovery was uneventful, with the patient discharged in good condition",
      "Two-year follow-up CT-scan showed no recurrence or new developments"
    ]
  },
  "Conclusion": {
    "Disease": "Sclerosing xanthofibroma",
    "Treatment": "Surgical resection and plastic repair with polypropylene mesh",
    "Findings": "Tumor of the thoracic wall with micronodular infiltrations in both lungs",
    "Outcome": "No recurrence or new developments two years post-surgery, patient returned to work one month after surgery"
  }
}
```

**Model Selection: BioMistral-7B**
For this research, we propose to utilize the BioMistral-7B model, a state-of-the-art generative LLM developed by Yanis Labrak, Adrien Bazoge, Emmanuel Morin, Pierre-Antoine Gourraud, Mickael Rouvier, and Richard Dufour. Launched in 2024, BioMistral-7B has demonstrated exceptional capabilities in processing complex biomedical and clinical text. The model stands out due to its ability to understand and generate human-like responses in a medical context, which is attributed to its extensive pre-training on a diverse corpus of biomedical literature, clinical guidelines, and patient records.  
- Paper address:https://arxiv.org/abs/2402.10373
- Open source address: https://huggingface.co/BioMistral/BioMistral-7B

**Key Features of BioMistral-7B:**
1. **Foundation Model**: BioMistral-7B is built on the Mistral 7B Instruct v0.1 model, which is designed for incorporating instructions in prompts and fine-tuning across diverse tasks.
2. **Pre-training Dataset**: The model is further pre-trained on the PubMed Central corpus, ensuring a comprehensive understanding of medical literature.
3. **Evaluation**: BioMistral-7B has been evaluated on a benchmark comprising 10 established medical question-answering (QA) tasks in English, demonstrating superior performance compared to existing open-source medical models.
4. **Multilingual Capabilities**: The model has been evaluated in multiple languages, showcasing its robustness across diverse linguistic contexts.
5. **Quantization and Model Merging**: BioMistral-7B explores lightweight models obtained through quantization and model merging approaches, making it suitable for deployment on consumer-grade devices.

**Research Plan:**
The proposed research will involve fine-tuning the BioMistral-7B model on the 'augmented-clinical-notes' dataset to generate concise and informative summaries of medical records. The training process will focus on optimizing the model's ability to identify and articulate the most relevant aspects of each case, such as the diagnosed condition, treatment administered, and patient outcomes.

The research will be structured as follows:
1. **Data Preprocessing**: Clean and format the 'augmented-clinical-notes' dataset for input into the model.
2. **Model Fine-tuning**: Adjust the BioMistral-7B model on the prepared dataset to enhance its performance on the summarization task.
3. **Evaluation**: Assess the model's output against a set of predefined metrics, including accuracy, coherence, and relevance.
4. **Analysis**: Analyze the model's performance and identify areas for further improvement.

**Expected Outcomes:**
The successful implementation of this project could lead to the development of a powerful tool for healthcare providers to quickly grasp the essence of patient cases, thereby improving the efficiency of clinical workflows. Additionally, the insights gained from this research could inform the future development of AI-assisted diagnostic tools and personalized treatment plans.

**Conclusion:**
The utilization of BioMistral-7B for summarizing medical records represents a significant step forward in leveraging AI for healthcare applications. This research has the potential to transform the way medical data is analyzed and interpreted, ultimately contributing to better patient outcomes and more informed clinical decisions.
