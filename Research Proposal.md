# Research Proposal: Utilizing Generative Large Language Models for Summarizing Clinical Records

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Prompt Engineering](#prompt-engineering)
4. [Model Selection: BioMistral-7B](#model-selection)
5. [Research Plan](#research-plan)
6. [Expected Outcomes](#expected-outcomes)
7. [Future Expectations](#future-expectations)
8. [Conclusion](#conclusion)

## Introduction:
The exponential growth of electronic health records (EHRs) has led to a vast accumulation of clinical data. Extracting actionable insights from these records is crucial for enhancing patient care, guiding clinical decisions, and supporting medical research. Traditional methods of manual chart review are time-consuming and prone to human error. The advent of large language models (LLMs) has opened new avenues for automating the summarization of medical narratives, potentially revolutionizing the way healthcare data is processed and understood.

## Dataset:
The dataset chosen for this study is the 'augmented-clinical-notes' dataset available on Hugging Face：https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes. This dataset comprises 30,000 real clinical notes, which offer a rich source for training and evaluating the performance of generative models in the medical domain. The notes are diverse and cover a wide range of medical conditions and treatments, making it an ideal dataset for developing a robust summarization model.

## Prompt Engineering:
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

## Model Selection: BioMistral-7B
For this research, we propose to utilize the BioMistral-7B model, a state-of-the-art generative LLM developed by Yanis Labrak, Adrien Bazoge, Emmanuel Morin, Pierre-Antoine Gourraud, Mickael Rouvier, and Richard Dufour. Launched in 2024, BioMistral-7B has demonstrated exceptional capabilities in processing complex biomedical and clinical text. The model stands out due to its ability to understand and generate human-like responses in a medical context, which is attributed to its extensive pre-training on a diverse corpus of biomedical literature, clinical guidelines, and patient records.  
- Paper address:https://arxiv.org/abs/2402.10373
- Open source address: https://huggingface.co/BioMistral/BioMistral-7B

**Key Features of BioMistral-7B:**
1. **Foundation Model**: BioMistral-7B is built on the Mistral 7B Instruct v0.1 model, which is designed for incorporating instructions in prompts and fine-tuning across diverse tasks.
2. **Pre-training Dataset**: The model is further pre-trained on the PubMed Central corpus, ensuring a comprehensive understanding of medical literature.
3. **Evaluation**: BioMistral-7B has been evaluated on a benchmark comprising 10 established medical question-answering (QA) tasks in English, demonstrating superior performance compared to existing open-source medical models.
4. **Multilingual Capabilities**: The model has been evaluated in multiple languages, showcasing its robustness across diverse linguistic contexts.
5. **Quantization and Model Merging**: BioMistral-7B explores lightweight models obtained through quantization and model merging approaches, making it suitable for deployment on consumer-grade devices.

您的研究计划已经非常详尽，以下是根据您的要求进行的修改和补充：

## Research Plan:
The proposed research will involve fine-tuning the BioMistral-7B model on the 'augmented-clinical-notes' dataset with a focus on generating concise and informative summaries of medical records. The process will be detailed as follows:

1. **Data Preprocessing and Annotation:**
   - Thoroughly clean and format the 'augmented-clinical-notes' dataset, ensuring that high-quality and relevant data are retained for the training set. This step will involve meticulous data annotation to mark key points within the medical records, which will serve as the ground truth for training the model.

2. **Model Fine-tuning:**
   - Utilize the LoRA (Low-Rank Adaptation) technique to reduce computational resource consumption while fine-tuning the BioMistral-7B model on the prepared dataset. LoRA introduces low-rank matrices to update model weights, allowing for a more efficient training process with reduced memory usage.

3. **Evaluation:**
   - Employ a variety of text generation evaluation methods to assess the model's output. This will include metrics such as ROUGE (Recall-Oriented Understudy for Gisting Evaluation), which is commonly used to evaluate the quality of generated text, especially in summarization tasks. Other metrics may include BLEU (Bilingual Evaluation Understudy), METEOR (Metric for Evaluation of Translation with Explicit Ordering), and perplexity scores to gauge the model's performance in terms of accuracy, coherence, and relevance.

4. **Analysis and Iteration:**
   - Analyze the model's performance based on the evaluation metrics and identify areas for further improvement. This iterative process will involve refining the model's architecture, training process, and evaluation criteria to optimize its performance on the medical summarization task.

## Expected Outcomes:
The successful implementation of this project will result in a fine-tuned model, tentatively named BioMistral-noteSum, which will be capable of providing concise summaries of medical records. This model will be made openly available on Hugging Face, allowing the medical community to utilize and build upon this resource for various applications in the healthcare sector.

## Future Expectations:
The extracted key points from the medical records can be integrated with knowledge graphs to enable more generalized applications in the medical field. This integration can enhance the model's ability to provide context-aware summaries and support decision-making processes in clinical settings.

## Conclusion:
The utilization of BioMistral-7B for summarizing medical records represents a significant step forward in leveraging AI for healthcare applications. This research has the potential to transform the way medical data is analyzed and interpreted, ultimately contributing to better patient outcomes and more informed clinical decisions.

By following this detailed research plan, we aim to develop a model that not only meets the current needs of medical data summarization but also paves the way for future advancements in AI-assisted healthcare. 

