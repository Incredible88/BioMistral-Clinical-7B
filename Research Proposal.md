# Research Proposal: Enhancing BioMistral-7B with Incremental Learning for Clinical Knowledge

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Prompt Engineering](#prompt-engineering)
4. [Model Selection: BioMistral-7B](#model-selection)
5. [Research Plan](#research-plan)
6. [Expected Outcomes](#expected-outcomes)


## Introduction:

The exponential growth of electronic health records (EHRs) has led to an unprecedented accumulation of clinical data, presenting both an opportunity and a challenge. The opportunity lies in the potential insights that can be gleaned from this data to enhance patient care, inform clinical decisions, and support medical research. The challenge, however, is the sheer volume and complexity of the data, which traditional manual review methods are ill-equipped to handle efficiently and accurately.

Large Language Models (LLMs), particularly those specialized for biomedical domains like BioMistral-7B, offer a promising avenue for automating the extraction and summarization of medical narratives. BioMistral-7B, an open-source LLM, has demonstrated remarkable capabilities in processing complex clinical texts. Yet, the model's understanding of the intricate relationships between medical conditions, symptoms, diagnoses, treatments, and outcomes can be further refined to better serve real-world clinical needs.

### Research Gaps:

The primary research gaps this project aims to address are:

1. **Depth of Clinical Understanding**: While BioMistral-7B is adept at handling biomedical texts, there is a need to enhance its understanding of the specific nuances of clinical data, including the subtleties of disease symptoms, diagnostic procedures, and treatment efficacy.

2. **Adaptability to Clinical Notes**: Clinical notes are rich in unstructured data that may not align with the structured datasets used for initial model training. There is a gap in adapting LLMs to effectively interpret and learn from the diverse and complex nature of real-world clinical narratives.

3. **Incremental Learning for Continuous Improvement**: The ability of LLMs to incrementally learn from new data, thereby continuously improving their knowledge base and performance, is a critical area for development. This is particularly important in the fast-paced field of healthcare, where medical knowledge and best practices evolve continuously.

The 'augmented-clinical-notes' dataset, available on Hugging Face, is a valuable resource for this project. It contains a wealth of clinical notes that provide a comprehensive view of patient symptoms, diagnoses, treatments, and outcomes. The quality and breadth of this dataset make it an excellent choice for training and evaluating the performance of generative models in the medical domain. By employing precise prompting engineering (PE), we can extract key knowledge points from these notes, which will be used to incrementally train BioMistral-7B. This approach has the potential to significantly enhance the model's clinical knowledge base, thereby improving its ability to generate informative and structured medical summaries.

To evaluate the effectiveness of the incremental training, we will utilize the Supervised Fine-tuning Benchmark datasets mentioned in the BioMistral-7B paper. These datasets, available at [bigbio/med_qa](https://huggingface.co/datasets/bigbio/med_qa) and [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa), provide a robust framework for benchmarking the performance of medical LLMs. Although there may have been temporary issues accessing these links, they offer a comprehensive set of medical QA tasks that will allow us to rigorously assess the improvements in BioMistral-7B's capabilities after incremental training.

By addressing these gaps, this project aims to enhance the clinical knowledge understanding of BioMistral-7B, making it a more powerful tool for processing and summarizing medical records, and ultimately contributing to more informed clinical decisions and better patient outcomes.


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

## Research Plan:

This research plan outlines a systematic approach to incrementally train the BioMistral-7B model using the 'augmented-clinical-notes' dataset, with the goal of enhancing its clinical knowledge understanding. The plan is designed to leverage the model's existing capabilities and improve its performance through targeted fine-tuning.

1. **Data Preparation and Analysis**:
   - Begin with a thorough analysis of the 'augmented-clinical-notes' dataset to understand its structure, content, and the range of medical conditions it covers.
   - Perform data cleaning to ensure that the dataset is free from inconsistencies or inaccuracies that could affect the training process.

2. **Prompt Engineering (PE)**:
   - Develop a set of prompts that will effectively guide the model in extracting key information from the clinical notes, such as symptoms, diagnoses, treatments, and outcomes.
   - Refine the prompts to maximize the relevance and accuracy of the knowledge extracted from the dataset.

3. **Incremental Training**:
   - Use the extracted knowledge points as input for incremental training of the BioMistral-7B model.
   - Implement a controlled training regimen that introduces new knowledge points gradually, allowing the model to integrate this information effectively without forgetting previously learned concepts.

4. **Model Evaluation**:
   - Employ the Supervised Fine-tuning Benchmark datasets to evaluate the performance of the incrementally trained model.
   - Compare the performance of the incrementally trained model against the original BioMistral-7B model on these benchmark tasks to quantify improvements in clinical knowledge understanding.

5. **Performance Metrics**:
   - Utilize evaluation metrics such as accuracy, precision, recall, and F1 score to assess the model's performance on the benchmark datasets.
   - Analyze the model's output quality in terms of coherence, relevance, and completeness of the summaries generated from the clinical notes.

By following this research plan, we aim to enhance BioMistral-7B's ability to understand and summarize clinical records effectively. The expected outcome is a model that can provide more accurate, coherent, and contextually relevant summaries, thereby supporting clinical decision-making and contributing to the advancement of AI-assisted healthcare.

## Expected Outcomes:

The project anticipates the following key outcomes:

1. **Advanced Clinical Knowledge**: The incrementally trained BioMistral-7B model is expected to exhibit an enhanced understanding of clinical data, including complex disease symptoms, diagnoses, and treatment efficacy.

2. **Improved Summarization**: The model should generate more accurate and insightful summaries of clinical notes, which are essential for efficient patient care and medical research.

3. **Benchmarked Performance**: Utilizing the Supervised Fine-tuning Benchmark, we expect to demonstrate the improved performance of the incrementally trained model over the original BioMistral-7B, validating the effectiveness of our approach.

4. **Model Availability**: The incrementally trained BioMistral-7B model, post successful enhancement, will be made publicly available on Hugging Face. This will allow the broader research community and healthcare practitioners to leverage this improved model for their own applications and further research.

By achieving these outcomes, the project will contribute to the advancement of AI in healthcare, providing tools that can better support the interpretation and utilization of medical data.
