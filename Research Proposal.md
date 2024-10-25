# Research Proposal: Improving Clinical Knowledge in Large Language Models through Incremental Learning Methods

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Prompt Engineering](#prompt-engineering)
4. [Model Selection: BioMistral-7B](#model-selection)
5. [Research Plan](#research-plan)
6. [Expected Outcomes](#expected-outcomes)


## Introduction:

The exponential growth of electronic health records (EHRs) has led to an unprecedented accumulation of clinical data, presenting both an opportunity and a challenge. The opportunity lies in the potential insights that can be gleaned from this data to enhance patient care, inform clinical decisions, and support medical research. The challenge, however, is the sheer volume and complexity of the data, which traditional manual review methods are ill-equipped to handle efficiently and accurately.

Large Language Models (LLMs), particularly those specialized for biomedical domains like BioMistral-7B, offer a promising avenue for automating the extraction and summarization of medical narratives. BioMistral-7B, an open-source LLM, has demonstrated remarkable capabilities in processing complex clinical texts. To further refine its understanding and better serve real-world clinical needs, we propose to employ incremental learning techniques. This approach involves continuously updating the model's knowledge base with new clinical data, allowing it to adapt and improve over time. Specifically, we will leverage a structured format of clinical notes to provide a clear and organized input for the model. This method will enable the model to more accurately identify patterns, correlations, and dependencies between different aspects of patient care, leading to improved diagnostic and treatment predictions. By focusing on the key elements of each patient's medical history and outcome, and by using prompt engineering to highlight these elements, we aim to enhance the model's ability to generalize and adapt to new, unseen medical data, thereby achieving a deeper comprehension of the clinical domain.

### Research Gaps:

The primary research gaps this project aims to address are:

1. **Depth of Clinical Understanding**: There is a need to enhance the understanding of the specific nuances of clinical data, including the subtleties of disease symptoms, diagnostic procedures, and treatment efficacy.  

2. **Adaptability to Clinical Notes**: Clinical notes are rich in unstructured data that may not align with the structured datasets used for initial model training. There is a gap in adapting LLMs to effectively interpret and learn from the diverse and complex nature of real-world clinical narratives.

3. **Incremental Learning for Continuous Improvement**: The ability of LLMs to incrementally learn from new data, thereby continuously improving their knowledge base and performance, is a critical area for development. This is particularly important in the fast-paced field of healthcare, where medical knowledge and best practices evolve continuously.

The 'augmented-clinical-notes' dataset, available on Hugging Face, is a valuable resource for this project. It contains a wealth of clinical notes that provide a comprehensive view of patient symptoms, diagnoses, treatments, and outcomes. The quality and breadth of this dataset make it an excellent choice for training and evaluating the performance of generative models in the medical domain. By employing precise prompting engineering (PE), we can extract key knowledge points from these notes, which will be used to incrementally train BioMistral-7B. This approach has the potential to significantly enhance the model's clinical knowledge base, thereby improving its ability to generate informative and structured medical summaries.

To evaluate the effectiveness of the incremental training, we will utilize the Supervised Fine-tuning Benchmark datasets mentioned in the BioMistral-7B paper. These datasets, available at [bigbio/med_qa](https://huggingface.co/datasets/bigbio/med_qa) and [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa), provide a robust framework for benchmarking the performance of medical LLMs. Although there may have been temporary issues accessing these links, they offer a comprehensive set of medical QA tasks that will allow us to rigorously assess the improvements in BioMistral-7B's capabilities after incremental training.

By addressing these gaps, this project aims to enhance the clinical knowledge understanding of BioMistral-7B, making it a more powerful tool for processing and summarizing medical records, and ultimately contributing to more informed clinical decisions and better patient outcomes.

### Research Questions

The following research questions will guide the exploration and development within this project, ensuring a focused and goal-oriented approach towards enhancing BioMistral-7B for clinical knowledge tasks:

1. **RQ1: How can incremental learning be effectively integrated into LLMs to improve their understanding of clinical narratives?**
   - This question aims to explore the feasibility and methods of implementing incremental learning within the BioMistral-7B model. The goal is to determine the best practices for continuously updating the model's knowledge base with new clinical data. Specifically, this incremental learning will utilize self-supervised training techniques, enabling the generative large model to better comprehend domain-specific knowledge before fine-tuning on downstream tasks.


2. **RQ2: What are the most effective prompt engineering strategies for extracting relevant medical information from unstructured clinical notes?**
   - This research question focuses on developing and refining prompt engineering techniques to maximize the extraction of key medical details from clinical notes. The aim is to identify prompts that lead to the most accurate and comprehensive data structuring. To achieve this, we will explore how core statements within clinical notes, supported by contextual background, can be effectively highlighted through prompt engineering. This approach will enable the large language model to not only extract data but also to better understand the causal relationships within the medical domain. By doing so, the model will be able to grasp the underlying mechanisms that connect symptoms, diagnoses, treatments, and outcomes, thereby enhancing its ability to process and summarize clinical narratives in a manner that is coherent with the domain's knowledge structure.


3. **RQ3: To what extent can a structured format of clinical notes enhance the model's ability to generalize and adapt to new, unseen medical data?**
   - This research question explores the impact of using a structured format, such as JSON, to enhance the BioMistral-7B model's capacity to generalize and adapt to novel medical data. The structured format is designed to capture the complexity and nuances of clinical narratives by articulating critical relationships and causal links among medical entities. This approach mirrors the associative capabilities of knowledge graphs, allowing the model to encapsulate core assertions within their contextual backdrop. The investigation will assess whether this method can provide the large language model with a more profound understanding of the interconnections within clinical data, similar to the effects achieved by knowledge graphs. By doing so, this research will offer insights into the potential of structured data representations to bolster clinical knowledge comprehension, especially in the absence of mature medical knowledge graph models.


4. **RQ4: How does the performance of the incrementally trained medical language model compare to the original model on standardized medical question-answering tasks, and what are the potential limitations and ethical considerations of using such a model in real-world clinical settings?**
   - This research question is designed to assess the effectiveness of incremental training on a medical language model, specifically BioMistral-7B, by comparing its performance against the original model on standardized medical QA tasks. The primary objective is to quantify the improvements in the model's ability to understand clinical knowledge and to summarize medical records effectively. Additionally, this question aims to investigate the potential limitations and ethical considerations associated with the deployment of AI models in real-world clinical settings. By examining these factors, the research will provide a thorough evaluation of the benefits and challenges of using incrementally trained medical language models in practical healthcare environments, with a focus on ensuring their application is both effective and adheres to ethical standards.

By addressing these research questions, the project will not only enhance the capabilities of BioMistral-7B but also contribute to a deeper understanding of the role of AI in healthcare, ultimately aiming to improve patient outcomes and support clinical decision-making.  

## Dataset:
The dataset chosen for this study is the 'augmented-clinical-notes' dataset available on Hugging Face：https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes. This dataset comprises 30,000 real clinical notes, which offer a rich source for training and evaluating the performance of generative models in the medical domain. The notes are diverse and cover a wide range of medical conditions and treatments, making it an ideal dataset for developing a robust summarization model.

## Prompt Engineering:
The primary objective of this research is to develop a structured approach to clinical note analysis that enhances the granularity and relational clarity of patient data. By transforming unstructured clinical notes into a standardized JSON format, we aim to capture more detailed and specific information for each patient case. This method will allow for clearer segmentation of data into distinct categories such as Chief Complaints, Medical History, Diagnostic Findings, Diagnosis, Treatment, and Outcome, with each category having a clearly defined set of sub-fields.

The structured JSON format will serve as the sole input for our model, replacing the need for raw text data. This approach is anticipated to significantly benefit the model's training and incremental learning processes by providing a more organized and comprehensive dataset. The model will be able to more accurately identify patterns, correlations, and dependencies between different aspects of patient care, leading to improved diagnostic and treatment predictions.

By focusing on the key elements of each patient's medical history and outcome, the model will be better equipped to handle new, unseen data, thus enhancing its generalization capabilities. This structured input will also facilitate the model's ability to incremental training, allowing it to adapt to new information and updates in medical knowledge over time without the need for complete retraining.

The use of this structured data format will not only streamline the data preprocessing stage but also provide a robust foundation for building a model that can scale and adapt to the evolving complexities of clinical data management.   

**Prompt as follows:** 
``` 
As a professional clinical notes organizer, my task is to take a set of clinical notes and structure them into a JSON format. This format will help in standardizing the patient data for better analysis and record-keeping. Below, I will explain each field in the JSON structure and provide an example based on a hypothetical clinical note.

JSON Field Explanation:

PatientInformation: This section captures the patient's presenting complaints and historical medical information.   
ChiefComplaints: A list of the main issues the patient is experiencing, as reported by the patient.  
MedicalHistory: Includes any past injuries or significant health issues that are relevant to the current condition.  
DiagnosticFindings: A list of diagnostic tests conducted and their findings. Each item includes the type of Test conducted and the Finding from that test  

Diagnosis: This section includes the diagnosed disease.  
Disease: Details the name, type, and location of the diagnosed disease.  

TreatmentAndOutcome: This section outlines the treatment provided and the subsequent outcome.  
Treatment: Describes the type of treatment and any specific details about the procedure.  
Type: The general category of treatment.  
Details: Specifics about what the treatment entailed.  
Postoperative Course: Information about the recovery process after surgery.  
Recovery: General description of the recovery process.  
FollowUp: Information about any follow-up care or observations.  
FunctionalStatus: Any observations about the patient's functionality or quality of life post-treatment.    

Example Output:

{
  "PatientInformation": {
    "ChiefComplaints": [
      "Complaints of pain and swelling in the right back for several weeks",
      "No significant health problems except a thoracic trauma one year prior"
    ],
    "MedicalHistory": {
      "PreviousInjury": "Thoracic trauma with a simple fracture of the 9th right rib"
    },
    "DiagnosticFindings": [
      {
        "Test": "X-ray",
        "Finding": "A shadow in the lower part of the right hemithorax"
      },
      {
        "Test": "CT-scan",
        "Finding": "A tumor with heterogeneous density and destruction of the 9th rib"
      }
    ]
  },
  "Diagnosis": {
    "Disease": {
      "Name": "Sclerosing xanthofibroma",
      "Type": "Benign tumor",
      "Location": "Thoracic wall"
    },
  },
  "TreatmentAndOutcome": {
    "Treatment": {
      "Type": "Surgical resection and plastic repair",
      "Details": "Involving three ribs and reconstruction with polypropylene mesh"
    },
    "Postoperative Course": {
      "Recovery": "Uneventful",
      "DischargeStatus": "Good condition"
    },
    "FollowUp": {
      "Duration": "Two years",
      "FunctionalStatus": "Patient returned to work one month after surgery"
    }
  }
}

```
## LLMs Selection: BioMistral-7B
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
   - **Datasets**: For the evaluation phase, I will be utilizing the same datasets mentioned in the BioMistral-7B paper: [bigbio/med_qa](https://huggingface.co/datasets/bigbio/med_qa) and [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa). These datasets will provide a relevant and challenging benchmark for assessing the performance of the incrementally trained model.
   - **Evaluation Method**: The model will be evaluated using the Supervised Fine-tuning Benchmark method, which is consistent with the approach taken in the original BioMistral-7B paper. This method ensures that the evaluation is conducted in a standardized manner, allowing for a direct comparison of the incrementally trained model against the original model.
   - **Performance Comparison**: The performance of the incrementally trained model will be compared against the original BioMistral-7B model on these benchmark tasks. This comparison will serve to quantify the improvements in the model's understanding of clinical knowledge and its ability to summarize medical records effectively.  

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
