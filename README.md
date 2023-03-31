# Syllabus-to-O\*NET


To predict career mobility, earnings, and future skill acquisition we need to develop a framework capable of unpacking the individual’s skills.
To do so, in this work we develop a natural language processing framework to identify the O\*NET Detailed Work Activities (DWAs) and Tasks from course descriptions.

## Description:

The following figure provides a high-level representation of the proposed system.

<p align="center">

  <img width=95% height=60% src="https://github.com/AlirezaJavadian/Syllabus-to-ONET/blob/17d5d45b80280534c0e2891f748944b653a02a56/syll_2_onet_workflow.png">

</p>

In order to extract the skills from course syllabi, at first we needed to remove the non-relevant text (such as the course schedule and policy) from the syllabi.
Each course syllabus record has "Course Description" metadata which is plain text. 
In other words, it doesn't have any structure which helps us to separate the "general" related sentences such as office hours, integrity statement, etc. from the "outcome" related sentences which contain the course.
Since there is no benchmark-labeled dataset for this purpose, we construct the following pipeline.


- **Sentence Segmentation:** At first, we utilize sentence segmentation provided by [Stanza](https://doi.org/10.48550/arXiv.2003.07082) which breaks a text into its sentences. *Note: This step is optional. If the course descriptions contain only the learning materials and do not need any cleaning -> with_bow_cleaning = False*
- **Human-in-the-loop Sentence Tagging:** In the next step, we create the following two lists for labeling. One containing the terms/phrases that mostly appear in "General" sentences, i.e. non-content related sentences (e.g., Plagiarism, Attendance, Office hour, etc.). The other contains the "Outcome" related terms (e.g., Analyze, Versus, Outcome, etc.). After some iterations of checking over $6,000$ course syllabi, the resulting General list contains 356 terms and phrases and the Content related list contains $51$ terms and phrases. It is worth mentioning that, while building the lists, we carefully revise the lists so that we do not remove the sentences from fields of study such as Education and Psychology which might contain General terms as their actual content-related. In the next step, for each sentence, we add two binary columns as General and Outcome. Then, if the sentence contains any of the corresponding terms, we change the value to 1. As a result, each sentence could belong to one of the categories reported in the following below. After evaluating the results using different combinations of the categories presented in the following Table, we keep only the sentences tagged as "Pure Outcome" (General=0 & Outcome=1).

<div align="center">

  <table><thead><tr><th>Category</th><th>General</th><th>Outcome</th></tr></thead><tbody><tr><td>Pure General</td><td>1</td><td>0</td></tr><tr><td>Pure Outcome</td><td>0</td><td>1</td></tr><tr><td>Mixed</td><td>1</td><td>1</td></tr><tr><td>Unknown</td><td>0</td><td>0</td></tr></tbody></table>

</div>

- **Sentence Embedding:** In the next step, we employ Sentence Embeddings using Siamese Bidirectional Encoder Representations from Transformers-Networks [(SBERT)](https://doi.org/10.48550/arXiv.1908.10084) ([all-mpnet-base-v2 model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) which transforms each sentence into a space of 768 dimensions.

- **Skills Similarity Calculation:** After embedding the course syllabi sentences and DWAs/Tasks, we compute the pairwise cosine similarity between each course syllabus and each DWA/Task. Given the similarity scores for all the sentences of a syllabus, we choose the maximum score for each DWA/Task (i.e., the most similar sentence) to obtain a 1-D vector of size 2070 (in case of DWA) or 18429 (in case of Task) representing the scores for all the DWAs/Tasks for a given course.
In other words, we assume that each score implies how much that course prepares the student for the given DWA/Task.



## Required packages:
### Reinstall the required packages under a conda environment named 'SylltoONET' using the provided export file ('package-list.txt') as follow.
```
~$ conda create -n SylltoONET --file package-list.txt
```

## Run:
1. Install the required packages as described before.
2. Update the variables and paths in the 'settings.py' file. *Please make sure to follow the comments/instructions provided for each variable.*
3. Run 'main.py' as follow.
```
~$ anaconda3/envs/SylltoONET/bin/python /Syllabus-to-ONET/main.py
```
