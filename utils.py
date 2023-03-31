import stanza
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

import os
import sys


# Step 2: Sentence Segmentation
def stanza_syl_2_lines_array(input_df, intermediate_path, file_name, segmnt_intermediate_save=True):

    """" Function for parsing the syllabi and save one row for each line.
         - Inputs:
         -- input_df: The dataframe containing the course syllabi text and their ids.
         -- intermediate_path: path to save the intermediate results of the sentence segmentation step.
         -- file_name: The current CSV file name to be used for saving the intermediate result.
         -- segmnt_intermediate_save: If ture, saves the intermediate segmented sentences.
         - Outputs: A dataframe containing the lines of syllabi and save the results in gzip format for future refrence.
         """

    # Loading the Stanza model
    input_nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=False, use_gpu=True, verbose=False)

    # Basic pre-processing
    ## Removing non-ascii characters
    input_df['syllabus_text'] = input_df.apply(lambda row: row['syllabus_text'].encode("ascii", "ignore"), axis=1)
    input_df['syllabus_text'] = input_df.apply(lambda row: row['syllabus_text'].decode("ascii", "ignore"), axis=1)
    ## Replacing white space characters with space
    input_df['syllabus_text'] = input_df.apply(
        lambda row: row['syllabus_text'].replace('\\n', ' ').replace('\n', ' ').replace('\t', ' ').replace('\t',
                                                                                                           ' ').replace(
            '\r', ' ').replace('\x0b', ' ').replace('\x0c', ' '), axis=1)
    ## replace multiple white spaces with one white space
    input_df['syllabus_text'] = input_df.apply(lambda row: " ".join(row['syllabus_text'].split()), axis=1)

    lines_separated_dfs_list = []

    for idx in range(len(input_df)):
        this_osp_id = input_df.loc[idx, 'syllabus_id']
        this_syll = input_df.loc[idx, 'syllabus_text']
        this_doc = input_nlp(this_syll)
        all_sentences = [sentence.text for sentence in this_doc.sentences]
        temp_df = pd.DataFrame()
        temp_df['syllabus_text'] = all_sentences
        temp_df['syllabus_id'] = this_osp_id
        temp_df = temp_df[['syllabus_id', 'syllabus_text']]
        temp_df['sent_id'] = [i for i in range(len(temp_df))]
        lines_separated_dfs_list.append(temp_df)

    df_result = pd.concat(lines_separated_dfs_list, ignore_index=True)

    if segmnt_intermediate_save:
        df_result.to_csv(os.path.join(intermediate_path, 'sentSegIntermediate_' + file_name + '.gzip'),
                         compression='gzip', index=False)

    return df_result


# Step 3: General and Outcome sentences filtering
def bag_of_word_tagger(input_df, intermediate_path, file_name, bow_intermediate_save=True):
    """ Function for tagging OSP lines according to the list of Outcome General words/phrases.
        - Inputs:
        -- input_df: The segmented course syllabi df.
        -- intermediate_path: Path to save the intermediate results of the sentence tagging step.
        -- file_name: The current CSV file name to be used for saving the intermediate result.
        -- bow_intermediate_save: If ture, saves the intermediate sentences tagged as General/Outcome."""

    input_df['General'] = 0
    input_df['Outcome'] = 0

    general_terms_str_all = "please|concourse|internet explorer|privacy terms|enable cookies|your browser|america/|newer browser|textbook|withdraw|grade|grading|responsible for|absences|lecture|instructors|nonattendance|students responsibility|responsibility for|attendance|absenteeism|intellidemia|dropped|faculty|final evaluation|final examination|permission|ask for help|during exams|during the exam|campus life|acknowledgment|ones own work|university-affiliated|federal, state and local|ensure integrity|avoid plagiarism|plagiarism|actively participate|devote sufficient time|assignments|code of conduct|student handbook|co-curricular activities|advising sessions|college studies|assignment sheets|tutorial sessions|personal obligation|personal integrity|staff|advisor or designee|withdrawal|accommodation|rehabilitation act|responsibility of the student|academic honesty|institutional policies|academic policies|your feedback|online classroom|to complete an online evaluation|kept confidential|classroom instruction|academic dishonesty|other courses|at all times|170.40 170.41 170.42|thousand oaks, ca: sage|our motto is|how it is met &|subject to change|below 67 f|page page|final exam|community college|page 1 of 1|page 2 of 2|page 3 of 3|page 4 of 4|page 5 of 5|page 6 of 6|page 7 of 7|page 8 of 8|page 9 of 9|page 10 of 10|what is it?|90 100% a 80 89% b 70 79% c 60 69% d < 60% f|academic integrity|c1. c2. c3. c4.|participate and communicate|find this article|title ix|login and click|off campus|log out / change|canvas|chain of command|code of ethics|apa style|will be cumulative|must be submitted by|make up exam|skip to body|this exam is|the exam is|thousand oaks|bring your own|isbn|the default theme|without the ads|email|teacher|make sure|each exam|telephone number|phone number|appointment|late|January|Jan |Jan. |February|Feb |Feb. |March|Mar |Mar. |April|Apr |Apr. |June|Jun |Jun. |July|Jul |Jul. |August|Aug |Aug. |September|Sep. |Sep |October|Oct |Oct. |November|Nov |Nov. |December|Dec. |Dec |Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|instructor|prerequisites|cannot|the assignment|make sure|Be sure|exam |Americans with Disabilities Act|During the test|fail the test|Purpose of Evaluation|FREQUENTLY ASKED QUESTIONS|FAQ|Edition|phone|your responsibility|honor code|discussion board|no break|individual responsibility|Violators|due date|submit the|per week|per session|Use browser|Press,|Publishers,|additional readings|the library|every class|http://|https://|double-spaced|Times New Roman|will submit|Blackboard|is worth|No food|No drink|Program Director|attachment schedule|Tentative|should be done|count toward|schedule class|make-up quizze|make up quizze|class hours|Prerequisite|noted below|Store or on-line|www.|will count as|to be submitted|create an account|Diversity Dialogue|scheduled course|Drop forms|division office|counseling office|registrar|semester/session|Your password|Classes Policy|family or medical emergency|following semester|Compliance with|Campus Policy|American Disabilities Act|order online|as described in the syllabus|accessibility resources|Violations of copyright|University policies|University policy|enrolled in|PowerPoint Slide|See Assignment|Rubric|correct response|questions correctly|study guide|The PowerPoint|must submit|should submit|Title V|if needed|Courses Requiring a|University of |Yes: No:|OpenCourseWare|Put |midterm|Final Assignment|Find material|Quiz Assignment|points|Department Chair|Division Dean|enrollment or registration|Fieldwork|Affirmative Action|Go online|the goals are clearly stated as learning outcomes|enroll|Dropping the Course|Pass/No Pass|SCHEDULE DESCRIPTION|No Class|is available|meeting time|will not be provided|minute|Program Policies|American Counseling Association|Texas Counseling Association|HARASSMENT|Time:|Place:|Location:|Office:|No credit|e-mail|email|Satisfactory/Unsatisfactory|s/u|hour/week|hours/week|hours/semester|will be announced|class progress|academic responsibilities|not permitted|Credit Hours|The time is|information below|information above|the department|as scheduled|rights reserved|Department Policies|deadline|LIBRARY SUPPORT|user and password|username and password|Google Drive|Distance Learning|Back to Top|your own work|Learning Management System|is recommended|bound by confidentiality|missing a course|Course menu|Courses menu|Class meetings|If in doubt|serious offenders|office hour|own work|Last day|Each assignment|copyright|on time|miss a class|plagiarize|fail the course|are allowed|WebAdvisor|drop courses|drop the course|A-F|P/NP|hours per semester|Help@|Help menu|must be approved|College Policies|No texting|FOOD AND DRINK:|Last Edited on|graduates|Read Chapter|The final|next class meeting|scheduled final|finals week|written excuse|The hours|multiple choice|true/false|course website|be allowed|There will be|Prof. |Dr. "

    input_df.loc[input_df['syllabus_text'].str.contains(general_terms_str_all, case=False), 'General'] = 1

    outcome_terms_str_all = "enable|Identifying|Develop|knowledge|Demonstrate|Engage|Engaging|learn|Method|Analyze|Predict|include|design|theories|theory|the study of|the relationship|versus|Definition|The Dynamic|evaluate|solve|problem|utilize|Explain|Understand|Outcome|to draw|Describe|Assess|Create|Identify|Experience|strategies|strategy|Apply|objective|adapt|Modify|Interpret|Measuring|prepare|teach you|goal|cover|involving|To address|Introduction to|Introduce|concept|Working"

    input_df.loc[input_df['syllabus_text'].str.contains(outcome_terms_str_all, case=False), 'Outcome'] = 1

    if bow_intermediate_save:
        input_df.to_csv(os.path.join(intermediate_path, 'sentTaggedIntermediate_' + file_name + '.gzip'),
                        compression='gzip', index=False)

    input_df = input_df[(input_df['General'] == 0) & (input_df['Outcome'] == 1)].reset_index(drop=True)
    input_df = input_df.drop(['General', 'Outcome'], axis=1)

    return input_df


# Step 4: O*NET Skills Embedding
def embedd_skill_type(onet_path, skill_type, bert_model):
    """ Function to check if the skills (DWA/Task) embeddings (dimensions) exist or not. If not, create them.
    - Inputs:
    -- onet_path: O*NET data path.
    -- skill_type: Skill type to be used for mapping.
    -- bert_model: SBERT langauge model. """

    skills_with_embedding_path = os.path.join(onet_path, skill_type + '_' + bert_model + '.csv')
    skills_without_embedding_path = os.path.join(onet_path, skill_type + '.csv')

    if not os.path.isfile(skills_without_embedding_path):
        raise Exception("Please upload the ONET file in the folder.")

    if not os.path.isfile(skills_with_embedding_path):

        input_df = pd.read_csv(skills_without_embedding_path, encoding='utf-8')

        model = SentenceTransformer(bert_model)

        if skill_type == 'dwa':
            column_name = 'DWA Title'
        elif skill_type == 'task':
            column_name = 'Task'

        embeddings = model.encode(input_df[column_name], show_progress_bar=False)

        embeddings_space_df = pd.DataFrame(embeddings)

        final_df = pd.merge(input_df, embeddings_space_df, left_index=True, right_index=True)

        final_df.to_csv(skills_with_embedding_path, index=False)

    else:
        final_df = pd.read_csv(skills_with_embedding_path)

    return final_df


# Step 5: Embedding the sentences
def sentences_embedding(input_df, bert_model):
    """ Function for extracting the embeddings of a given syllabi CSV file
        Inputs:
        - input_df: Input df.
        - bert_model_name: SBERT model to be used for embedding.

        Output:
        - A dataframe file containing the sentences and the dimensions. """

    model = SentenceTransformer(bert_model)
    # print("Model loaded ...")

    embeddings = model.encode(input_df['syllabus_text'], show_progress_bar=False)
    # print("Embedding done ...")

    embeddings_space_df = pd.DataFrame(embeddings)

    final_df = pd.merge(input_df, embeddings_space_df, left_index=True, right_index=True)

    return final_df


# Step 6: Skills similarities
def skills_similarities(input_syllabi_df, input_skills_df, outputs_path, input_file_name,
                        save_relevant_sent_ids, n_skills, skill_type, bert_model, smilarities_dtype='float16'):
    """ Function for computing the similarities between Skills (DWAs/Tasks)
        and sentences and return the MAX similarity.
        - Inputs:
        -- input_syllabi_df: The syllabi daframe with their language model embedding dimenssions.
        -- input_skills_df: O*NET skills (DWA/Task) daframe with their language model embedding dimenssions.
        -- outputs_path: Output path to save the results.
        -- input_file_name: File name to be used for saving the results.
        -- save_relevant_sent_ids: If true, saves the sentence ids with the maximum score for
            each DWA/Task in a seprace file. The columns of the resulting CSV file correspond to
            the DWA/TAsk ids (input files in 'onet_path').
        -- n_skills: Number of skills according to DWA or Task.
        -- skill_type: O*NET Skills to be used as the refrence 'dwa' or 'task'.
        -- bert_model:SBERT language model
        -- smilarities_dtype: Similarity scores datatype. 'float16', 'float32', or 'float64'.

        - Outputs:
        -- Gzip containing the Final Scores: The naming convention ->
            input_file_name + '_FinalScores_' + skill_type +'_'+ bert_model +'.gzip'
        -- Gzip containing the Sentence IDs of the maximum Final Scores
            (Optional, will be genrated if save_relevant_sent_ids=True): The naming convention ->
            input_file_name + '_FinalSentIds_' + skill_type +'_'+ bert_model +'.gzip'
        *Note: Both outputs will be saved at 'outputs_path'. """

    skills_np = input_skills_df.iloc[:, 1:].to_numpy()

    input_syllabi_f_df = input_syllabi_df.drop(['syllabus_text'], axis=1)

    unique_ids = list(input_syllabi_f_df['syllabus_id'].unique())

    final_max_similarities = np.zeros([len(unique_ids), skills_np.shape[0]], dtype=smilarities_dtype)

    ids = np.zeros([len(unique_ids)], dtype='int64')

    sentene_ids = np.zeros([len(unique_ids), len(skills_np)], dtype='uint16')

    for syll_idx in range(len(unique_ids)):
        uid = unique_ids[syll_idx]
        this_syll_df = input_syllabi_f_df[input_syllabi_f_df['syllabus_id'] == uid].reset_index(drop=True)
        this_syll_np = this_syll_df.iloc[:, 2:].to_numpy()
        this_syll_similarities = cosine_similarity(this_syll_np, skills_np)
        max_similarities_np = np.max(this_syll_similarities, axis=0)

        argmax_indices = np.argmax(this_syll_similarities, axis=0)
        sentence_indices = this_syll_df.loc[argmax_indices, 'sent_id'].to_list()

        final_max_similarities[syll_idx, :] = max_similarities_np
        ids[syll_idx] = uid
        sentene_ids[syll_idx, :] = sentence_indices

    skills_cols_names = [i for i in range(n_skills)]

    scores_df = pd.DataFrame(columns=['syllabus_id'] + skills_cols_names, index=range(sentene_ids.shape[0]))
    scores_df['syllabus_id'] = ids
    scores_df.loc[:, 0:] = final_max_similarities

    final_save_path_scores = os.path.join(outputs_path,
                                          input_file_name + '_FinalScores_' + skill_type + '_' + bert_model + '.gzip')

    scores_df.to_csv(final_save_path_scores, index=False, compression='gzip')

    if save_relevant_sent_ids:
        rele_sentIds_df = pd.DataFrame(columns=['syllabus_id'] + skills_cols_names, index=range(sentene_ids.shape[0]))
        rele_sentIds_df['syllabus_id'] = ids
        rele_sentIds_df.loc[:, 0:] = sentene_ids

        final_save_path_sentIds = os.path.join(outputs_path,
                                               input_file_name + '_FinalSentIds_' + skill_type + '_' + bert_model + '.gzip')

        rele_sentIds_df.to_csv(final_save_path_sentIds, index=False, compression='gzip')

    ## uncomment if you want to have all results in npz file
    # final_save_path = os.path.join(outputs_path, input_file_name + '_FinalScores_' + skill_type +'_'+ bert_model)
    # np.savez_compressed(final_save_path,
    #                     similarities=final_max_similarities,
    #                     syll_ids=ids,
    #                     sent_ids=sentene_ids)

    # return ids, sentene_ids, final_max_similarities


def syllabus_skills_extractor_pipeline(inputs_path, f_name, comp_meth, intermediate_path,
                                       segmnt_intermediate_save, with_bow_cleaning, bow_intermediate_save,
                                       onet_path, skill_type, bert_model, outputs_path,
                                       save_relevant_sent_ids, smilarities_dtype):
    """ Function to extract the O*NET skills (DWA/Tasks) given CSV file containing some syllabi.
        - Inputs:
        -- inputs_path: Input folder path
        -- f_name: Input syllabus file name in CSV format
        -- comp_meth: Input CSV file compression method. E.g., 'gzip', ...
        -- intermediate_path: Path to save the intermediate results.
        -- segmnt_intermediate_save: If ture, saves the intermediate segmented sentences.
        -- with_bow_cleaning: If True, removes the general terms from the sentences ('bag_of_word_tagger' function).
        -- bow_intermediate_save: If ture, saves the intermediate sentences tagged as General/Outcome.
        -- onet_path: O*NET folder path
        -- skill_type: O*NET Skills to be used as the refrence 'dwa' or 'task'.
        -- bert_model: SBERT language model
        -- outputs_path: Output folder path
        -- save_relevant_sent_ids: If true, saves the sentence ids with the maximum score for
            each DWA/Task in a seprace file. The columns of the resulting CSV file correspond to
            the DWA/TAsk ids (input files in 'onet_path').
        -- smilarities_dtype: Similarity scores datatype. 'float16', 'float32', or 'float64'.

        - Outputs:
        -- Gzip containing the Final Scores: The naming convention ->
            input_file_name + '_FinalScores_' + skill_type +'_'+ bert_model +'.gzip'
        -- Gzip containing the Sentence IDs of the maximum Final Scores
            (Optional, will be genrated if save_relevant_sent_ids=True): The naming convention ->
            input_file_name + '_FinalSentIds_' + skill_type +'_'+ bert_model +'.gzip'
        *Note: Both outputs will be saved at 'outputs_path'."""

    # *** Step 1: Load data
    file_name = f_name.split('.')[0]
    syll_df_path = os.path.join(inputs_path, f_name)
    syll_df = pd.read_csv(syll_df_path, encoding='utf-8', compression=comp_meth)

    try:
        syll_df['syllabus_id'] = syll_df['syllabus_id'].astype('int')
    except:
        raise Exception("'sllabus_id' should contain only numbers.")

    # *** Step 2: Sentence Segmentation
    syll_df = stanza_syl_2_lines_array(input_df=syll_df,
                                       intermediate_path=intermediate_path,
                                       file_name=file_name,
                                       segmnt_intermediate_save=segmnt_intermediate_save)

    # *** Step 3 (Optional): General and Outcome sentences filtering
    if with_bow_cleaning:
        syll_df = bag_of_word_tagger(input_df=syll_df,
                                     intermediate_path=intermediate_path,
                                     file_name=file_name,
                                     bow_intermediate_save=bow_intermediate_save)

    # *** Step 4: O*NET Skills Embedding
    if skill_type == 'dwa':
        n_skills = 2070
    elif skill_type == 'task':
        n_skills = 18429
    else:
        raise Exception("Sorry, wrong skill_type. Please choose either 'dwa' or 'task'.")

    onet_dimension_df = embedd_skill_type(onet_path, skill_type, bert_model)

    # *** Step 5: Embedding the sentences
    syll_df = sentences_embedding(input_df=syll_df,
                                  bert_model=bert_model)

    # *** Step 6: Skills similarities
    skills_similarities(input_syllabi_df=syll_df,
                        input_skills_df=onet_dimension_df,
                        outputs_path=outputs_path,
                        input_file_name=file_name,
                        save_relevant_sent_ids=save_relevant_sent_ids,
                        n_skills=n_skills,
                        skill_type=skill_type,
                        bert_model=bert_model,
                        smilarities_dtype=smilarities_dtype)
