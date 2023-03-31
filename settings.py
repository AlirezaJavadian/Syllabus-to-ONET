## *** Paths

# Input folder path
# inputs_path = './inputs'
inputs_path = '/ihome/mrfrank/alj112/Projects/Notebooks/syllabus_2_ONET/inputs'

# Input syllabi names files in CSV format.
# Each record should contains the following two columns "syllabus_id" (integer), "syllabus_text" (string).
file_names_list = ['Test_Data.csv']

# Input CSV file compression method. E.g., 'gzip', ...
comp_meth = None

# O*NET folder path
# Note: Please do not delete the content of this folder
# onet_path = './ONET_Data'
onet_path = '/ihome/mrfrank/alj112/Projects/Notebooks/syllabus_2_ONET/ONET_Data'

# Output folder path
# outputs_path ='./outputs'
outputs_path = '/ihome/mrfrank/alj112/Projects/Notebooks/syllabus_2_ONET/outputs'
# Intermediate results
# intermediate_path ='./outputs/intermediate_results'
intermediate_path = '/ihome/mrfrank/alj112/Projects/Notebooks/syllabus_2_ONET/outputs/intermediate_results'

## *** Other Settings

# O*NET Skills to be used as the refrence 'dwa' or 'task'
skill_type = 'dwa'

# SBERT language model
bert_model = 'all-mpnet-base-v2'

# If True, removes the general terms from the sentences ('bag_of_word_tagger' function)
with_bow_cleaning = True

# If ture, saves the intermediate segmented sentences.
segmnt_intermediate_save = True

# If ture, saves the intermediate sentences tagged as General/Outcome.
bow_intermediate_save = True

# If true, saves the sentence ids with the maximum score for each DWA/Task in a seprace file.
# The columns of the resulting CSV file correspond to the DWA/TAsk ids (input files in 'onet_path')
save_relevant_sent_ids = True

# Similarity scores datatype. 'float16', 'float32', or 'float64'
smilarities_dtype='float16'
