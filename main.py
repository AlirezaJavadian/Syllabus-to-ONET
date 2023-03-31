from utils import *
from settings import *

if __name__ == '__main__':

    for f_name in file_names_list:
        print(" Start: ", f_name)
        syllabus_skills_extractor_pipeline(inputs_path, f_name, comp_meth, intermediate_path,
                                           segmnt_intermediate_save, with_bow_cleaning, bow_intermediate_save,
                                           onet_path, skill_type, bert_model, outputs_path,
                                           save_relevant_sent_ids, smilarities_dtype)
        print(" End: ", f_name)
