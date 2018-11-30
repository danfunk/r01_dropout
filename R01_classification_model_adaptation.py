from sys import argv

from feature_generation import feature_vector_r01_overlap_with_templeton, feature_vector_r01_overlap_with_mindtrails
from basic import R01_model_adaptation_participant_list, R01_model_adaptation_feature_generation, \
    classification_model_training_templeton, classification_model_training_mindtrails, svm_model_training, \
    logistic_regression_model_training, random_forest_model_training, svm_multi_task_training, trained_model_store

def model_adaption(R01_database_host, port_number, username, password, db_name, model_save_directory):
    # extract participant_id_list and corresponding dropout labels for classification model adaptation
    participant_id_list, dropout_label_list = R01_model_adaptation_participant_list(R01_database_host, port_number,
                                                                                    username, password, db_name)

    # generate feature vectors from R01 for model adaptation
    RR_dict, BBSIQ_dict, OASIS_dict, demographic_r01_dict, timeOnPage_dict, credibility_dict, mental_dict, affect_dict, \
    trial_dict = R01_model_adaptation_feature_generation(R01_database_host, port_number, username, password, db_name,
                                                         participant_id_list)


    # generate feature vectors from mindtrails and templeton for model adaptation
    platform_list = ['mindtrails', 'templeton']
    for platform in platform_list:
        if platform == 'mindtrails':
            feature_vector_overlap_with_mindtrails = feature_vector_r01_overlap_with_mindtrails(RR_dict, BBSIQ_dict,
                                                                                                OASIS_dict,
                                                                                                demographic_r01_dict,
                                                                                                timeOnPage_dict,
                                                                                                participant_id_list)
            feature_vector_mindtrails, truth_vector_mindtrails, participant_list_mindtrails, demographic_mindtrails_dict = \
                classification_model_training_mindtrails(platform, prediction_session_index=2)

            feature_vector = feature_vector_mindtrails + feature_vector_overlap_with_mindtrails
            truth_vector = dropout_label_list + truth_vector_mindtrails
            participant_list = participant_id_list + participant_list_mindtrails
            demographic_all_dict = demographic_r01_dict + demographic_mindtrails_dict

        elif platform == 'templeton':
            feature_vector_overlap_with_templeton = feature_vector_r01_overlap_with_templeton(credibility_dict,
                                                                                              mental_dict,
                                                                                              affect_dict, trial_dict,
                                                                                              demographic_r01_dict,
                                                                                              participant_id_list)
            feature_vector_templeton, truth_vector_templeton, participant_list_templeton, demographic_templeton_dict = \
                classification_model_training_templeton(platform, prediction_session_index=2)

            feature_vector = feature_vector_templeton + feature_vector_overlap_with_templeton
            truth_vector = dropout_label_list + truth_vector_templeton
            participant_list = participant_id_list + participant_list_templeton
            demographic_all_dict = demographic_r01_dict + demographic_templeton_dict

        # model retraining
        svm_model = svm_model_training(feature_vector, truth_vector)
        lr_model = logistic_regression_model_training(feature_vector, truth_vector)
        rf_model = random_forest_model_training(feature_vector, truth_vector)
        multi_svm_model = svm_multi_task_training(feature_vector, truth_vector, participant_list, demographic_all_dict)

        # save the parameters of the retrained models
        trained_model_store(svm_model, model_save_directory, platform, '_SVM_training_parameter.txt')
        trained_model_store(lr_model, model_save_directory, platform, '_logistic_regression_training_parameter.txt')
        trained_model_store(rf_model, model_save_directory, platform, '_random_forest_training_parameter.txt')
        trained_model_store(multi_svm_model, model_save_directory, platform, '_multi_svm_training_parameter.txt')


R01_database_host = argv[1]
port_number = argv[2]
username = argv[3]
password =argv[4]
db_name = argv[5]
model_save_directory = argv[6]

model_adaption(R01_database_host, port_number, username, password, db_name, model_save_directory)