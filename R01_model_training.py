from sys import argv

from basic import classification_model_training_mindtrails, classification_model_training_templeton, svm_model_training, \
    logistic_regression_model_training, random_forest_model_training, svm_multi_task_training, trained_model_store


def model_training(R01_classification_model_dir, model_save_directory):
    '''
    training the classification model based on the input prediction_session_index, and the trained model will be stored
    in the model_save_directory.
    :param R01_classification_model_dir:
    :param model_save_directory:
    :return:
    '''

    platform_list = ['mindtrails', 'templeton']
    prediction_session_index = 2

    for platform in platform_list:
        if platform == 'mindtrails':
            feature_vector, truth_vector, participant_list, demographic_dict = classification_model_training_mindtrails(
                platform, prediction_session_index, R01_classification_model_dir)

        elif platform == 'templeton':
            feature_vector, truth_vector, participant_list, demographic_dict = classification_model_training_templeton(
                platform, prediction_session_index, R01_classification_model_dir)

        # model training
        svm_model = svm_model_training(feature_vector, truth_vector)
        lr_model = logistic_regression_model_training(feature_vector, truth_vector)
        rf_model = random_forest_model_training(feature_vector, truth_vector)
        multi_svm_model = svm_multi_task_training(feature_vector, truth_vector, participant_list, demographic_dict)

        # save the parameters of the retrained models
        trained_model_store(svm_model, model_save_directory, platform, '_SVM_training_parameter.txt')
        trained_model_store(lr_model, model_save_directory, platform, '_logistic_regression_training_parameter.txt')
        trained_model_store(rf_model, model_save_directory, platform, '_random_forest_training_parameter.txt')
        trained_model_store(multi_svm_model, model_save_directory, platform, '_multi_svm_training_parameter.txt')


R01_classification_model_dir = argv[1]
model_save_directory = argv[2]
model_training(R01_classification_model_dir, model_save_directory)