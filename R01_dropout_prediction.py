import numpy as np
import MySQLdb
from sys import argv


from feature_generation import R01_credibility_extract, R01_BBSIQ_extract, R01_demographic_extract, R01_trial_extract, \
    R01_mental_extract, R01_OASIS_extract, R01_RR_extract, R01_affect_extract, \
    feature_vector_r01_overlap_with_mindtrails, feature_vector_r01_overlap_with_templeton

from basic import svm_model_prediction, logistic_regression_model_prediction, random_forest_model_prediction, \
    svm_multi_task_prediction, trained_model_retrieve, prediction_result_save_to_csv_file


def R01_prediction(R01_database_host, port_number, username, password, db_name, participantId, model_save_directory,
                   prediction_result_save_filename):

    # connect the R01 database
    R01_db = MySQLdb.connect(host=R01_database_host, port=port_number, user=username, passwd=password, db=db_name,
                             charset='utf8')
    cursor = R01_db.cursor()

    # extract data of participantId from tables
    cursor.execute("SELECT * FROM affect WHERE 'participant_id'=%s" %participantId)
    result = cursor.fetchall()
    affect_dict = R01_affect_extract(result)

    cursor.execute("SELECT * FROM bbsiq WHERE 'participant_id'=%s" % participantId)
    result = cursor.fetchall()
    BBSIQ_dict = R01_BBSIQ_extract(result)

    cursor.execute("SELECT * FROM credibility WHERE 'participant_id'=%s" % participantId)
    result = cursor.fetchall()
    credibility_dict = R01_credibility_extract(result)

    cursor.execute("SELECT * FROM demographics WHERE 'participant_id'=%s" % participantId)
    result = cursor.fetchall()
    demographics_dict = R01_demographic_extract(result)

    cursor.execute("SELECT * FROM angular_training WHERE 'participant_id'=%s" % participantId)
    result = cursor.fetchall()
    trial_dict = R01_trial_extract(result)

    cursor.execute("SELECT * FROM mental_health_history WHERE 'participant_id'=%s" % participantId)
    result = cursor.fetchall()
    mental_dict = R01_mental_extract(result)

    cursor.execute("SELECT * FROM oa WHERE 'participant_id'=%s" % participantId)
    result = cursor.fetchall()
    OASIS_dict = R01_OASIS_extract(result)

    cursor.execute("SELECT * FROM rr WHERE 'participant_id'=%s" % participantId)
    result = cursor.fetchall()
    RR_dict = R01_RR_extract(result)


    questionnaire_list = {'preTest': ['demographics', 'mental_health_history', 'anxiety_identity', 'oa', 'anxiety_triggers', 'bbsiq', 'comorbid', 'wellness', 'mechanisms'],
                          'firstSession': ['affect', 'angular_training', 'cc', 'oa', 'return_intention']}
    timeOnPage_dict = {}
    for sessionId in questionnaire_list:
        timeOnPage_dict[sessionId] = {}
        for item in questionnaire_list[sessionId]:
            if item == 'affect':
                cursor.execute("SELECT 'tag', 'time_on_page' FROM %s WHERE 'participant_id'=%s AND session='%s'" % (
                item, participantId, sessionId))

                result = cursor.fetchall()
                for line in result:

                    tag = line[1]
                    timeOnPage_dict[sessionId][item + '_' + tag] = line[1]
            elif item == 'Training':
                time_elapsed_list = []
                cursor.execute("SELECT 'time_elapsed' FROM %s WHERE 'participant_id'=%s AND session='%s'" % (
                item, participantId, sessionId))

                result = cursor.fetchall()
                for line in result:
                    time_elapsed = line[0]
                    time_elapsed_list.append(time_elapsed)
                timeOnPage_dict[sessionId][item] = np.max(time_elapsed_list/1000)
            else:
                cursor.execute("SELECT 'time_on_page' FROM %s WHERE 'participant_id'=%s AND session='%s'" % (
                item, participantId, sessionId))

                result = cursor.fetchall()
                for line in result:
                    timeOnPage_dict[sessionId][item] = line[0]
    R01_db.close()


    # generate feature vector overlapping with mindtrails or templeton
    feature_vector_overlap_with_mindtrails = feature_vector_r01_overlap_with_mindtrails(RR_dict, BBSIQ_dict, OASIS_dict,
                                                                                        demographics_dict,
                                                                                        timeOnPage_dict,
                                                                                        [participantId])
    feature_vector_overlap_with_templeton = feature_vector_r01_overlap_with_templeton(credibility_dict, mental_dict,
                                                                                      affect_dict, trial_dict,
                                                                                      demographics_dict,
                                                                                      [participantId])


    # retrieve trained classification models
    svm_model_mindtrails = trained_model_retrieve(model_save_directory, 'mindtrails', '_SVM_training_parameter.txt')
    lr_model_mindtrails = trained_model_retrieve(model_save_directory, 'mindtrails',
                                                 '_logistic_regression_training_parameter.txt')
    rf_model_mindtrails = trained_model_retrieve(model_save_directory, 'mindtrails',
                                                 '_random_forest_training_parameter.txt')
    svm_multi_task_model_mindtrails = trained_model_retrieve(model_save_directory, 'mindtrails',
                                                             '_multi_svm_training_parameter.txt')

    svm_model_templeton = trained_model_retrieve(model_save_directory, 'templeton', '_SVM_training_parameter.txt')
    lr_model_templeton = trained_model_retrieve(model_save_directory, 'templeton',
                                                '_logistic_regression_training_parameter.txt')
    rf_model_templeton = trained_model_retrieve(model_save_directory, 'templeton',
                                                '_random_forest_training_parameter.txt')
    svm_multi_task_model_templeton = trained_model_retrieve(model_save_directory, 'templeton',
                                                            '_multi_svm_training_parameter.txt')

    # compute predicted values of participantId
    svm_mindtrails_prediction = svm_model_prediction(feature_vector_overlap_with_mindtrails, svm_model_mindtrails)
    lr_mindtrails_prediction = logistic_regression_model_prediction(feature_vector_overlap_with_mindtrails,
                                                                    lr_model_mindtrails)
    rf_mindtrails_prediction = random_forest_model_prediction(feature_vector_overlap_with_mindtrails,
                                                              rf_model_mindtrails)
    svm_multi_task_mindtrails_prediction = svm_multi_task_prediction(feature_vector_overlap_with_mindtrails,
                                                                     demographics_dict, participantId,
                                                                     svm_multi_task_model_mindtrails)

    svm_templeton_prediction = svm_model_prediction(feature_vector_overlap_with_templeton, svm_model_templeton)
    lr_templeton_prediction = logistic_regression_model_prediction(feature_vector_overlap_with_templeton,
                                                                   lr_model_templeton)
    rf_templeton_prediction = random_forest_model_prediction(feature_vector_overlap_with_templeton, rf_model_templeton)
    svm_multi_task_templeton_prediction = svm_multi_task_prediction(feature_vector_overlap_with_templeton,
                                                                    demographics_dict, participantId,
                                                                    svm_multi_task_model_templeton)

    prediction_value_list = []

    prediction_value_list.append(svm_mindtrails_prediction)
    prediction_value_list.append(lr_mindtrails_prediction)
    prediction_value_list.append(rf_mindtrails_prediction)
    prediction_value_list.append(svm_multi_task_mindtrails_prediction)

    prediction_value_list.append(svm_templeton_prediction)
    prediction_value_list.append(lr_templeton_prediction)
    prediction_value_list.append(rf_templeton_prediction)
    prediction_value_list.append(svm_multi_task_templeton_prediction)

    prediction_value_max = max(prediction_value_list)

    # store the predicted values to csv file
    prediction_result_save_to_csv_file(svm_mindtrails_prediction, lr_mindtrails_prediction, rf_mindtrails_prediction,
                                       svm_multi_task_mindtrails_prediction, svm_templeton_prediction,
                                       lr_templeton_prediction, rf_templeton_prediction,
                                       svm_multi_task_templeton_prediction, prediction_value_max, participantId,
                                       prediction_result_save_filename)


R01_database_host = argv[1]
port_number = eval(argv[2])
username = argv[3]
password =argv[4]
db_name = argv[5]
participantId = eval(argv[6])
model_save_directory = argv[7]
prediction_result_save_filename = argv[8]

R01_prediction(R01_database_host, port_number, username, password, db_name, participantId, model_save_directory, prediction_result_save_filename)