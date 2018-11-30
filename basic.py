import os
import copy
import MySQLdb
import time
import csv
import numpy as np
import pandas as pd


from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier

from feature_generation import calibrate, file_read_and_feature_extract, mindtrails_feature_vector_generation, \
    templeton_feature_vector_generation, R01_affect_extract, R01_RR_extract, R01_OASIS_extract, R01_mental_extract, \
    R01_trial_extract, R01_demographic_extract, R01_credibility_extract, R01_BBSIQ_extract, saveDictFile, loadDictFile



def trained_model_store(model, model_save_directory, platform, filename):
    if os.path.isdir(model_save_directory):
        saveDictFile(model, model_save_directory + platform + filename)
    else:
        os.mkdir(model_save_directory)


def trained_model_retrieve(model_save_directory, platform, filename):
    model = loadDictFile(model_save_directory + platform + filename)
    return model


def prediction_result_save_to_csv_file(svm_mindtrails_prediction, lr_mindtrails_prediction, rf_mindtrails_prediction,
                                       svm_multi_task_mindtrails_prediction, svm_templeton_prediction,
                                       lr_templeton_prediction, rf_templeton_prediction,
                                       svm_multi_task_templeton_prediction, prediction_value_max, participantId,
                                       prediction_result_save_filename):

    if os.path.isdir(prediction_result_save_filename) is False:
        os.mkdir(prediction_result_save_filename)
        with open(prediction_result_save_filename, 'a+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['participantId', 'prediction with features from mindtrails_SVM',
                            'prediction with features from mindtrails_Logistic Regression',
                            'prediction with features from mindtrails_Random Forest',
                            'prediction with features from mindtrails_SMV multi-task',
                            'prediction with features from templeton_SVM',
                            'prediction with features from templeton_Logistic Regression',
                            'prediction with features from templeton_Random Forest',
                            'prediction with features from templeton_SMV multi-task',
                            'maximum prediction value'])
        f.close()

    with open(prediction_result_save_filename, 'a+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow([participantId, svm_mindtrails_prediction, lr_mindtrails_prediction, rf_mindtrails_prediction,
                        svm_multi_task_mindtrails_prediction, svm_templeton_prediction, lr_templeton_prediction,
                        rf_templeton_prediction, svm_multi_task_templeton_prediction, prediction_value_max])

def svm_model_training(feature_vector, truth_list):
    svm_model = svm.LinearSVC(C=1, tol=1e-3)
    svm_model.fit(feature_vector, truth_list)
    return svm_model

def logistic_regression_model_training(feature_vector, truth_list):
    lr_model = linear_model.LogisticRegression()
    lr_model.fit(feature_vector, truth_list)
    return lr_model

def random_forest_model_training(feature_vector, truth_list):
    rf_model = RandomForestClassifier(
        n_estimators=10, criterion="gini", max_features="auto", max_depth=2, min_samples_split=2,
        min_samples_leaf=1, random_state=0, bootstrap=True, min_weight_fraction_leaf=0.0,
        n_jobs=1, oob_score=False, verbose=0, warm_start=False
    )
    rf_model.fit(feature_vector, truth_list)
    return rf_model

def svm_multi_task_training(feature_vector, truth_list, participant_list, demo_info):
    model_coef_interpreter = []
    d = len(feature_vector[0])  # feature dimension for each participant
    group_feature_vector = []
    group_truth_vector = []
    for i in range(len(feature_vector)):
        participant_id = participant_list[i]

        single_participant_feature_vector = np.zeros(d * 12)
        if participant_id in demo_info:
            gender = demo_info[participant_id]['gender']
            if gender == 'Male':
                single_participant_feature_vector[0:d] = feature_vector[i]
            elif gender == 'Female':
                single_participant_feature_vector[d: 2 * d] = feature_vector[i]
            else:
                gender = 'gender_other'
                single_participant_feature_vector[2 * d: 3 * d] = feature_vector[i]

            education = demo_info[participant_id]['education']
            if education in ['Elementary School', 'High School Graduate', 'Some College', "Associate's Degree"]:
                education = 'before Bachelor'
                single_participant_feature_vector[3 * d: 4 * d] = feature_vector[i]
            elif education in ["Bachelor's Degree", "Master's Degree", 'Ph.D.']:
                education = 'after Bachelor'
                single_participant_feature_vector[4 * d: 5 * d] = feature_vector[i]
            else:
                education = 'education_other'
                single_participant_feature_vector[5 * d: 6 * d] = feature_vector[i]

            age = 2018 - int(demo_info[participant_id]['birth_year'])
            if age < 20:
                age = '<20'
                single_participant_feature_vector[6 * d: 7 * d] = feature_vector[i]
            elif (age >= 20) and (age < 30):
                age = '20-30'
                single_participant_feature_vector[7 * d: 8 * d] = feature_vector[i]
            elif (age >= 30) and (age < 40):
                age = '30-40'
                single_participant_feature_vector[8 * d: 9 * d] = feature_vector[i]
            elif (age >= 40) and (age < 50):
                age = '40-50'
                single_participant_feature_vector[9 * d: 10 * d] = feature_vector[i]
            elif (age >= 50):
                age = '>50'
                single_participant_feature_vector[10 * d: 11 * d] = feature_vector[i]
            else:
                age = 'age_other'
                single_participant_feature_vector[11 * d:] = feature_vector[i]

        else:
            gender = 'gender_other'
            education = 'education_other'
            age = 'age_other'

            single_participant_feature_vector[2 * d: 3 * d] = feature_vector[i]
            single_participant_feature_vector[5 * d: 6 * d] = feature_vector[i]
            single_participant_feature_vector[11 * d:] = feature_vector[i]

        group_feature_vector.append(single_participant_feature_vector)
        group_truth_vector.append(truth_list[i])

    multi_svm_model = svm.LinearSVC(C=1)
    multi_svm_model.fit(group_feature_vector, truth_list)

    return multi_svm_model

def svm_model_prediction(testing_feature_vector, svm_model):
    prediction_value = svm_model.predict(testing_feature_vector)
    return prediction_value

def logistic_regression_model_prediction(testing_feature_vector, lr_model):
    prediction_value = lr_model.predict(testing_feature_vector)
    return prediction_value

def random_forest_model_prediction(testing_feature_vector, rf_model):
    prediction_value = rf_model.predict(testing_feature_vector)
    return prediction_value

def svm_multi_task_prediction(testing_feature_vector, demo_info, participant_id, multi_svm_model):
    coef_old = copy.deepcopy(multi_svm_model.coef_)[0]

    multi_svm_model_new = copy.deepcopy(multi_svm_model)
    d = len(testing_feature_vector)
    if participant_id in demo_info:
        gender = demo_info[participant_id]['gender']
        if gender == 'Male':
            gender_coef = coef_old[0: d]
        elif gender == 'Female':
            gender_coef = coef_old[d: 2 * d]
        else:
            gender_coef = coef_old[2 * d: 3 * d]

        education = demo_info[participant_id]['education']
        if education in ['Elementary School', 'High School Graduate', 'Some College', "Associate's Degree"]:
            education_coef = coef_old[3 * d: 4 * d]
        elif education in ["Bachelor's Degree", "Master's Degree", 'Ph.D.']:
            education_coef = coef_old[4 * d: 5 * d]
        else:
            education_coef = coef_old[5 * d: 6 * d]

        age = 2018 - int(demo_info[participant_id]['birth_year'])
        if age < 20:
            age_coef = coef_old[6 * d: 7 * d]
        elif (age >= 20) and (age < 30):
            age_coef = coef_old[7 * d: 8 * d]
        elif (age >= 30) and (age < 40):
            age_coef = coef_old[8 * d: 9 * d]
        elif (age >= 40) and (age < 50):
            age_coef = coef_old[9 * d: 10 * d]
        elif (age >= 50):
            age_coef = coef_old[10 * d: 11 * d]
        else:
            age_coef = coef_old[11 * d:]
    else:
        gender_coef = coef_old[2 * d: 3 * d]
        education_coef = coef_old[5 * d: 6 * d]
        age_coef = coef_old[11 * d:]

    coef_new = np.array([gender_coef + age_coef + education_coef])
    multi_svm_model_new.coef_ = coef_new
    prediction_value = multi_svm_model_new.predict(testing_feature_vector)

    return prediction_value

def classification_model_training_mindtrails(platform, prediction_session_index, R01_classification_model_dir):
    session_list = ['PRE', 'SESSION1', 'SESSION2', 'SESSION3', 'SESSION4', 'SESSION5', 'SESSION6', 'SESSION7',
                    'SESSION8']

    demographic_dict, QOL_dict, OASIS_dict, RR_dict, BBSIQ_dict, DASS21_AS_dict, DASS21_DS_dict, trial_dict, \
    dwell_time_dict, session_completion_dict, dropout_label, control_normal_dict = file_read_and_feature_extract(
        platform, R01_classification_model_dir)

    prediction_session = session_list[prediction_session_index]
    training_set_session = session_list[0: prediction_session_index]
    participant_list = []
    for e in dropout_label:
        if (int(e) > 419) or (int(e) < 20):
            if (e in control_normal_dict['training']) and (
                    dropout_label[e][session_list[prediction_session_index - 1]] == '0'):
                participant_list.append(e)

    feature_vector, truth_vector, feature_item_list = mindtrails_feature_vector_generation(training_set_session,
                                                                                           prediction_session,
                                                                                           participant_list,
                                                                                           demographic_dict,
                                                                                           QOL_dict, OASIS_dict,
                                                                                           RR_dict, BBSIQ_dict,
                                                                                           DASS21_AS_dict,
                                                                                           DASS21_DS_dict,
                                                                                           trial_dict,
                                                                                           dwell_time_dict,
                                                                                           dropout_label)
    print 'mindtrails number of participants for predict session ', prediction_session, len(feature_vector), \
        'feature dimension', len(feature_vector[0]), len(feature_item_list)

    return feature_vector, truth_vector, participant_list, demographic_dict

def classification_model_training_templeton(platform, prediction_session_index, R01_classification_model_dir):
    session_list = ['preTest', 'firstSession', 'secondSession', 'thridSession', 'fourthSession']

    demographic_dict, affect_dict, credibility_dict, mental_dict, whatibelieve_dict, relatability_dict, \
    expectancy_dict, phq4_dict, trial_dict, session_completion_dict, dropout_label = file_read_and_feature_extract(
        platform, R01_classification_model_dir)

    prediction_session = session_list[prediction_session_index]
    training_set_session = session_list[0: prediction_session_index]
    participant_list = []
    for e in dropout_label:
        if dropout_label[e][session_list[prediction_session_index - 1]] == '0':
            participant_list.append(e)

    feature_vector, truth_vector, feature_item_list = templeton_feature_vector_generation(training_set_session,
                                                                                          prediction_session,
                                                                                          participant_list,
                                                                                          demographic_dict,
                                                                                          affect_dict,
                                                                                          credibility_dict,
                                                                                          mental_dict,
                                                                                          whatibelieve_dict,
                                                                                          relatability_dict,
                                                                                          expectancy_dict,
                                                                                          phq4_dict, trial_dict,
                                                                                          dropout_label)


    print 'templeton number of participants for predict session ', prediction_session, len(feature_vector), \
        'feature dimension', len(feature_vector[0]), len(feature_item_list)

    return feature_vector, truth_vector, participant_list, demographic_dict

def R01_table_extract(R01_database_host, port_number, username, password, db_name, participant_id):
    R01_db = MySQLdb.connect(host=R01_database_host, port=port_number, user=username, passwd=password, db=db_name,
                             charset='utf8')
    cursor = R01_db.cursor()

    participant_info_dict = {}

    # information from demographics
    sql = "SELECT * FROM demographics WHERE participant_id = '%d' " % participant_id
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            session = row[2]
            if session == 'SessionPre':
                time_on_page = row[4]
                birth_year = int(row[5])
                country = row[6]
                device = row[7]
                education = row[8]
                employmentStat = row[9]
                ethnicity = row[10]
                gender = row[11]
                income = row[12]
                maritalStat = row[13]
                ptp_reason = row[14]
                ptp_reason_other = row[15]
                race = row[16]

                participant_info_dict['demographics'] = {
                    'timeonpage': time_on_page,
                    'birth_year': birth_year,
                    'country': country,
                    'device': device,
                    'education': education,
                    'employmentStat': employmentStat,
                    'ethnicity': ethnicity,
                    'gender': gender,
                    'income': income,
                    'maritalStat': maritalStat,
                    'ptp_reason': ptp_reason,
                    'ptp_reason_other': ptp_reason_other,
                    'race': race
                }
    except:
        print "Error: unable to fecth data from demographics table"

    # information from affect table
    sql = "SELECT * FROM affect WHERE participant_id = '%d' " % participant_id
    participant_info_dict['affect'] = {}
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            session = row[2]
            if session == 'Session1':
                tag = row[3]
                time_on_page = row[4]
                neg_feelings = calibrate(555, int(row[5]))
                pos_feelings = calibrate(555, int(row[6]))

                participant_info_dict['affect'][tag] = {
                    'timeonpage': time_on_page,
                    'neg_feelings': neg_feelings,
                    'pos_feelings': pos_feelings
                }
    except:
        print "Error: unable to fecth data from affect table"


    # information from BBSIQ
    sql = "SELECT * FROM bbsiq WHERE participant_id = '%d' " % participant_id
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            session = row[2]
            if session == 'Session1':
                time_on_page = row[4]
                breath_flu = calibrate(555, int(row[5]))
                breath_physically = calibrate(555, int(row[6]))
                breath_suffocate = calibrate(555, int(row[7]))
                chest_heart = calibrate(555, int(row[8]))
                chest_indigestion = calibrate(555, int(row[9]))
                chest_sore = calibrate(555, int(row[10]))
                confused_cold = calibrate(555, int(row[11]))
                confused_outofmind = calibrate(555, int(row[12]))
                confused_work = calibrate(555, int(row[13]))
                dizzy_ate = calibrate(555, int(row[14]))
                dizzy_ill = calibrate(555, int(row[15]))
                dizzy_overtired = calibrate(555, int(row[16]))
                friend_helpful = calibrate(555, int(row[17]))
                friend_incompetent = calibrate(555, int(row[18]))
                friend_moreoften = calibrate(555, int(row[19]))
                heart_active = calibrate(555, int(row[20]))
                heart_excited = calibrate(555, int(row[21]))
                heart_wrong = calibrate(555, int(row[22]))
                jolt_burglar = calibrate(555, int(row[23]))
                jolt_dream = calibrate(555, int(row[24]))
                jolt_wind = calibrate(555, int(row[25]))
                lightheaded_eat = calibrate(555, int(row[26]))
                lightheaded_faint = calibrate(555, int(row[27]))
                lightheaded_sleep = calibrate(555, int(row[28]))
                party_boring = calibrate(555, int(row[29]))
                party_hear = calibrate(555, int(row[30]))
                party_preoccupied = calibrate(555, int(row[31]))
                shop_bored = calibrate(555, int(row[32]))
                hop_concentrating = calibrate(555, int(row[33]))
                shop_irritating = calibrate(555, int(row[34]))
                smoke_cig = calibrate(555, int(row[35]))
                smoke_food = calibrate(555, int(row[36]))
                smoke_house = calibrate(555, int(row[37]))
                urgent_bill = calibrate(555, int(row[38]))
                urgent_died = calibrate(555, int(row[39]))
                urgent_junk = calibrate(555, int(row[40]))
                vision_glasses = calibrate(555, int(row[41]))
                vision_illness = calibrate(555, int(row[42]))
                vision_strained = calibrate(555, int(row[43]))
                visitors_bored = calibrate(555, int(row[44]))
                visitors_engagement = calibrate(555, int(row[45]))
                visitors_outstay = calibrate(555, int(row[46]))

                participant_info_dict['BBSIQ'] = {
                    'timeonpage': time_on_page,
                    'breath_flu': breath_flu, 'breath_physically': breath_physically, 'breath_suffocate': breath_suffocate,
                    'chest_heart': chest_heart, 'chest_indigestion': chest_indigestion, 'chest_sore': chest_sore,
                    'confused_cold': confused_cold, 'confused_outofmind': confused_outofmind, 'confused_work': confused_work,
                    'dizzy_ate': dizzy_ate, 'dizzy_ill': dizzy_ill, 'dizzy_overtired': dizzy_overtired,
                    'friend_helpful': friend_helpful, 'friend_incompetent': friend_incompetent,
                    'friend_moreoften': friend_moreoften,
                    'heart_active': heart_active, 'heart_excited': heart_excited, 'heart_wrong': heart_wrong,
                    'jolt_burglar': jolt_burglar, 'jolt_dream': jolt_dream, 'jolt_wind': jolt_wind,
                    'lightheaded_eat': lightheaded_eat, 'lightheaded_faint': lightheaded_faint,
                    'lightheaded_sleep': lightheaded_sleep,
                    'party_boring': party_boring, 'party_hear': party_hear, 'party_preoccupied': party_preoccupied,
                    'shop_bored': shop_bored, 'hop_concentrating': hop_concentrating, 'shop_irritating': shop_irritating,
                    'smoke_cig': smoke_cig, 'smoke_food': smoke_food, 'smoke_house': smoke_house,
                    'urgent_bill': urgent_bill, 'urgent_died': urgent_died, 'urgent_junk': urgent_junk,
                    'vision_glasses': vision_glasses, 'vision_illness': vision_illness, 'vision_strained': vision_strained,
                    'visitors_bored': visitors_bored, 'visitors_engagement': visitors_engagement,
                    'visitors_outstay': visitors_outstay
                }
    except:
        print "Error: unable to fecth data from bbsiq table"


    # information from credibility
    sql = "SELECT * FROM bbsiq WHERE participant_id = '%d' " % participant_id
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            session = row[2]
            if session == 'SessionPre':
                time_on_page = row[4]
                participant_info_dict['credibility'] = {
                    'timeonpage': time_on_page
                }
    except:
        print "Error: unable to fecth data from credibility table"

    # information from js_psych_trial table
    sql = "SELECT * FROM js_psych_trial WHERE participant_id = '%d' " % participant_id
    participant_info_dict['affect'] = {}
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            session = row[2]
            if session == 'Session1':
                time_elapsed = row[13]
                condition = row[2]
                first_try_correct = row[3]
                device = row[5]

                participant_info_dict['affect'][tag] = {
                    'time_elapsed': [time_elapsed],
                    'device': device,
                    'first_try_correct': [first_try_correct]
                }
    except:
        print "Error: unable to fecth data from js_psych_trial table"
    R01_db.close()

def R01_model_adaptation_participant_list(R01_database_host, port_number, username, password, db_name):
    # connect the R01 database
    R01_db = MySQLdb.connect(host=R01_database_host, port=port_number, user=username, passwd=password, db=db_name,
                             charset='utf8')
    cursor = R01_db.cursor()

    # extract the participant list from R01 to do classification model adaptation
    cursor.execute("SELECT 'date', 'participant_id' FROM 'return_intention' WHERE 'session'=%s" % 'firstSession')
    result = cursor.fetchall()
    participant_id_list = []
    first_session_finish_date_dict = {}
    for line in result:
        date = line[0]
        participant_id = int(line[1])
        if participant_id not in participant_id_list:
            participant_id_list.append(participant_id)
            first_session_finish_date_dict[participant_id] = date

    cursor.execute("SELECT 'participant_id' FROM 'return_intention' WHERE 'session'=%s" % 'secondSession')
    result = cursor.fetchall()
    dropout_dict = {}
    for line in result:
        participant_id = int(line[0])
        if participant_id not in dropout_dict.keys():
            dropout_dict[participant_id] = 0

    for participant_id in participant_id_list:
        if participant_id not in dropout_dict:
            first_session_finish_date = first_session_finish_date_dict[participant_id]
            first_session_finish_date_ts = time.mktime(
                time.strptime(str(pd.to_datetime(first_session_finish_date)), '%Y-%m-%d %H:%M:%S'))
            current_data_ts = time.mktime(time.strptime(str(date.today())), '%Y-%m-%d %H:%M:%S')
            if (current_data_ts - first_session_finish_date_ts) / (3600 * 24) > 14:
                dropout_dict[participant_id] = 1

    participant_id_list = []
    dropout_label_list = []
    for participant_id in dropout_dict:
        participant_id_list.append(participant_id)
        dropout_label_list.append(dropout_dict[participant_id])

    R01_db.close()

    return participant_id_list, dropout_label_list

def R01_model_adaptation_feature_generation(R01_database_host, port_number, username, password, db_name,
                                            participant_id_list):
    # connect the R01 database
    R01_db = MySQLdb.connect(host=R01_database_host, port=port_number, user=username, passwd=password, db=db_name,
                             charset='utf8')
    cursor = R01_db.cursor()

    # extract data of participantId from tables
    cursor.execute("SELECT * FROM 'affect'")
    result = cursor.fetchall()
    affect_dict = R01_affect_extract(result)

    cursor.execute("SELECT * FROM 'bbsiq'")
    result = cursor.fetchall()
    BBSIQ_dict = R01_BBSIQ_extract(result)

    cursor.execute("SELECT * FROM 'credibility'")
    result = cursor.fetchall()
    credibility_dict = R01_credibility_extract(result)

    cursor.execute("SELECT * FROM 'demographics'")
    result = cursor.fetchall()
    demographic_r01_dict = R01_demographic_extract(result)

    cursor.execute("SELECT * FROM 'Training'")
    result = cursor.fetchall()
    trial_dict = R01_trial_extract(result)

    cursor.execute("SELECT * FROM 'mental_health_history' ")
    result = cursor.fetchall()
    mental_dict = R01_mental_extract(result)

    cursor.execute("SELECT * FROM 'oa'")
    result = cursor.fetchall()
    OASIS_dict = R01_OASIS_extract(result)

    cursor.execute("SELECT * FROM 'rr'")
    result = cursor.fetchall()
    RR_dict = R01_RR_extract(result)

    questionnaire_list = {
        'preTest': ['demographics', 'mental_health_history', 'anxiety_identity', 'oa', 'anxiety_triggers', 'bbsiq',
                    'comorbid', 'wellness', 'mechanisms'],
        'firstSession': ['affect', 'Training', 'cc', 'oa', 'return_intention']}
    timeOnPage_dict = {}
    for sessionId in questionnaire_list:
        timeOnPage_dict[sessionId] = {}
        for item in questionnaire_list[sessionId]:
            if item == 'affect':
                cursor.execute("SELECT tag, time_on_page,participant_id FROM item")
                result = cursor.fetchall()
                for line in result:
                    tag = line[1]
                    timeOnPage = int(line[1])
                    participant_id = int(line[2])
                    if (participant_id in participant_id_list) and (participant_id in timeOnPage_dict):
                        if sessionId in timeOnPage_dict[participant_id]:
                            timeOnPage_dict[participant_id][sessionId][item + '_' + tag] = timeOnPage
                        else:
                            timeOnPage_dict[participant_id][sessionId] = {
                                item + '_' + tag: timeOnPage
                            }
                    else:
                        timeOnPage_dict[participant_id] = {
                            sessionId: {
                                item + '_' + tag: timeOnPage
                            }
                        }

            elif item == 'Training':
                cursor.execute("SELECT time_elapsed,participant_id FROM item")
                result = cursor.fetchall()

                time_elapsed_dict = {}
                for line in result:
                    time_elapsed = int(line[0])
                    participant_id = int(line[1])
                    if (participant_id in participant_id_list) and (participant_id in time_elapsed_dict.keys()):
                        time_elapsed_dict[participant_id].append(time_elapsed)
                    else:
                        time_elapsed_dict[participant_id] = [time_elapsed]
                for participant_id in time_elapsed_dict:
                    timeOnPage_dict[sessionId][item] = np.max(time_elapsed_dict[participant_id] / 1000)

            else:
                cursor.execute("SELECT time_on_page,participant_id FROM item")
                result = cursor.fetchall()
                for line in result:
                    timeOnPage = int(line[0])
                    participant_id = int(line[1])

                    if (participant_id in participant_id_list) and (participant_id in timeOnPage_dict):
                        if sessionId in timeOnPage_dict[participant_id]:
                            timeOnPage_dict[participant_id][sessionId][item] = timeOnPage
                        else:
                            timeOnPage_dict[participant_id][sessionId] = {
                                item: timeOnPage
                            }
                    else:
                        timeOnPage_dict[participant_id] = {
                            sessionId: {
                                item: timeOnPage
                            }
                        }
    R01_db.close()

    return RR_dict, BBSIQ_dict, OASIS_dict, demographic_r01_dict, timeOnPage_dict, credibility_dict, mental_dict, \
           affect_dict, trial_dict