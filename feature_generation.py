from __future__ import division
import os
import csv
import time
import numpy as np
import pandas as pd
import cPickle as pickle


def csvAdd(filename,ss):
    with open(filename,'a+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(ss)
    f.close()

def csvRead(filename):
    ff = csv.reader(open(filename,'rb'))
    return ff

def saveDictFile(dict, fileName):
    f = file(fileName,'wb')
    pickle.dump(dict,f)
    f.close()

def loadDictFile(fileName):
    if (os.path.isfile(fileName)) == True:
        f = open(fileName,'rb')
        modelDict = pickle.load(f)
        f.close()
        return modelDict
    else:
        print 'load model function, file is not exist!'

def calibrate(missing_value,correct_value):
    if correct_value != missing_value:
        return correct_value
    else:
        return np.NaN

def file_read_and_feature_extract(platform, R01_classification_model_dir):
    if platform == 'mindtrails':
        dir_path = R01_classification_model_dir + 'Mindtrails_11_10_2017/'

        task_dict = {
            'PRE': ['QOL', 'RR', 'BBSIQ', 'DASS21_DS', 'OASIS', 'trial', 'demographic'],
            'SESSION1': ['OASIS', 'trial'],
            'SESSION2': ['OASIS', 'trial'],
            'SESSION3': ['RR', 'BBSIQ', 'QOL', 'DASS21_DS', 'OASIS', 'trial'],
            'SESSION4': ['OASIS', 'trial'],
            'SESSION5': ['OASIS', 'trial'],
            'SESSION6': ['RR', 'BBSIQ', 'QOL', 'DASS21_DS', 'OASIS', 'trial'],
            'SESSION7': ['OASIS', 'trial'],
            'SESSION8': ['RR', 'BBSIQ', 'QOL', 'DASS21_DS', 'OASIS', 'DASS21_AS', 'trial'],
            'POST': ['RR', 'BBSIQ', 'QOL', 'DASS21_DS', 'OASIS', 'DASS21_AS']
        }

        task_time = dir_path + 'preprocessing/time_log_1110.csv'


    elif platform == 'templeton':
        dir_path = R01_classification_model_dir + 'Templeton_Oct_10_2017/'

        task_dict = {
            'preTest': ['credibility', 'demographic', 'mental', 'whatibelieve', 'phq4'],
            'firstSession': ['affect_pre', 'trial', 'affect_post', 'relatability', ''],
            'secondSession': ['affect_pre', 'trial', 'affect_post', 'expectancy_bias', 'whatibelieve'],
            'thridSession': ['affect_pre', 'trial', 'affect_post', 'expectancy_bias'],
            'fourthSession': ['affect_pre', 'trial', 'affect_post', 'relatability', 'expectancy_bias', 'whatibelieve',
                              'phq4']
        }

        task_time = dir_path + 'preprocessing/time_log_1010.csv'

    session_name_list = task_dict.keys()
    session_completion_dict, dropout_label = participant_session_completion_extract(platform, task_time, session_name_list)

    if platform == 'mindtrails':
        demographic_dict, QOL_dict, OASIS_dict, RR_dict, BBSIQ_dict, DASS21_AS_dict, DASS21_DS_dict, trial_dict, \
        dwell_time_dict, control_normal_dict= feature_generate_mindtrails(dir_path)

        return demographic_dict, QOL_dict, OASIS_dict, RR_dict, BBSIQ_dict, DASS21_AS_dict, DASS21_DS_dict, trial_dict, \
               dwell_time_dict, session_completion_dict, dropout_label, control_normal_dict

    elif platform == 'templeton':
        demographic_dict, affect_dict, credibility_dict, mental_dict, whatibelieve_dict, relatability_dict, \
        expectancy_dict, phq4_dict, trial_dict = feature_generate_templeton(dir_path)

        return demographic_dict, affect_dict, credibility_dict, mental_dict, whatibelieve_dict, relatability_dict, \
        expectancy_dict, phq4_dict, trial_dict, session_completion_dict, dropout_label


def participant_session_completion_extract(platform, task_time, session_name_list):
    session_completion_dict = {}
    if platform == 'mindtrails':
        '''
        task_time_dict = loadDictFile(task_time)
        for e in task_time_dict:
            session_completion_dict[e] = {}
            for session_id in task_time_dict[e]:
                session_completion_dict[e][session_id] = task_time_dict[e][session_id]
        '''
        ff = csvRead(task_time)
        for line in ff:
            if (line[0] != 'session_name') and (line[0] != 'participantdao_id'):
                if (line[1] != 'datetime_CR'):
                    id = int(line[0])
                    session_completion_dict[id] = {
                        'PRE': line[2],
                        'SESSION1': line[3],
                        'SESSION2': line[4],
                        'SESSION3': line[5],
                        'SESSION4': line[6],
                        'SESSION5': line[7],
                        'SESSION6': line[8],
                        'SESSION7': line[9],
                        'SESSION8': line[10],
                        'POST': line[1]
                    }

    elif platform == 'templeton':
        ff = csvRead(task_time)
        for line in ff:
            if (line[0] != 'sessionName') and (line[0] != 'participantId'):
                if(line[1] != 'datetime_CR'):
                    id =int(line[0])
                    session_completion_dict[id] = {
                        'preTest': line[2],
                        'firstSession': line[3],
                        'secondSession': line[4],
                        'thridSession': line[5],
                        'fourthSession': line[6],
                    }

    dropout_label = {}
    for e in session_completion_dict:
        dropout_label[e] = {}
        for i in session_name_list:

            if session_completion_dict[e][i] != '':
                dropout_label[e][i] = '0'
            else:
                dropout_label[e][i] = '1'

    return session_completion_dict, dropout_label


##########################################################
####         feature of mindtrails
##########################################################

def feature_generate_mindtrails(dir_path):
    SUDs = dir_path + 'SUDS_recovered_Nov_10_2017.csv'
    QOL = dir_path + 'QOL_recovered_Nov_10_2017.csv'
    OASIS = dir_path + 'OA_label_fixed_11_10_2017.csv'
    RR = dir_path + 'RR_recovered_Nov_10_2017.csv'
    BBSIQ = dir_path + 'BBSIQ_recovered_Nov_10_2017.csv'
    DASS21_AS = dir_path + 'DASS21_AS_recovered_Nov_10_2017.csv'
    DASS21_DS = dir_path + 'DASS21_DS_recovered_Nov_10_2017.csv'

    trial = dir_path + 'TrialDAO_recovered_Nov_10_2017.csv'
    demographic = dir_path + 'Demographic_recovered_Nov_10_2017.csv'

    control = dir_path + 'ParticipantExportDAO_recovered_Nov_10_2017.csv'

    test_dict = {'QOL': QOL, 'OASIS': OASIS, 'RR': RR, 'BBSIQ': BBSIQ, 'DASS21_AS': DASS21_AS,
                 'DASS21_DS': DASS21_DS, 'trial': trial, 'demographic': demographic, 'control_normal_group': control}

    for t in test_dict:
        if t == 'demographic':
            demographic_dict = mindtrails_demographic_extract(test_dict[t])
        elif t == 'QOL':
            QOL_dict = mindtrails_QOL_extract(test_dict[t])
        elif t == 'OASIS':
            OASIS_dict = mindtrails_OASIS_extract(test_dict[t])
        elif t == 'RR':
            RR_dict = mindtrails_RR_extract(test_dict[t])
        elif t == 'BBSIQ':
            BBSIQ_dict = mindtrails_BBSIQ_extract(test_dict[t])
        elif t == 'DASS21_AS':
            DASS21_AS_dict = mindtrails_DASS21_AS_extract(test_dict[t])
        elif t == 'DASS21_DS':
            DASS21_DS_dict = mindtrails_DASS21_DS_extract(test_dict[t])
        elif t == 'trial':
            trial_dict = mindtrails_trial_extract(test_dict[t])
        elif t == 'control_normal_group':
            control_normal_dict = mindtrails_control_normal_group_extract(control)

    dwell_time_dict = mindtrails_dwell_time_extract(dir_path)

    return demographic_dict, QOL_dict, OASIS_dict, RR_dict, BBSIQ_dict, DASS21_AS_dict, DASS21_DS_dict, trial_dict, \
           dwell_time_dict, control_normal_dict

def mindtrails_demographic_extract(demographic):
    ff = csvRead(demographic)

    demographic_dict = {}
    for line in ff:
        if line[9] != 'participantRSA':
            participant_id = line[9]
            time = line[1]
            birth_year = line[0]
            education = line[2]
            employmentStat = line[3]
            ethnicity = line[4]
            gender = line[5]
            income = line[7]
            maritalStat = line[8]
            participant_reason = line[10]
            race = line[11]
            country = line[12]

            demographic_dict[participant_id] = {
                'birth_year': birth_year,
                'country': country,
                'education': education,
                'employmentStat': employmentStat,
                'ethnicity': ethnicity,
                'gender': gender,
                'income': income,
                'maritalStat': maritalStat,
                'race': race
            }

    return demographic_dict

def mindtrails_QOL_extract(QOL):
    ff = csvRead(QOL)

    QOL_dict = {}
    for line in ff:
        if line[10] != 'participantRSA':
            participant_id = line[10]
            time = line[1]
            sessionid = line[15]

            children = calibrate(555, int(line[0]))
            expression = calibrate(555, int(line[2]))
            friend = calibrate(555, int(line[3]))
            health = calibrate(555, int(line[4]))
            helping = calibrate(555, int(line[5]))
            independence = calibrate(555, int(line[7]))
            learning = calibrate(555, int(line[8]))
            material = calibrate(555, int(line[9]))
            reading = calibrate(555, int(line[12]))
            recreation = calibrate(555, int(line[13]))
            relationship = calibrate(555, int(line[14]))
            socializing = calibrate(555, int(line[16]))
            spouse= calibrate(555, int(line[17]))
            understanding = calibrate(555, int(line[18]))
            work = calibrate(555, int(line[19]))

            if participant_id in QOL_dict:
                QOL_dict[participant_id][sessionid] = {
                    'children': children,
                    'expression': expression,
                    'friend': friend,
                    'health': health,
                    'helping': helping,
                    'independence': independence,
                    'learning': learning,
                    'material': material,
                    'reading': reading,
                    'recreation': recreation,
                    'relationship': relationship,
                    'socializing': socializing,
                    'spouse': spouse,
                    'understanding': understanding,
                    'work': work
                }
            else:
                QOL_dict[participant_id] = {
                    sessionid: {
                        'children': children,
                        'expression': expression,
                        'friend': friend,
                        'health': health,
                        'helping': helping,
                        'independence': independence,
                        'learning': learning,
                        'material': material,
                        'reading': reading,
                        'recreation': recreation,
                        'relationship': relationship,
                        'socializing': socializing,
                        'spouse': spouse,
                        'understanding': understanding,
                        'work': work
                    }
                }

    return QOL_dict

def mindtrails_OASIS_extract(OASIS):
    ff = csvRead(OASIS)

    OASIS_dict = {}
    for line in ff:
        if line[8] != 'participantRSA':
            participant_id = line[8]
            sessionid = line[9]

            anxious_freq = calibrate(555, int(line[0]))
            anxious_sev = calibrate(555, int(line[1]))
            avoid = calibrate(555, int(line[2]))
            interfere = calibrate(555, int(line[5]))
            interfere_social = calibrate(555, int(line[6]))

            if participant_id in OASIS_dict:
                OASIS_dict[participant_id][sessionid] = {
                    'anxious_freq': anxious_freq,
                    'anxious_sev': anxious_sev,
                    'avoid': avoid,
                    'interfere': interfere,
                    'interfere_social': interfere_social
                }
            else:
                OASIS_dict[participant_id] = {
                    sessionid:{
                        'anxious_freq': anxious_freq,
                        'anxious_sev': anxious_sev,
                        'avoid': avoid,
                        'interfere': interfere,
                        'interfere_social': interfere_social
                    }
                }
    return OASIS_dict


def mindtrails_RR_extract(RR):
    ff = csvRead(RR)

    RR_dict = {}
    for line in ff:
        if line[26] != 'participantRSA':
            participant_id = line[26]
            sessionid = line[31]
            time = line[4]

            blood_test_NF = calibrate(555, int(line[0]))
            blood_test_NS = calibrate(555, int(line[1]))
            blood_test_PF = calibrate(555, int(line[2]))
            blood_test_PS = calibrate(555, int(line[3]))
            elevator_NF = calibrate(555, int(line[5]))
            elevator_NS = calibrate(555, int(line[6]))
            elevator_PF = calibrate(555, int(line[7]))
            elevator_PS = calibrate(555, int(line[8]))
            job_NF = calibrate(555, int(line[10]))
            job_NS = calibrate(555, int(line[11]))
            job_PF = calibrate(555, int(line[12]))
            job_PS = calibrate(555, int(line[13]))
            lunch_NF = calibrate(555, int(line[14]))
            lunch_NS = calibrate(555, int(line[15]))
            lunch_PF = calibrate(555, int(line[16]))
            lunch_PS = calibrate(555, int(line[17]))
            meeting_friend_NF = calibrate(555, int(line[18]))
            meeting_friend_NS = calibrate(555, int(line[19]))
            meeting_friend_PF = calibrate(555, int(line[20]))
            meeting_friend_PS = calibrate(555, int(line[21]))
            noise_NF = calibrate(555, int(line[22]))
            noise_NS = calibrate(555, int(line[23]))
            noise_PF = calibrate(555, int(line[24]))
            noise_PS = calibrate(555, int(line[25]))
            scrape_NF = calibrate(555, int(line[27]))
            scrape_NS = calibrate(555, int(line[28]))
            scrape_PF = calibrate(555, int(line[29]))
            scrape_PS = calibrate(555, int(line[30]))
            shopping_NF = calibrate(555, int(line[32]))
            shopping_NS = calibrate(555, int(line[33]))
            shopping_PF = calibrate(555, int(line[34]))
            shopping_PS = calibrate(555, int(line[35]))
            wedding_NF = calibrate(555, int(line[36]))
            wedding_NS = calibrate(555, int(line[37]))
            wedding_PF = calibrate(555, int(line[38]))
            wedding_PS = calibrate(555, int(line[39]))

            if participant_id in RR_dict:
                RR_dict[participant_id][sessionid] = {
                    'blood_test_NF': blood_test_NF, 'blood_test_NS': blood_test_NS, 'blood_test_PF': blood_test_PF,
                    'blood_test_PS': blood_test_PS, 'elevator_NF': elevator_NF, 'elevator_NS': elevator_NS,
                    'elevator_PF': elevator_PF, 'elevator_PS': elevator_PS, 'job_NF': job_NF,
                    'job_NS': job_NS, 'job_PF': job_PF, 'job_PS': job_PS,
                    'lunch_NF': lunch_NF, 'lunch_NS': lunch_NS, 'lunch_PF': lunch_PF,
                    'lunch_PS': lunch_PS, 'meeting_friend_NF': meeting_friend_NF, 'meeting_friend_NS': meeting_friend_NS,
                    'meeting_friend_PF': meeting_friend_PF, 'meeting_friend_PS': meeting_friend_PS, 'noise_NF': noise_NF,
                    'noise_NS': noise_NS, 'noise_PF': noise_PF, 'noise_PS': noise_PS,
                    'scrape_NF': scrape_NF, 'scrape_NS': scrape_NS, 'scrape_PF': scrape_PF,
                    'scrape_PS': scrape_PS, 'shopping_NF': shopping_NF, 'shopping_NS': shopping_NS,
                    'shopping_PF': shopping_PF, 'shopping_PS': shopping_PS, 'wedding_NF': wedding_NF,
                    'wedding_NS': wedding_NS, 'wedding_PF': wedding_PF, 'wedding_PS': wedding_PS
                }
            else:
                RR_dict[participant_id] = {
                    sessionid: {
                        'blood_test_NF': blood_test_NF, 'blood_test_NS': blood_test_NS, 'blood_test_PF': blood_test_PF,
                        'blood_test_PS': blood_test_PS, 'elevator_NF': elevator_NF, 'elevator_NS': elevator_NS,
                        'elevator_PF': elevator_PF, 'elevator_PS': elevator_PS, 'job_NF': job_NF,
                        'job_NS': job_NS, 'job_PF': job_PF, 'job_PS': job_PS,
                        'lunch_NF': lunch_NF, 'lunch_NS': lunch_NS, 'lunch_PF': lunch_PF,
                        'lunch_PS': lunch_PS, 'meeting_friend_NF': meeting_friend_NF,
                        'meeting_friend_NS': meeting_friend_NS,
                        'meeting_friend_PF': meeting_friend_PF, 'meeting_friend_PS': meeting_friend_PS,
                        'noise_NF': noise_NF,
                        'noise_NS': noise_NS, 'noise_PF': noise_PF, 'noise_PS': noise_PS,
                        'scrape_NF': scrape_NF, 'scrape_NS': scrape_NS, 'scrape_PF': scrape_PF,
                        'scrape_PS': scrape_PS, 'shopping_NF': shopping_NF, 'shopping_NS': shopping_NS,
                        'shopping_PF': shopping_PF, 'shopping_PS': shopping_PS, 'wedding_NF': wedding_NF,
                        'wedding_NS': wedding_NS, 'wedding_PF': wedding_PF, 'wedding_PS': wedding_PS
                    }
                }


    return RR_dict


def mindtrails_BBSIQ_extract(BBSIQ):
    ff = csvRead(BBSIQ)

    BBSIQ_dict = {}
    for line in ff:
        if line[26] != 'participantRSA':
            participant_id = line[26]
            time = line[9]
            sessionid = line[30]

            breath_flu = calibrate(555, int(line[0]))
            breath_physically = calibrate(555, int(line[1]))
            breath_suffocate = calibrate(555, int(line[2]))
            chest_heart = calibrate(555, int(line[3]))
            chest_indigestion = calibrate(555, int(line[4]))
            chest_sore = calibrate(555, int(line[5]))
            confused_cold = calibrate(555, int(line[6]))
            confused_outofmind = calibrate(555, int(line[7]))
            confused_work = calibrate(555, int(line[8]))
            dizzy_ate = calibrate(555, int(line[10]))
            dizzy_ill = calibrate(555, int(line[11]))
            dizzy_overtired = calibrate(555, int(line[12]))
            friend_helpful = calibrate(555, int(line[13]))
            friend_incompetent = calibrate(555, int(line[14]))
            friend_moreoften = calibrate(555, int(line[15]))
            heart_active = calibrate(555, int(line[16]))
            heart_excited = calibrate(555, int(line[17]))
            heart_wrong = calibrate(555, int(line[18]))
            jolt_burglar = calibrate(555, int(line[20]))
            jolt_dream = calibrate(555, int(line[21]))
            jolt_wind = calibrate(555, int(line[22]))
            lightheaded_eat = calibrate(555, int(line[23]))
            lightheaded_faint = calibrate(555, int(line[24]))
            lightheaded_sleep = calibrate(555, int(line[25]))
            party_boring = calibrate(555, int(line[27]))
            party_hear = calibrate(555, int(line[28]))
            party_preoccupied = calibrate(555, int(line[29]))
            shop_bored = calibrate(555, int(line[31]))
            hop_concentrating = calibrate(555, int(line[32]))
            shop_irritating = calibrate(555, int(line[33]))
            smoke_cig = calibrate(555, int(line[34]))
            smoke_food = calibrate(555, int(line[35]))
            smoke_house = calibrate(555, int(line[36]))
            urgent_bill = calibrate(555, int(line[37]))
            urgent_died = calibrate(555, int(line[38]))
            urgent_junk = calibrate(555, int(line[39]))
            vision_glasses = calibrate(555, int(line[40]))
            vision_illness = calibrate(555, int(line[41]))
            vision_strained = calibrate(555, int(line[42]))
            visitors_bored = calibrate(555, int(line[43]))
            visitors_engagement = calibrate(555, int(line[44]))
            visitors_outstay = calibrate(555, int(line[45]))

            if participant_id in BBSIQ_dict:
                BBSIQ_dict[participant_id][sessionid] = {
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
            else:
                BBSIQ_dict[participant_id] = {
                    sessionid: {
                        'breath_flu': breath_flu, 'breath_physically': breath_physically,
                        'breath_suffocate': breath_suffocate,
                        'chest_heart': chest_heart, 'chest_indigestion': chest_indigestion, 'chest_sore': chest_sore,
                        'confused_cold': confused_cold, 'confused_outofmind': confused_outofmind,
                        'confused_work': confused_work,
                        'dizzy_ate': dizzy_ate, 'dizzy_ill': dizzy_ill, 'dizzy_overtired': dizzy_overtired,
                        'friend_helpful': friend_helpful, 'friend_incompetent': friend_incompetent,
                        'friend_moreoften': friend_moreoften,
                        'heart_active': heart_active, 'heart_excited': heart_excited, 'heart_wrong': heart_wrong,
                        'jolt_burglar': jolt_burglar, 'jolt_dream': jolt_dream, 'jolt_wind': jolt_wind,
                        'lightheaded_eat': lightheaded_eat, 'lightheaded_faint': lightheaded_faint,
                        'lightheaded_sleep': lightheaded_sleep,
                        'party_boring': party_boring, 'party_hear': party_hear, 'party_preoccupied': party_preoccupied,
                        'shop_bored': shop_bored, 'hop_concentrating': hop_concentrating,
                        'shop_irritating': shop_irritating,
                        'smoke_cig': smoke_cig, 'smoke_food': smoke_food, 'smoke_house': smoke_house,
                        'urgent_bill': urgent_bill, 'urgent_died': urgent_died, 'urgent_junk': urgent_junk,
                        'vision_glasses': vision_glasses, 'vision_illness': vision_illness,
                        'vision_strained': vision_strained,
                        'visitors_bored': visitors_bored, 'visitors_engagement': visitors_engagement,
                        'visitors_outstay': visitors_outstay
                    }
                }
    return BBSIQ_dict


def mindtrails_DASS21_AS_extract(DASS21_AS):
    ff = csvRead(DASS21_AS)

    DASS21_AS_dict = {}
    for line in ff:
        if line[7] != 'participantRSA':
            participant_id = line[7]
            time = line[1]
            sessionid = line[9]
            breathing = calibrate(555, int(line[0]))
            dryness = calibrate(555, int(line[2]))
            heart = calibrate(555, int(line[3]))
            panic = calibrate(555, int(line[5]))
            scared = calibrate(555, int(line[8]))
            trembling = calibrate(555, int(line[11]))
            worry = calibrate(555, int(line[12]))


            if participant_id in DASS21_AS_dict:
                DASS21_AS_dict[participant_id][sessionid] = {
                    'breathing': breathing,
                    'dryness': dryness,
                    'heart': heart,
                    'panic': panic,
                    'scared': scared,
                    'trembling': trembling,
                    'worry': worry
                }
            else:
                DASS21_AS_dict[participant_id] = {
                    sessionid: {
                        'breathing': breathing,
                        'dryness': dryness,
                        'heart': heart,
                        'panic': panic,
                        'scared': scared,
                        'trembling': trembling,
                        'worry': worry
                    }
                }
    return DASS21_AS_dict


def mindtrails_DASS21_DS_extract(DASS21_DS):
    ff = csvRead(DASS21_DS)

    DASS21_DS_dict = {}
    for line in ff:
        if line[9] != 'participantRSA':
            participant_id = line[9]
            time = line[1]
            sessionid = line[10]
            blue = calibrate(555, int(line[0]))
            difficult = calibrate(555, int(line[2]))
            meaningless = calibrate(555, int(line[5]))
            noenthusiastic = calibrate(555, int(line[6]))
            nopositive = calibrate(555, int(line[7]))
            noworth = calibrate(555, int(line[8]))

            if participant_id in DASS21_DS_dict:
                DASS21_DS_dict[participant_id][sessionid] = {
                    'blue': blue,
                    'difficult': difficult,
                    'meaningless': meaningless,
                    'noenthusiastic': noenthusiastic,
                    'nopositive': nopositive,
                    'noworth': noworth
                }
            else:
                DASS21_DS_dict[participant_id] = {
                    sessionid: {
                        'blue': blue,
                        'difficult': difficult,
                        'meaningless': meaningless,
                        'noenthusiastic': noenthusiastic,
                        'nopositive': nopositive,
                        'noworth': noworth
                    }
                }
    return DASS21_DS_dict


def mindtrails_trial_extract(trial):
    ff = csvRead(trial)

    trial_dict = {}
    for line in ff:
        if (line[6] != 'participant'):
            participant_id = line[6]
            sessionid = line[13]
            trial_id = line[14]

            if line[0] != '':
                letter_latency_time_first = int(np.ceil(int(line[0])/1000))
            else:
                letter_latency_time_first = 60

            if line[4] != '':
                letter_latency_time = int(np.ceil(int(line[4])/1000))
            else:
                letter_latency_time = 60
            letter_correct = line[3]    # true or false

            if line[1] != '':
                question_latency_time_first = int(np.ceil(int(line[1])/1000))
            else:
                question_latency_time_first = 60

            if line[10] != '':
                question_latency_time = int(np.ceil(int(line[10])/1000))
            else:
                question_latency_time = 60
            question_correct = line[9]  # true or false

            positive = line[7]  # true or false

            if participant_id in trial_dict:
                if sessionid in trial_dict[participant_id]:
                    trial_dict[participant_id][sessionid][trial_id] = {
                        'letter_latency_first': letter_latency_time_first,
                        'letter_latency': letter_latency_time,
                        'letter_correct': letter_correct,
                        'question_latency_first': question_latency_time_first,
                        'question_latency': question_latency_time,
                        'question_correct': question_correct,
                        'positive': positive
                    }
                else:
                    trial_dict[participant_id][sessionid] = {
                        trial_id: {
                            'letter_latency_first': letter_latency_time_first,
                            'letter_latency': letter_latency_time,
                            'letter_correct': letter_correct,
                            'question_latency_first': question_latency_time_first,
                            'question_latency': question_latency_time,
                            'question_correct': question_correct,
                            'positive': positive
                        }
                    }
            else:
                trial_dict[participant_id] = {
                    sessionid: {
                        trial_id: {
                            'letter_latency_first': letter_latency_time_first,
                            'letter_latency': letter_latency_time,
                            'letter_correct': letter_correct,
                            'question_latency_first': question_latency_time_first,
                            'question_latency': question_latency_time,
                            'question_correct': question_correct,
                            'positive': positive
                        }
                    }
                }

    return trial_dict

def mindtrails_dwell_time_extract(dir_path):
    task_time_dict = loadDictFile(dir_path + 'preprocessing/task_time_dict.txt')
    dwell_time_dict = {}
    duration_two_task = []
    for participant_id in task_time_dict:
        # print e, task_dict[e]
        dwell_time_dict[participant_id] = {}
        for session_id in task_time_dict[participant_id]:
            if len(task_time_dict[participant_id][session_id]) == 1:
                dwell_time_dict[participant_id][session_id] = 0
            else:
                t_list = []
                for k in task_time_dict[participant_id][session_id]:
                    t = task_time_dict[participant_id][session_id][k]
                    ts = time.mktime(time.strptime(str(pd.to_datetime(t)), '%Y-%m-%d %H:%M:%S'))
                    t_list.append(ts)
                dur = max(t_list) - min(t_list)
                dwell_time_dict[participant_id][session_id] = dur

                t_list_sort = sorted(t_list)
                for i in range(0, len(t_list_sort) - 1):
                    duration_two_task.append(t_list_sort[i + 1] - t_list_sort[i])

    return dwell_time_dict

def mindtrails_control_normal_group_extract(control):
    no_training = []
    training = []
    ff = csvRead(control)
    for line in ff:
        if line[0] != 'active':
            id = int(line[5])
            control_flag = line[2]
            if (control_flag == 'NEUTRAL') and ((id in no_training) is False):
                no_training.append(id)
            elif ((id in training) is False):
                training.append(id)
        control_normal_dict = {'no_training': no_training, 'training': training}

    return control_normal_dict

def mindtrails_feature_vector_generation(training_set_session, prediction_session, participant_list, demographic_dict, QOL_dict,
                                         OASIS_dict, RR_dict, BBSIQ_dict, DASS21_AS_dict, DASS21_DS_dict,
                                         trial_dict, dwell_time_dict, dropout_label):
    '''
    Generate the feature vector for each participant in participant_list based on the training_set_session.
    :param training_set_session:
    :param prediction_session:
    :param user_list:
    :param demographic_dict:
    :param QOL_dict:
    :param OASIS_dict:
    :param RR_dict:
    :param BBSIQ_dict:
    :param DASS21_AS_dict:
    :param DASS21_DS_dict:
    :param trial_dict:
    :param dwell_time_dict:
    :param dropout_label:
    :return:
    '''
    '''
    task_dict = {
        'PRE': ['QOL', 'RR', 'BBSIQ', 'DASS21_DS', 'OASIS', 'trial', 'demographic', 'dwell_time'],
        'SESSION1': ['OASIS', 'trial', 'dwell_time'],
        'SESSION2': ['OASIS', 'trial', 'dwell_time'],
        'SESSION3': ['RR', 'BBSIQ', 'QOL', 'DASS21_DS', 'OASIS', 'trial', 'dwell_time'],
        'SESSION4': ['OASIS', 'trial', 'dwell_time'],
        'SESSION5': ['OASIS', 'trial', 'dwell_time'],
        'SESSION6': ['RR', 'BBSIQ', 'QOL', 'DASS21_DS', 'OASIS', 'trial', 'dwell_time'],
        'SESSION7': ['OASIS', 'trial', 'dwell_time'],
        'SESSION8': ['RR', 'BBSIQ', 'QOL', 'DASS21_DS', 'OASIS', 'DASS21_AS', 'trial', 'dwell_time'],
        'POST': ['RR', 'BBSIQ', 'QOL', 'DASS21_DS', 'OASIS', 'DASS21_AS', 'dwell_time']
    }
    '''
    # overlapping questionnarires with R01
    task_dict = {
        'PRE': ['RR', 'BBSIQ', 'OASIS', 'demographic', 'session_dwell_time'],
        'SESSION1': ['OASIS', 'session_dwell_time']
    }

    education_level = {'Prefer not to answer': 0, 'Elementary School': 1, 'Some High School': 2,
                       'High School Graduate': 3, 'Some College': 4, "Associate's Degree": 5, 'Some Graduate School': 6,
                       "Bachelor's Degree": 7, 'M.B.A.': 8, "Master's Degree": 9, 'Ph.D.': 10, 'J.D.': 11,
                       'M.D.': 12, 'Other': 13}
    income_level = {'Less than $5,000': 0, '$5,000 through $11,999': 1, '$12,000 through $15,999': 2,
                    '$16,000 through $24,999': 3, '$25,000 through $34,999': 4, '$35,000 through $49,999': 5,
                    '$50,000 through $74,999': 6, '$75,000 through $99,999': 7, '$100,000 through $149,999': 8,
                    '$150,000 through $199,999': 9, '$200,000 through $249,999': 10, '$250,000 or greater': 11,
                    'Other': 12, "Don't know": 13, 'Prefer not to answer': 14}

    truth_vector = []
    feature_vector = []
    for e in participant_list:
        feature_item_list = []
        if dropout_label[e][prediction_session] == '1':
            truth_vector.append(1)
        else:
            truth_vector.append(0)

        single_e_feature_vector = []
        for item in training_set_session:
            # adding features of scores of each questionnaire
            for sub_item in task_dict[item]:
                if sub_item == 'demographic':
                    demo_item_list = ['education', 'income']
                    temp_demo_value_dict = {}
                    if e in demographic_dict:
                        for demo_item in demo_item_list:
                            if demo_item not in demographic_dict[e]:
                                demo_value == 'Other'
                            else:
                                demo_value = demographic_dict[e][demo_item]

                            if ('?' in demo_value) or (demo_value == '') or ('Other' in demo_value) or \
                                    (demo_value == 'Junior High'):
                                temp_demo_value_dict[demo_item] = 'Other'
                            else:
                                temp_demo_value_dict[demo_item] = demo_value
                    else:
                        for demo_item in demo_item_list:
                            temp_demo_value_dict[demo_item] = 'Other'

                    for demo_item in demo_item_list:
                        if demo_item == 'education':
                            single_e_feature_vector.append(education_level[temp_demo_value_dict[demo_item]])
                            feature_item_list.append(item + '_' + sub_item + '_edu')
                        elif demo_item == 'income':
                            single_e_feature_vector.append(income_level[temp_demo_value_dict[demo_item]])
                            feature_item_list.append(item + '_' + sub_item + '_income')

                elif sub_item == 'QOL':
                    temp_QOL_value_list = []
                    if e in QOL_dict:
                        if item in QOL_dict[e]:
                            for QOL_item in QOL_dict[e][item]:
                                if np.isnan(QOL_dict[e][item][QOL_item]) == False:
                                    temp_QOL_value_list.append(QOL_dict[e][item][QOL_item])

                            single_e_feature_vector.append(np.sum(temp_QOL_value_list))
                        else:
                            single_e_feature_vector.append(0.0)
                    else:
                        single_e_feature_vector.append(0.0)
                    feature_item_list.append(item + '_' + sub_item)

                elif sub_item == 'RR':
                    target_value_list = []
                    non_target_value_list = []

                    if e in RR_dict:
                        if item in RR_dict[e]:
                            for RR_item in RR_dict[e][item]:
                                if np.isnan(RR_dict[e][item][RR_item] == False):
                                    if '_NS' in RR_item:
                                        target_value_list.append(RR_dict[e][item][RR_item])
                                    elif '_PS' in RR_item:
                                        target_value_list.append(RR_dict[e][item][RR_item])

                            if np.mean(non_target_value_list) != 0.0:
                                single_e_feature_vector.append(np.mean(target_value_list)/np.mean(non_target_value_list))
                            else:
                                single_e_feature_vector.append(0.0)
                        else:
                            single_e_feature_vector.append(0.0)
                    else:
                        single_e_feature_vector.append(0.0)
                    feature_item_list.append(item + '_' + sub_item)

                elif sub_item == 'BBSIQ':
                    physical_list = ['breath_suffocate', 'chest_heart', 'confused_outofmind', 'dizzy_ill', 'heart_wrong',
                                     'lightheaded_faint', 'vision_illness']

                    non_physical_list = ['breath_flu', 'breath_physically', 'vision_glasses', 'vision_strained',
                                         'lightheaded_eat', 'lightheaded_sleep', 'chest_indigestion', 'chest_sore',
                                         'heart_active', 'heart_excited', 'confused_cold', 'confused_work', 'dizzy_ate',
                                         'dizzy_overtired']

                    threat_list = ['visitors_bored', 'shop_irritating', 'smoke_house', 'friend_incompetent',
                                   'jolt_burglar', 'party_boring', 'urgent_died']

                    non_threat_list = ['visitors_engagement', 'visitors_outstay', 'shop_bored', 'shop_concentrating',
                                       'smoke_cig', 'smoke_food', 'friend_helpful', 'friend_moreoften', 'jolt_dream',
                                       'jolt_wind', 'party_hear', 'party_preoccupied', 'urgent_bill', 'urgent_junk']

                    if e in BBSIQ_dict:
                        if item in BBSIQ_dict[e]:
                            physical_value_list = []
                            non_physical_value_list = []
                            threat_value_list = []
                            non_threat_value_list = []

                            for sub_item in BBSIQ_dict[e][item]:
                                if np.nan(BBSIQ_dict[e][item][sub_item]) == False:
                                    if sub_item in physical_list:
                                        physical_value_list.append(BBSIQ_dict[e][item][sub_item])
                                    elif sub_item in non_physical_list:
                                        non_physical_value_list.append(BBSIQ_dict[e][item][sub_item])
                                    elif sub_item in threat_list:
                                        threat_value_list.append(BBSIQ_dict[e][item][sub_item])
                                    elif sub_item in non_threat_list:
                                        non_threat_value_list.append(BBSIQ_dict[e][item][sub_item])

                            if np.mean(non_physical_value_list) != 0.0:
                                single_e_feature_vector.append(np.mean(physical_value_list)/np.mean(non_physical_value_list))
                            else:
                                single_e_feature_vector.append(0.0)

                            if np.mean(non_threat_value_list) != 0.0:
                                single_e_feature_vector.append(np.mean(threat_value_list)/np.mean(non_threat_value_list))
                            else:
                                single_e_feature_vector.append(0.0)

                        else:
                            single_e_feature_vector.append(0.0)
                            single_e_feature_vector.append(0.0)
                    else:
                        single_e_feature_vector.append(0.0)
                        single_e_feature_vector.append(0.0)
                    feature_item_list.append(item + '_' + sub_item + '_physical')
                    feature_item_list.append(item + '_' + sub_item + '_threat')

                elif sub_item == 'DASS21_DS':
                    if e in DASS21_DS_dict:
                        if item in DASS21_DS_dict[e]:
                            temp_DASS21_DS_value_list = []
                            for sub_item in DASS21_DS_dict[e][item]:
                                if np.isnan(DASS21_DS_dict[e][item][sub_item]) == False:
                                    temp_DASS21_DS_value_list.append(DASS21_DS_dict[e][item][sub_item])

                            single_e_feature_vector.append(np.sum(temp_DASS21_DS_value_list))
                        else:
                            single_e_feature_vector.append(0.0)
                    else:
                        single_e_feature_vector.append(0.0)
                    feature_item_list.append(item + '_' + sub_item)

                elif sub_item == 'DASS21_AS':
                    if e in DASS21_AS_dict:
                        if item in DASS21_AS_dict[e]:
                            temp_DASS21_AS_value_list = []
                            for sub_item in DASS21_AS_dict[e][item]:
                                if np.isnan(DASS21_AS_dict[e][item][sub_item]) == False:
                                    temp_DASS21_AS_value_list.append(DASS21_AS_dict[e][item][sub_item])

                            single_e_feature_vector.append(np.sum(temp_DASS21_AS_value_list))
                        else:
                            single_e_feature_vector.append(0.0)
                    else:
                        single_e_feature_vector.append(0.0)

                    feature_item_list.append(item + '_' + sub_item)

                elif sub_item == 'OASIS':
                    if e in OASIS_dict:
                        if item in OASIS_dict[e]:
                            temp_OASIS_value_list = []
                            for sub_item in OASIS_dict[e][item]:
                                if np.isnan(OASIS_dict[e][item][sub_item]) == False:
                                    temp_OASIS_value_list.append(OASIS_dict[e][item][sub_item])

                            single_e_feature_vector.append(np.sum(temp_OASIS_value_list))
                        else:
                            single_e_feature_vector.append(0.0)
                    else:
                        single_e_feature_vector.append(0.0)

                    feature_item_list.append(item + '_' + sub_item)

                elif sub_item == 'trial':
                    if e in trial_dict:
                        if item in trial_dict[e]:
                            temp_letter_latency_time_list = []
                            temp_question_latency_time_list = []
                            temp_letter_latency_time_diff_list = []
                            temp_question_latency_time_diff_list = []
                            num_correctness = 0

                            for trial_id in trial_dict[e][item]:
                                temp_letter_latency_time_list.append(trial_dict[e][item][trial_id]['letter_latency'])
                                temp_question_latency_time_list.append(trial_dict[e][item][trial_id]['question_latency'])

                                if trial_dict[e][item][trial_id]['letter_latency_first'] != '':
                                    temp_letter_latency_time_diff_list.append(trial_dict[e][item][trial_id]['letter_latency'] -
                                                                              trial_dict[e][item][trial_id]['letter_latency_first'])

                                if trial_dict[e][item][trial_id]['question_latency_first'] != '':
                                    temp_question_latency_time_diff_list.append(trial_dict[e][item][trial_id]['question_latency'] -
                                                                                trial_dict[e][item][trial_id]['question_latency_first'])

                                if trial_dict[e][item][trial_id]['letter_correct'] == 'True':
                                    num_correctness += 1
                                if trial_dict[e][item][trial_id]['question_correct'] == 'True':
                                    num_correctness += 1

                            if len(temp_letter_latency_time_list) != 0:
                                single_e_feature_vector.append(np.mean(temp_letter_latency_time_list))
                                single_e_feature_vector.append(np.std(temp_letter_latency_time_list))
                            else:
                                single_e_feature_vector.append(0.0)
                                single_e_feature_vector.append(0.0)

                            if len(temp_question_latency_time_list) != 0:
                                single_e_feature_vector.append(np.mean(temp_question_latency_time_list))
                                single_e_feature_vector.append(np.std(temp_question_latency_time_list))
                            else:
                                single_e_feature_vector.append(0.0)
                                single_e_feature_vector.append(0.0)

                            if len(temp_letter_latency_time_diff_list) != 0:
                                single_e_feature_vector.append(np.mean(temp_letter_latency_time_diff_list))
                                single_e_feature_vector.append(np.std(temp_letter_latency_time_diff_list))
                            else:
                                single_e_feature_vector.append(0.0)
                                single_e_feature_vector.append(0.0)


                            if len(temp_question_latency_time_diff_list) != 0:
                                single_e_feature_vector.append(np.mean(temp_question_latency_time_diff_list))
                                single_e_feature_vector.append(np.std(temp_question_latency_time_diff_list))
                            else:
                                single_e_feature_vector.append(0.0)
                                single_e_feature_vector.append(0.0)

                            single_e_feature_vector.append(num_correctness)

                        else:
                            for l in range(9):
                                single_e_feature_vector.append(0.0)
                    else:
                        for l in range(9):
                            single_e_feature_vector.append(0.0)
                    feature_item_list.append(item + '_' + sub_item + '_letter_latency_time_mean')
                    feature_item_list.append(item + '_' + sub_item + '_letter_latency_time_std')
                    feature_item_list.append(item + '_' + sub_item + '_question_latency_time_mean')
                    feature_item_list.append(item + '_' + sub_item + '_question_latency_time_std')
                    feature_item_list.append(item + '_' + sub_item + '_letter_latency_diff_time_mean')
                    feature_item_list.append(item + '_' + sub_item + '_letter_latency_diff_time_std')
                    feature_item_list.append(item + '_' + sub_item + '_question_latency_diff_time_mean')
                    feature_item_list.append(item + '_' + sub_item + '_question_latency_diff_time_std')
                    feature_item_list.append(item + '_' + sub_item + '_number_correctness')

                if sub_item == 'session_dwell_time':
                    # adding features of dwell time
                    if e in dwell_time_dict:
                        if item in dwell_time_dict[e]:
                            if sub_item in dwell_time_dict[e][item]:
                                single_e_feature_vector.append(dwell_time_dict[e][item][sub_item])
                            else:
                                single_e_feature_vector.append(0)
                        else:
                            for i in range(len(task_dict[item]) - 1):
                                single_e_feature_vector.append(0)
                    else:
                        for i in range(len(task_dict[item]) - 1):
                            single_e_feature_vector.append(0)

                    for sub_sub_item in task_dict[item]:
                        if sub_sub_item != 'session_dwell_time':
                            feature_item_list.append(item + '_' +  sub_sub_item + 'session_dwell_time')

        feature_vector.append(single_e_feature_vector)

    return feature_vector, truth_vector, feature_item_list

##########################################################
####         feature of Templeton
##########################################################
def feature_generate_templeton(dir_path):
    credibility = dir_path + 'Credibility_recovered_Oct_10_2017.csv'
    mental = dir_path + 'MentalHealthHistory_recovered_Oct_10_2017.csv'
    whatibelieve = dir_path + 'WhatIBelieve_recovered_Oct_10_2017.csv'
    phq4 = dir_path + 'Phq4_recovered_Oct_10_2017.csv'
    affect = dir_path + 'Affect_recovered_Oct_10_2017.csv'
    relatability = dir_path + 'Relatability_recovered_Oct_10_2017.csv'
    expectancy = dir_path + 'ExpectancyBias_recovered_Oct_10_2017.csv'
    trial = dir_path + 'JsPsychTrial_recovered_Oct_10_2017.csv'
    demographic = dir_path + 'Demographics_recovered_Oct_10_2017.csv'

    test_dict = {'affect': affect, 'credibility': credibility, 'mental': mental, 'whatibelieve': whatibelieve,
                 'relatability': relatability, 'expectancy': expectancy, 'phq4': phq4, 'trial': trial,
                 'demographic': demographic}

    for t in test_dict:
        if t == 'demographic':
            demographic_dict = templeton_demographic_extract(test_dict[t])
        elif t == 'affect':
            affect_dict = templeton_affect_extract(test_dict[t])
        elif t == 'credibility':
            credibility_dict = templeton_credibility_extract(test_dict[t])
        elif t == 'mental':
            mental_dict = templeton_mental_extract(test_dict[t])
        elif t == 'whatibelieve':
            whatibelieve_dict = templeton_whatibelieve_extract(test_dict[t])
        elif t == 'relatability':
            relatability_dict = templeton_relatability_extract(test_dict[t])
        elif t == 'expectancy':
            expectancy_dict = templeton_expectancy_extract(test_dict[t])
        elif t == 'phq4':
            phq4_dict = templeton_phq4_extract(test_dict[t])
        elif t == 'trial':
            trial_dict = templeton_trial_extract(test_dict[t])

    return demographic_dict, affect_dict, credibility_dict, mental_dict, whatibelieve_dict, relatability_dict, \
           expectancy_dict, phq4_dict, trial_dict

def templeton_demographic_extract(demographic):
    ff = csvRead(demographic)

    demographic_dict = {}
    for line in ff:
        participant_id = line[11]

        if participant_id != 'participantRSA':
            participant_id = int(participant_id)
            timeonpage = float(line[17])

            birth_year = int(line[0])
            country = line[1]
            device = line[3]
            education = line[4]
            employmentStat = line[5]
            ethnicity = line[6]
            gender = line[7]
            income = line[9]
            maritalStat = line[10]
            race = line[14]

            demographic_dict[participant_id] = {
                'timeonpage': timeonpage,
                'birth_year': birth_year,
                'country': country,
                'device': device,
                'education': education,
                'employmentStat': employmentStat,
                'ethnicity': ethnicity,
                'gender': gender,
                'income': income,
                'maritalStat': maritalStat,
                'race': race
            }
    return demographic_dict

def templeton_affect_extract(affect):
    ff = csvRead(affect)

    affect_dict = {}
    for line in ff:
        participant_id = line[3]
        sessionid = line[5]
        tag = line[6]

        if participant_id != 'participantRSA':
            participant_id = int(participant_id)
            timeonpage = float(line[7])
            negFeelings = calibrate(555,int(line[2]))
            posFeelings = calibrate(555,int(line[4]))

            if participant_id in affect_dict:
                if sessionid in affect_dict[participant_id]:
                    affect_dict[participant_id][sessionid][tag] = {
                        'negFeelings': negFeelings,
                        'posFeelings': posFeelings,
                        'timeonpage': timeonpage
                    }
                else:
                    affect_dict[participant_id][sessionid] = {
                        tag: {
                            'negFeelings': negFeelings,
                            'posFeelings': posFeelings,
                            'timeonpage': timeonpage
                        }
                    }

            else:
                affect_dict[participant_id] = {
                    sessionid: {
                        tag: {
                            'negFeelings': negFeelings,
                            'posFeelings': posFeelings,
                            'timeonpage': timeonpage
                        }
                    }
                }

    return affect_dict

def templeton_credibility_extract(credibility):
    ff = csvRead(credibility)

    credibility_dict = {}
    for line in ff:
        participant_id = line[5]
        if participant_id != 'participantRSA':
            participant_id = int(participant_id)
            timeonpage = float(line[8])

            if participant_id in credibility_dict:
                credibility_dict[participant_id]['preTest'] = {'timeonpage': timeonpage}
            else:
                credibility_dict[participant_id] = {
                    'preTest': {
                        'timeonpage': timeonpage
                    }
                }
    return credibility_dict

def templeton_mental_extract(mental):
    ff = csvRead(mental)

    mental_dict = {}
    for line in ff:
        participant_id = line[32]

        if participant_id != 'participantRSA':
            participant_id = int(participant_id)
            timeonpage = float(line[49])

            if participant_id in mental_dict:
                mental_dict[participant_id]['timeonpage'] = timeonpage
            else:
                mental_dict[participant_id] = {
                    'timeonpage': timeonpage
                }
    return mental_dict

def templeton_whatibelieve_extract(whatibelieve):
    ff = csvRead(whatibelieve)

    whatibelieve_dict = {}
    for line in ff:
        participant_id = line[7]
        sessionid = line[10]

        if participant_id != 'participantRSA':
            participant_id = int(participant_id)
            timeonpage = float(line[12])

            alwaysChangeThinking = calibrate(555,int(line[0]))
            compared = calibrate(555,int(line[1]))
            difficultTasks = calibrate(555,int(line[3]))
            hardlyEver = calibrate(555,int(line[4]))
            learn = calibrate(555,int(line[6]))
            particularThinking = calibrate(555,int(line[8]))
            performEffectively = calibrate(555,int(line[9]))
            wrongWill = calibrate(555,int(line[13]))

            ####### Scoring of three different subscales ########
            NGSES = np.nanmean(np.array([difficultTasks,compared,performEffectively])) * 3
            GMM = np.nanmean(np.array([5-learn,particularThinking,alwaysChangeThinking])) * 3
            Optimism = np.nanmean(np.array([hardlyEver,wrongWill])) * 2


            if participant_id in whatibelieve_dict:
                whatibelieve_dict[participant_id][sessionid] = {
                    'alwaysChangeThinking': alwaysChangeThinking,
                    'compared': compared,
                    'difficultTasks': difficultTasks,
                    'hardlyEver': hardlyEver,
                    'learn': learn,
                    'particularThinking':particularThinking,
                    'performEffectively': performEffectively,
                    'wrongWill': wrongWill,
                    'timeonpage': timeonpage,
                    'NGSES': NGSES,
                    'GMM': GMM,
                    'Optimism': Optimism
                }
            else:
                whatibelieve_dict[participant_id] = {}
                whatibelieve_dict[participant_id][sessionid] = {
                    'alwaysChangeThinking': alwaysChangeThinking,
                    'compared': compared,
                    'difficultTasks': difficultTasks,
                    'hardlyEver': hardlyEver,
                    'learn': learn,
                    'particularThinking': particularThinking,
                    'performEffectively': performEffectively,
                    'wrongWill': wrongWill,
                    'timeonpage': timeonpage,
                    'NGSES': NGSES,
                    'GMM': GMM,
                    'Optimism': Optimism
                }
    return whatibelieve_dict

def templeton_relatability_extract(relatability):
    ff = csvRead(relatability)

    relatability_dict = {}
    for line in ff:
        participant_id = line[3]
        sessionid = line[5]

        if participant_id != 'participantRSA':
            participant_id = int(participant_id)

            behaving = calibrate(555,int(line[0]))
            relate = calibrate(555,int(line[4]))
            relatability = behaving + relate
            timeonpage = float(line[7])

            if participant_id in relatability_dict:
                relatability_dict[participant_id][sessionid] = {
                    'behaving': behaving,
                    'relate': relate,
                    'timeonpage': timeonpage,
                    'relatability': relatability
                }
            else:
                relatability_dict[participant_id] = {
                    sessionid: {
                        'behaving': behaving,
                        'relate': relate,
                        'timeonpage': timeonpage,
                        'relatability': relatability
                    }
                }
    return relatability_dict

def templeton_expectancy_extract(expectancy):
    ff = csvRead(expectancy)

    expectancy_dict = {}
    for line in ff:
        participant_id = line[7]
        sessionid = line[10]

        if (participant_id != '') and (participant_id != 'participant'):
            participant_id = int(participant_id)
            timeonpage = float(line[17])

            bagel = calibrate(555,int(line[0]))
            consideredAdvancement = calibrate(555,int(line[1]))
            financiallySecure = calibrate(555,int(line[3]))
            lunch = calibrate(555,int(line[5]))
            offend = calibrate(555,int(line[6]))
            reruns = calibrate(555,int(line[8]))
            ruining = calibrate(555,int(line[9]))
            settleIn = calibrate(555,int(line[12]))
            shortRest = calibrate(555,int(line[13]))
            stuck = calibrate(555,int(line[14]))
            thermostat = calibrate(555,int(line[16]))
            verySick = calibrate(555,int(line[18]))

            ###### Scoring of EB ########
            positive_AVE = np.nanmean(np.array([shortRest,settleIn,consideredAdvancement,financiallySecure]))
            negative_AVE = np.nanmean(np.array([verySick,offend,stuck,ruining]))
            ExpectancyBias = positive_AVE - negative_AVE

            if participant_id in expectancy_dict:
                expectancy_dict[participant_id][sessionid] = {
                    'bagel': bagel,
                    'consideredAdvancement': consideredAdvancement,
                    'financiallySecure': financiallySecure,
                    'lunch': lunch,
                    'offend':offend,
                    'reruns': reruns,
                    'ruining': ruining,
                    'settleIn': settleIn,
                    'shortRest': shortRest,
                    'stuck': stuck,
                    'thermostat': thermostat,
                    'verySick': verySick,
                    'timeonpage': timeonpage,
                    'positive_AVE': positive_AVE,
                    'negative_AVE': negative_AVE,
                    'ExpectancyBias': ExpectancyBias
                }
            else:
                expectancy_dict[participant_id] = {
                    sessionid: {
                        'bagel': bagel,
                        'consideredAdvancement': consideredAdvancement,
                        'financiallySecure': financiallySecure,
                        'lunch': lunch,
                        'offend': offend,
                        'reruns': reruns,
                        'ruining': ruining,
                        'settleIn': settleIn,
                        'shortRest': shortRest,
                        'stuck': stuck,
                        'thermostat': thermostat,
                        'verySick': verySick,
                        'timeonpage': timeonpage,
                        'positive_AVE': positive_AVE,
                        'negative_AVE': negative_AVE,
                        'ExpectancyBias': ExpectancyBias
                    }
                }
    return expectancy_dict

def templeton_phq4_extract(phq4):
    ff = csvRead(phq4)

    phq4_dict = {}
    for line in ff:
        participant_id = line[4]
        sessionid = line[6]

        if participant_id != 'participantRSA':
            participant_id = int(participant_id)
            timeonpage = float(line[8])

            depressed = int(line[1])
            nervous = int(line[3])
            pleasure = int(line[5])
            worry = int(line[9])
            anxiety = nervous + worry
            depression = depressed + pleasure


            if participant_id in phq4_dict:
                phq4_dict[participant_id][sessionid] = {
                    'depressed': depressed,
                    'nervous': nervous,
                    'pleasure': pleasure,
                    'worry': worry,
                    'timeonpage': timeonpage,
                    'anxiety': anxiety,
                    'depression': depression
                }
            else:
                phq4_dict[participant_id] = {}
                phq4_dict[participant_id][sessionid] = {
                    'depressed': depressed,
                    'nervous': nervous,
                    'pleasure': pleasure,
                    'worry': worry,
                    'timeonpage': timeonpage,
                    'anxiety': anxiety,
                    'depression': depression
                }

    return phq4_dict

def templeton_trial_extract(trial):
    ##########################################################################################
    # 1. rt: react time
    # 2. rt_correct: react time of the first attempt
    # 3. time_elapsed: accumulate time of rt
    # 4. unit: millisecond
    ##########################################################################################

    ff = csvRead(trial)

    enroll_condition = {}
    trial_dict = {}
    for line in ff:
        participant_id = line[6]
        sessionid = line[9]
        condition = line[1]

        first_try_correct = line[2]
        device = line[3]
        rt_correct = line[7]

        if participant_id != 'participantId':
            participant_id = int(participant_id)
            time_elapsed = float(line[12])

            if (participant_id in enroll_condition) is False:
                enroll_condition[participant_id] = condition

            if participant_id in trial_dict:
                if sessionid in trial_dict[participant_id]:
                    trial_dict[participant_id][sessionid]['time_elapsed'].append(time_elapsed)
                    trial_dict[participant_id][sessionid]['first_try_correct'].append(first_try_correct)
                    trial_dict[participant_id][sessionid]['rt_correct'].append(rt_correct)
                else:
                    trial_dict[participant_id][sessionid] = {
                        'time_elapsed': [time_elapsed],
                        'device': device,
                        'first_try_correct': [first_try_correct],
                        'rt_correct': [rt_correct]
                    }
            else:
                trial_dict[participant_id] = {
                    sessionid:
                    {
                    'time_elapsed': [time_elapsed],
                    'device': device,
                    'first_try_correct': [first_try_correct],
                    'rt_correct': [rt_correct]
                    }
                }

    return trial_dict

def templeton_feature_vector_generation(training_set_session, prediction_session, participant_list, demographic_dict,
                                        affect_dict, credibility_dict, mental_dict, whatibelieve_dict, relatability_dict,
                                        expectancy_dict, phq4_dict, trial_dict, dropout_label):
    '''
    Generate the feature vector for each participant in participant_list based on the training_set_session.
    :param training_set_session:
    :param prediction_session:
    :param user_list:
    :param demographic_dict:
    :param affect_dict:
    :param credibility_dict:
    :param mental_dict:
    :param whatibelieve_dict:
    :param relatability_dict:
    :param expectancy_dict:
    :param phq4_dict:
    :param trial_dict:
    :param dropout_label:
    :return:
    '''

    '''
    task_dict = {
        'preTest': ['credibility', 'demographic', 'mental', 'whatibelieve', 'phq4'],
        'firstSession': ['affect_pre', 'trial', 'affect_post', 'relatability'],
        'secondSession': ['affect_pre', 'trial', 'affect_post', 'expectancy_bias', 'whatibelieve'],
        'thridSession': ['affect_pre', 'trial', 'affect_post', 'expectancy_bias'],
        'fourthSession': ['affect_pre', 'trial', 'affect_post', 'relatability', 'expectancy_bias', 'whatibelieve',
                          'phq4']
    }
    '''
    # overlapping questionnarires with R01
    task_dict = {
        'preTest': ['credibility', 'demographic', 'mental'],
        'firstSession': ['affect_pre', 'trial', 'affect_post']
    }


    education_level = {'Prefer not to answer': 0, 'Elementary School': 1, 'Some High School': 2,
                       'High School Graduate': 3, 'Some College': 4, "Associate's Degree": 5, 'Some Graduate School': 6,
                       "Bachelor's Degree": 7, 'M.B.A.': 8, "Master's Degree": 9, 'Ph.D.': 10, 'J.D.': 11,
                       'M.D.': 12, 'Other': 13}
    income_level = {'Less than $5,000': 0, '$5,000 through $11,999': 1, '$12,000 through $15,999': 2,
                    '$16,000 through $24,999': 3, '$25,000 through $34,999': 4, '$35,000 through $49,999': 5,
                    '$50,000 through $74,999': 6, '$75,000 through $99,999': 7, '$100,000 through $149,999': 8,
                    '$150,000 through $199,999': 9, '$200,000 through $249,999': 10, '$250,000 or greater': 11,
                    'Other': 12, "Don't know": 13, 'Prefer not to answer': 14}

    truth_vector = []
    feature_vector = []

    for e in participant_list:
        feature_item_list = []
        if dropout_label[e][prediction_session] == '1':
            truth_vector.append(1)
        else:
            truth_vector.append(0)

        single_e_feature_vector = []
        for item in training_set_session:
            # adding features of scores of questionnaire
            for sub_item in task_dict[item]:
                if sub_item == 'demographic':
                    demo_item_list = ['education', 'income']
                    temp_demo_value_dict = {}
                    if e in demographic_dict:
                        for demo_item in demo_item_list:
                            if demo_item not in demographic_dict[e]:
                                demo_value == 'Other'
                            else:
                                demo_value = demographic_dict[e][demo_item]

                            if ('?' in demo_value) or (demo_value == '') or ('Other' in demo_value) or \
                                    (demo_value == 'Junior High'):
                                temp_demo_value_dict[demo_item] = 'Other'
                            else:
                                temp_demo_value_dict[demo_item] = demo_value
                    else:
                        for demo_item in demo_item_list:
                            temp_demo_value_dict[demo_item] = 'Other'
                        temp_demo_value_dict['timeonpage'] = 3600*24*2

                    for demo_item in demo_item_list:
                        if demo_item == 'education':
                            single_e_feature_vector.append(education_level[temp_demo_value_dict[demo_item]])
                            feature_item_list.append(item + '_' + sub_item + '_edu')
                        elif demo_item == 'income':
                            single_e_feature_vector.append(income_level[temp_demo_value_dict[demo_item]])
                            feature_item_list.append(item + '_' + sub_item + '_income')
                        elif demo_item == 'timeonpage':
                            single_e_feature_vector.append(temp_demo_value_dict['timeonpage'])
                            feature_item_list.append(item + '_' + sub_item + '_timeonpage')


                elif sub_item == 'credibility':
                    if e in credibility_dict:
                        if item in credibility_dict[e]:
                            single_e_feature_vector.append(credibility_dict[e][item]['timeonpage'])
                        else:
                            single_e_feature_vector.append(0)
                    else:
                        single_e_feature_vector.append(0)

                    feature_item_list.append(item + '_' + sub_item + '_timeonpage')

                elif sub_item == 'mental':
                    if e in mental_dict:
                        if item in mental_dict[e]:
                            single_e_feature_vector.append(mental_dict[e][item]['timeonpage'])
                        else:
                            single_e_feature_vector.append(0)
                    else:
                        single_e_feature_vector.append(0)

                    feature_item_list.append(item + '_' + sub_item + '_timeonpage')

                elif sub_item == 'whatibelieve':
                    score_list = ['NGSES', 'GMM', 'Optimism']
                    if e in whatibelieve_dict:
                        if item in whatibelieve_dict[e]:
                            for score_item in score_list:
                                if np.isnan(whatibelieve_dict[e][item][score_item]) == False:
                                    single_e_feature_vector.append(whatibelieve_dict[e][item][score_item])
                                else:
                                    single_e_feature_vector.append(0.0)

                            single_e_feature_vector.append(whatibelieve_dict[e][item]['timeonpage'])
                        else:
                            for l in range(3):
                                single_e_feature_vector.append(0.0)
                            single_e_feature_vector.append(0)
                    else:
                        for l in range(3):
                            single_e_feature_vector.append(0.0)
                        single_e_feature_vector.append(0)

                    for score_item in score_list:
                        feature_item_list.append(item + '_' + sub_item + '_' + score_item)
                    feature_item_list.append(item + '_' + sub_item + '_timeonpage')


                elif sub_item == 'phq4':
                    score_list = ['anxiety', 'depression']
                    if e in phq4_dict:
                        if item in phq4_dict[e]:
                            for score_item in score_list:
                                if np.isnan(phq4_dict[e][item][score_item]) == False:
                                    single_e_feature_vector.append(phq4_dict[e][item][score_item])
                                else:
                                    single_e_feature_vector.append(0.0)

                            single_e_feature_vector.append(phq4_dict[e][item]['timeonpage'])
                        else:
                            for l in range(2):
                                single_e_feature_vector.append(0.0)
                            single_e_feature_vector.append(0)
                    else:
                        for l in range(3):
                            single_e_feature_vector.append(0.0)
                    for score_item in score_list:
                        feature_item_list.append(item + '_' + sub_item + '_' + score_item)
                    feature_item_list.append(item + '_' + sub_item + '_timeonpage')

                elif sub_item == 'affect_pre':
                    score_list = ['posFeelings', 'negFeelings']
                    if e in affect_dict:
                        if item in affect_dict[e]:
                            if 'pre' in affect_dict[e][item]:
                                for score_item in score_list:
                                    single_e_feature_vector.append(affect_dict[e][item]['pre'][score_item])

                                single_e_feature_vector.append(affect_dict[e][item]['pre']['timeonpage'])
                            else:
                                for l in range(3):
                                    single_e_feature_vector.append(0.0)
                        else:
                            for l in range(3):
                                single_e_feature_vector.append(0.0)
                    else:
                        for l in range(3):
                            single_e_feature_vector.append(0.0)

                    for score_item in score_list:
                        feature_item_list.append(item + '_' + sub_item + score_item)
                    feature_item_list.append(item + '_' + sub_item + '_timeonpage')


                elif sub_item == 'affect_post':
                    score_list = ['posFeelings', 'negFeelings']
                    if e in affect_dict:

                        if item in affect_dict[e]:
                            if 'post' in affect_dict[e][item]:
                                for score_item in score_list:
                                    single_e_feature_vector.append(affect_dict[e][item]['post'][score_item])
                                single_e_feature_vector.append(affect_dict[e][item]['post']['timeonpage'])

                            else:
                                for l in range(3):
                                    single_e_feature_vector.append(0.0)
                        else:
                            for l in range(3):
                                single_e_feature_vector.append(0.0)
                    else:
                        for l in range(3):
                            single_e_feature_vector.append(0.0)

                    for score_item in score_list:
                        feature_item_list.append(item + '_' + sub_item  + score_item)
                    feature_item_list.append(item + '_' + sub_item + '_timeonpage')

                elif sub_item == 'relatability':
                    score_list = ['relatability']
                    if e in relatability_dict:
                        if item in relatability_dict[e]:
                            for score_item in score_list:
                                if np.isnan(relatability_dict[e][item][score_item]):
                                    single_e_feature_vector.append(0.0)
                                else:
                                    single_e_feature_vector.append(relatability_dict[e][item][score_item])

                            single_e_feature_vector.append(relatability_dict[e][item]['timeonpage'])
                        else:
                            single_e_feature_vector.append(0.0)
                            single_e_feature_vector.append(0)
                    else:
                        single_e_feature_vector.append(0.0)
                        single_e_feature_vector.append(0)

                    for score_item in score_list:
                        feature_item_list.append(item + '_' + sub_item + '_' + score_item)
                    feature_item_list.append(item + '_' + sub_item + '_timeonpage')


                elif sub_item == 'expectancy_bias':
                    score_list = ['positive_AVE', 'negative_AVE', 'ExpectancyBias']
                    if e in expectancy_dict:
                        if item in expectancy_dict[e]:
                            for score_item in score_list:
                                single_e_feature_vector.append(expectancy_dict[e][item][score_item])
                            single_e_feature_vector.append(expectancy_dict[e][item]['timeonpage'])
                        else:
                            for l in range(4):
                                single_e_feature_vector.append(0.0)
                    else:
                        for l in range(4):
                            single_e_feature_vector.append(0.0)
                    for score_item in score_list:
                        feature_item_list.append(item + '_' + sub_item + '_' + score_item)
                    feature_item_list.append(item + '_' + sub_item + '_timeonpage')

                elif sub_item == 'trial':
                    if e in trial_dict:
                        if item in trial_dict[e]:
                            if 'first_try_correct' in trial_dict[e][item]:
                                number_correctness = trial_dict[e][item]['first_try_correct'].count('TRUE')
                                single_e_feature_vector.append(number_correctness)
                            else:
                                single_e_feature_vector.append(0)

                            if ('time_elapsed' in trial_dict[e][item]) and (len(trial_dict[e][item]['time_elapsed']) > 1):
                                time_elapsed_list = []
                                for i in range(len(trial_dict[e][item]['time_elapsed'])):
                                    if i == 0:
                                        time_elapsed_list.append(trial_dict[e][item]['time_elapsed'][i])
                                    else:
                                        time_elapsed_list.append(trial_dict[e][item]['time_elapsed'][i] - trial_dict[e][item]['time_elapsed'][i - 1])
                                if len(time_elapsed_list) > 0:
                                    single_e_feature_vector.append(np.mean(time_elapsed_list))
                                    single_e_feature_vector.append(np.std(time_elapsed_list))
                                    single_e_feature_vector.append(trial_dict[e][item]['time_elapsed'][len(trial_dict[e][item]['time_elapsed']) - 1])
                                else:
                                    for l in range(3):
                                        single_e_feature_vector.append(0)
                            else:
                                for l in range(3):
                                    single_e_feature_vector.append(0)
                        else:
                            for l in range(4):
                                single_e_feature_vector.append(0)
                    else:
                        for l in range(4):
                            single_e_feature_vector.append(0)

                    feature_item_list.append(item + '_' + sub_item + '_first_try_correct')
                    feature_item_list.append(item + '_' + sub_item + '_latency_time_mean')
                    feature_item_list.append(item + '_' + sub_item + '_latency_time_std')
                    feature_item_list.append(item + '_' + sub_item + '_timeonpage')

        feature_vector.append(single_e_feature_vector)

    return feature_vector, truth_vector, feature_item_list


##########################################################
####         feature of R01
##########################################################
def R01_affect_extract(affect_result):
    affect_dict = {}
    for line in affect_result:
        sessionid = line[2]
        tag = line[3]
        timeonpage = float(line[4])
        negFeelings = calibrate(555, int(line[5]))
        posFeelings = calibrate(555, int(line[6]))
        participant_id = int(line[7])

        if participant_id in affect_dict:
            if sessionid in affect_dict[participant_id]:
                affect_dict[participant_id][sessionid][tag] = {
                    'negFeelings': negFeelings,
                    'posFeelings': posFeelings,
                    'timeonpage': timeonpage
                }
            else:
                affect_dict[participant_id][sessionid] = {
                    tag: {
                        'negFeelings': negFeelings,
                        'posFeelings': posFeelings,
                        'timeonpage': timeonpage
                    }
                }

        else:
            affect_dict[participant_id] = {
                sessionid: {
                    tag: {
                        'negFeelings': negFeelings,
                        'posFeelings': posFeelings,
                        'timeonpage': timeonpage
                    }
                }
            }

    return affect_dict

def R01_BBSIQ_extract(BBSIQ_result):
    BBSIQ_dict = {}
    for line in BBSIQ_result:
        sessionid = line[2]
        timeonpage = float(line[4])
        participant_id = int(line[47])

        breath_flu = calibrate(555, int(line[5]))
        breath_physically = calibrate(555, int(line[6]))
        breath_suffocate = calibrate(555, int(line[7]))
        chest_heart = calibrate(555, int(line[8]))
        chest_indigestion = calibrate(555, int(line[9]))
        chest_sore = calibrate(555, int(line[10]))
        confused_cold = calibrate(555, int(line[11]))
        confused_outofmind = calibrate(555, int(line[12]))
        confused_work = calibrate(555, int(line[13]))
        dizzy_ate = calibrate(555, int(line[14]))
        dizzy_ill = calibrate(555, int(line[15]))
        dizzy_overtired = calibrate(555, int(line[16]))
        friend_helpful = calibrate(555, int(line[17]))
        friend_incompetent = calibrate(555, int(line[18]))
        friend_moreoften = calibrate(555, int(line[19]))
        heart_active = calibrate(555, int(line[20]))
        heart_excited = calibrate(555, int(line[21]))
        heart_wrong = calibrate(555, int(line[22]))
        jolt_burglar = calibrate(555, int(line[23]))
        jolt_dream = calibrate(555, int(line[24]))
        jolt_wind = calibrate(555, int(line[25]))
        lightheaded_eat = calibrate(555, int(line[26]))
        lightheaded_faint = calibrate(555, int(line[27]))
        lightheaded_sleep = calibrate(555, int(line[28]))
        party_boring = calibrate(555, int(line[29]))
        party_hear = calibrate(555, int(line[30]))
        party_preoccupied = calibrate(555, int(line[31]))
        shop_bored = calibrate(555, int(line[32]))
        hop_concentrating = calibrate(555, int(line[33]))
        shop_irritating = calibrate(555, int(line[34]))
        smoke_cig = calibrate(555, int(line[35]))
        smoke_food = calibrate(555, int(line[36]))
        smoke_house = calibrate(555, int(line[37]))
        urgent_bill = calibrate(555, int(line[38]))
        urgent_died = calibrate(555, int(line[39]))
        urgent_junk = calibrate(555, int(line[40]))
        vision_glasses = calibrate(555, int(line[41]))
        vision_illness = calibrate(555, int(line[42]))
        vision_strained = calibrate(555, int(line[43]))
        visitors_bored = calibrate(555, int(line[44]))
        visitors_engagement = calibrate(555, int(line[45]))
        visitors_outstay = calibrate(555, int(line[46]))

        if participant_id in BBSIQ_dict:
            BBSIQ_dict[participant_id][sessionid] = {
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
                'visitors_outstay': visitors_outstay, 'timeonpage': timeonpage
            }
        else:
            BBSIQ_dict[participant_id] = {
                sessionid: {
                    'breath_flu': breath_flu, 'breath_physically': breath_physically,
                    'breath_suffocate': breath_suffocate,
                    'chest_heart': chest_heart, 'chest_indigestion': chest_indigestion, 'chest_sore': chest_sore,
                    'confused_cold': confused_cold, 'confused_outofmind': confused_outofmind,
                    'confused_work': confused_work,
                    'dizzy_ate': dizzy_ate, 'dizzy_ill': dizzy_ill, 'dizzy_overtired': dizzy_overtired,
                    'friend_helpful': friend_helpful, 'friend_incompetent': friend_incompetent,
                    'friend_moreoften': friend_moreoften,
                    'heart_active': heart_active, 'heart_excited': heart_excited, 'heart_wrong': heart_wrong,
                    'jolt_burglar': jolt_burglar, 'jolt_dream': jolt_dream, 'jolt_wind': jolt_wind,
                    'lightheaded_eat': lightheaded_eat, 'lightheaded_faint': lightheaded_faint,
                    'lightheaded_sleep': lightheaded_sleep,
                    'party_boring': party_boring, 'party_hear': party_hear, 'party_preoccupied': party_preoccupied,
                    'shop_bored': shop_bored, 'hop_concentrating': hop_concentrating,
                    'shop_irritating': shop_irritating,
                    'smoke_cig': smoke_cig, 'smoke_food': smoke_food, 'smoke_house': smoke_house,
                    'urgent_bill': urgent_bill, 'urgent_died': urgent_died, 'urgent_junk': urgent_junk,
                    'vision_glasses': vision_glasses, 'vision_illness': vision_illness,
                    'vision_strained': vision_strained,
                    'visitors_bored': visitors_bored, 'visitors_engagement': visitors_engagement,
                    'visitors_outstay': visitors_outstay, 'timeonpage': timeonpage
                }
            }
    return BBSIQ_dict

def R01_DASS21_AS_extract(DASS21_AS_result):
    DASS21_AS_dict = {}
    for line in DASS21_AS_result:
        participant_id = int(line[13])
        timeonpage = float(line[4])
        sessionid = line[2]
        breathing = calibrate(555, int(line[5]))
        dryness = calibrate(555, int(line[6]))
        heart = calibrate(555, int(line[7]))
        panic = calibrate(555, int(line[8]))
        scared = calibrate(555, int(line[9]))
        trembling = calibrate(555, int(line[11]))
        worry = calibrate(555, int(line[12]))


        if participant_id in DASS21_AS_dict:
            DASS21_AS_dict[participant_id][sessionid] = {
                'breathing': breathing,
                'dryness': dryness,
                'heart': heart,
                'panic': panic,
                'scared': scared,
                'trembling': trembling,
                'worry': worry,
                'timeonpage': timeonpage
            }
        else:
            DASS21_AS_dict[participant_id] = {
                sessionid: {
                    'breathing': breathing,
                    'dryness': dryness,
                    'heart': heart,
                    'panic': panic,
                    'scared': scared,
                    'trembling': trembling,
                    'worry': worry,
                    'timeonpage': timeonpage
                }
            }
    return DASS21_AS_dict

def R01_demographic_extract(demographic_result):
    demographic_dict = {}
    for line in demographic_result:
        participant_id = int(line[16])
        timeonpage = float(line[4])

        birth_year = int(line[5])
        country = line[6]
        education = line[7]
        employmentStat = line[8]
        ethnicity = line[9]
        gender = line[10]
        income = line[11]
        maritalStat = line[12]
        race = line[15]

        demographic_dict[participant_id] = {
            'timeonpage': timeonpage,
            'birth_year': birth_year,
            'country': country,
            'education': education,
            'employmentStat': employmentStat,
            'ethnicity': ethnicity,
            'gender': gender,
            'income': income,
            'maritalStat': maritalStat,
            'race': race
        }
    return demographic_dict



def R01_OASIS_extract(OASIS_result):
    OASIS_dict = {}
    for line in OASIS_result:
        participant_id = int(line[10])
        sessionid = line[2]
        timeonpage = float(line[4])

        anxious_freq = calibrate(555, int(line[5]))
        anxious_sev = calibrate(555, int(line[6]))
        avoid = calibrate(555, int(line[7]))
        interfere = calibrate(555, int(line[8]))
        interfere_social = calibrate(555, int(line[9]))

        if participant_id in OASIS_dict:
            OASIS_dict[participant_id][sessionid] = {
                'anxious_freq': anxious_freq,
                'anxious_sev': anxious_sev,
                'avoid': avoid,
                'interfere': interfere,
                'interfere_social': interfere_social,
                'timeonpage': timeonpage
            }
        else:
            OASIS_dict[participant_id] = {
                sessionid:{
                    'anxious_freq': anxious_freq,
                    'anxious_sev': anxious_sev,
                    'avoid': avoid,
                    'interfere': interfere,
                    'interfere_social': interfere_social,
                    'timeonpage': timeonpage
                }
            }
    return OASIS_dict

def R01_RR_extract(RR_result):
    RR_dict = {}
    for line in RR_result:

        participant_id = int(line[41])
        sessionid = line[2]
        timeonpage = float(line[4])

        blood_test_NF = calibrate(555, int(line[5]))
        blood_test_NS = calibrate(555, int(line[6]))
        blood_test_PF = calibrate(555, int(line[7]))
        blood_test_PS = calibrate(555, int(line[8]))
        elevator_NF = calibrate(555, int(line[9]))
        elevator_NS = calibrate(555, int(line[10]))
        elevator_PF = calibrate(555, int(line[11]))
        elevator_PS = calibrate(555, int(line[12]))
        job_NF = calibrate(555, int(line[13]))
        job_NS = calibrate(555, int(line[14]))
        job_PF = calibrate(555, int(line[15]))
        job_PS = calibrate(555, int(line[16]))
        lunch_NF = calibrate(555, int(line[17]))
        lunch_NS = calibrate(555, int(line[18]))
        lunch_PF = calibrate(555, int(line[19]))
        lunch_PS = calibrate(555, int(line[20]))
        meeting_friend_NF = calibrate(555, int(line[21]))
        meeting_friend_NS = calibrate(555, int(line[22]))
        meeting_friend_PF = calibrate(555, int(line[23]))
        meeting_friend_PS = calibrate(555, int(line[24]))
        noise_NF = calibrate(555, int(line[25]))
        noise_NS = calibrate(555, int(line[26]))
        noise_PF = calibrate(555, int(line[27]))
        noise_PS = calibrate(555, int(line[28]))
        scrape_NF = calibrate(555, int(line[29]))
        scrape_NS = calibrate(555, int(line[30]))
        scrape_PF = calibrate(555, int(line[31]))
        scrape_PS = calibrate(555, int(line[32]))
        shopping_NF = calibrate(555, int(line[33]))
        shopping_NS = calibrate(555, int(line[34]))
        shopping_PF = calibrate(555, int(line[35]))
        shopping_PS = calibrate(555, int(line[36]))
        wedding_NF = calibrate(555, int(line[37]))
        wedding_NS = calibrate(555, int(line[38]))
        wedding_PF = calibrate(555, int(line[39]))
        wedding_PS = calibrate(555, int(line[40]))

        if participant_id in RR_dict:
            RR_dict[participant_id][sessionid] = {
                'blood_test_NF': blood_test_NF, 'blood_test_NS': blood_test_NS, 'blood_test_PF': blood_test_PF,
                'blood_test_PS': blood_test_PS, 'elevator_NF': elevator_NF, 'elevator_NS': elevator_NS,
                'elevator_PF': elevator_PF, 'elevator_PS': elevator_PS, 'job_NF': job_NF,
                'job_NS': job_NS, 'job_PF': job_PF, 'job_PS': job_PS,
                'lunch_NF': lunch_NF, 'lunch_NS': lunch_NS, 'lunch_PF': lunch_PF,
                'lunch_PS': lunch_PS, 'meeting_friend_NF': meeting_friend_NF, 'meeting_friend_NS': meeting_friend_NS,
                'meeting_friend_PF': meeting_friend_PF, 'meeting_friend_PS': meeting_friend_PS, 'noise_NF': noise_NF,
                'noise_NS': noise_NS, 'noise_PF': noise_PF, 'noise_PS': noise_PS,
                'scrape_NF': scrape_NF, 'scrape_NS': scrape_NS, 'scrape_PF': scrape_PF,
                'scrape_PS': scrape_PS, 'shopping_NF': shopping_NF, 'shopping_NS': shopping_NS,
                'shopping_PF': shopping_PF, 'shopping_PS': shopping_PS, 'wedding_NF': wedding_NF,
                'wedding_NS': wedding_NS, 'wedding_PF': wedding_PF, 'wedding_PS': wedding_PS,
                'timeonpage': timeonpage
            }
        else:
            RR_dict[participant_id] = {
                sessionid: {
                    'blood_test_NF': blood_test_NF, 'blood_test_NS': blood_test_NS, 'blood_test_PF': blood_test_PF,
                    'blood_test_PS': blood_test_PS, 'elevator_NF': elevator_NF, 'elevator_NS': elevator_NS,
                    'elevator_PF': elevator_PF, 'elevator_PS': elevator_PS, 'job_NF': job_NF,
                    'job_NS': job_NS, 'job_PF': job_PF, 'job_PS': job_PS,
                    'lunch_NF': lunch_NF, 'lunch_NS': lunch_NS, 'lunch_PF': lunch_PF,
                    'lunch_PS': lunch_PS, 'meeting_friend_NF': meeting_friend_NF,
                    'meeting_friend_NS': meeting_friend_NS,
                    'meeting_friend_PF': meeting_friend_PF, 'meeting_friend_PS': meeting_friend_PS,
                    'noise_NF': noise_NF,
                    'noise_NS': noise_NS, 'noise_PF': noise_PF, 'noise_PS': noise_PS,
                    'scrape_NF': scrape_NF, 'scrape_NS': scrape_NS, 'scrape_PF': scrape_PF,
                    'scrape_PS': scrape_PS, 'shopping_NF': shopping_NF, 'shopping_NS': shopping_NS,
                    'shopping_PF': shopping_PF, 'shopping_PS': shopping_PS, 'wedding_NF': wedding_NF,
                    'wedding_NS': wedding_NS, 'wedding_PF': wedding_PF, 'wedding_PS': wedding_PS,
                    'timeonpage': timeonpage
                }
            }


    return RR_dict


def R01_credibility_extract(credibility_result):
    credibility_dict = {}
    for line in credibility_result:
        sessionid = line[2]
        timeonpage = float(line[4])
        participant_id = int(line[8])

        if participant_id in credibility_dict:
            credibility_dict[participant_id][sessionid] = {'timeonpage': timeonpage}
        else:
            credibility_dict[participant_id] = {
                sessionid: {
                    'timeonpage': timeonpage
                }
            }
    return credibility_dict


def R01_mental_extract(mental_result):
    mental_dict = {}
    for line in mental_result:
        sessionid = line[2]
        timeonpage = float(line[4])
        participant_id = int(line[43])

        if participant_id in mental_dict:
            mental_dict[participant_id][sessionid] = {'timeonpage': timeonpage}
        else:
            mental_dict[participant_id] = {
                sessionid:{
                    'timeonpage': timeonpage
                }
            }
    return mental_dict

def R01_trial_extract(trial_result):
    ##########################################################################################
    # 1. rt: react time
    # 2. rt_correct: react time of the first attempt
    # 3. time_elapsed: accumulate time of rt
    # 4. unit: millisecond
    ##########################################################################################
    enroll_condition = {}
    trial_dict = {}
    for line in trial_result:
        sessionid = line[2]
        first_try_correct = line[6]
        rt_correct = float(line[10])
        time_elapsed = float(line[13])
        participant_id = int(line[16])

        if participant_id in trial_dict:
            if sessionid in trial_dict[participant_id]:
                trial_dict[participant_id][sessionid]['time_elapsed'].append(time_elapsed)
                trial_dict[participant_id][sessionid]['first_try_correct'].append(first_try_correct)
                trial_dict[participant_id][sessionid]['rt_correct'].append(rt_correct)
            else:
                trial_dict[participant_id][sessionid] = {
                    'time_elapsed': [time_elapsed],
                    'first_try_correct': [first_try_correct],
                    'rt_correct': [rt_correct]
                }
        else:
            trial_dict[participant_id] = {
                sessionid:
                {
                'time_elapsed': [time_elapsed],
                'first_try_correct': [first_try_correct],
                'rt_correct': [rt_correct]
                }
            }

    return trial_dict

def feature_vector_r01_overlap_with_mindtrails(RR_dict, BBSIQ_dict, OASIS_dict, demographics_dict, timeOnPage_dict,
                                               participant_id_list):
    task_dict = {
        'preTest': ['RR', 'BBSIQ', 'OASIS', 'demographic', 'session_dwell_time'],
        'firstSession': ['OASIS', 'session_dwell_time']
    }
    education_level = {'Prefer not to answer': 0, 'Elementary School': 1, 'Some High School': 2,
                       'High School Graduate': 3, 'Some College': 4, "Associate's Degree": 5, 'Some Graduate School': 6,
                       "Bachelor's Degree": 7, 'M.B.A.': 8, "Master's Degree": 9, 'Ph.D.': 10, 'J.D.': 11,
                       'M.D.': 12, 'Other': 13}
    income_level = {'Less than $5,000': 0, '$5,000 through $11,999': 1, '$12,000 through $15,999': 2,
                    '$16,000 through $24,999': 3, '$25,000 through $34,999': 4, '$35,000 through $49,999': 5,
                    '$50,000 through $74,999': 6, '$75,000 through $99,999': 7, '$100,000 through $149,999': 8,
                    '$150,000 through $199,999': 9, '$200,000 through $249,999': 10, '$250,000 or greater': 11,
                    'Other': 12, "Don't know": 13, 'Prefer not to answer': 14}

    single_e_feature_vector = []
    feature_item_list = []

    for sessionid in ['preTest', 'firstSession']:
        for sub_item in task_dict[sessionid]:
            if sub_item == 'demographic':
                demo_item_list = ['education', 'income']
                temp_demo_value_dict = {}
                if len(demo_item_list) > 0:
                    for demo_item in demo_item_list:
                        if demo_item not in demographics_dict:
                            demo_value = 'Other'
                        else:
                            demo_value = demographics_dict[demo_item]

                        if ('?' in demo_value) or (demo_value == '') or ('Other' in demo_value) or \
                                (demo_value == 'Junior High'):
                            temp_demo_value_dict[demo_item] = 'Other'
                        else:
                            temp_demo_value_dict[demo_item] = demo_value
                else:
                    for demo_item in demo_item_list:
                        temp_demo_value_dict[demo_item] = 'Other'
                    temp_demo_value_dict['timeonpage'] = 3600 * 24 * 2

                for demo_item in demo_item_list:
                    if demo_item == 'education':
                        single_e_feature_vector.append(education_level[temp_demo_value_dict[demo_item]])
                        feature_item_list.append(sessionid + '_' + sub_item + '_edu')
                    elif demo_item == 'income':
                        single_e_feature_vector.append(income_level[temp_demo_value_dict[demo_item]])
                        feature_item_list.append(sessionid + '_' + sub_item + '_income')
                    elif demo_item == 'timeonpage':
                        single_e_feature_vector.append(temp_demo_value_dict['timeonpage'])
                        feature_item_list.append(sessionid + '_' + sub_item + '_timeonpage')

            elif sub_item == 'RR':
                target_value_list = []
                non_target_value_list = []

                if sessionid in RR_dict:
                    for RR_item in RR_dict[sessionid]:
                        if np.isnan(RR_dict[sessionid][RR_item] == False):
                            if '_NS' in RR_item:
                                target_value_list.append(RR_dict[sessionid][RR_item])
                            elif '_PS' in RR_item:
                                target_value_list.append(RR_dict[sessionid][RR_item])

                    if np.mean(non_target_value_list) != 0.0:
                        single_e_feature_vector.append(np.mean(target_value_list) / np.mean(non_target_value_list))
                    else:
                        single_e_feature_vector.append(0.0)
                else:
                    single_e_feature_vector.append(0.0)

                feature_item_list.append(sessionid + '_' + sub_item)

            elif sub_item == 'BBSIQ':
                physical_list = ['breath_suffocate', 'chest_heart', 'confused_outofmind', 'dizzy_ill', 'heart_wrong',
                                 'lightheaded_faint', 'vision_illness']

                non_physical_list = ['breath_flu', 'breath_physically', 'vision_glasses', 'vision_strained',
                                     'lightheaded_eat', 'lightheaded_sleep', 'chest_indigestion', 'chest_sore',
                                     'heart_active', 'heart_excited', 'confused_cold', 'confused_work', 'dizzy_ate',
                                     'dizzy_overtired']

                threat_list = ['visitors_bored', 'shop_irritating', 'smoke_house', 'friend_incompetent',
                               'jolt_burglar', 'party_boring', 'urgent_died']

                non_threat_list = ['visitors_engagement', 'visitors_outstay', 'shop_bored', 'shop_concentrating',
                                   'smoke_cig', 'smoke_food', 'friend_helpful', 'friend_moreoften', 'jolt_dream',
                                   'jolt_wind', 'party_hear', 'party_preoccupied', 'urgent_bill', 'urgent_junk']


                if sessionid in BBSIQ_dict:
                    physical_value_list = []
                    non_physical_value_list = []
                    threat_value_list = []
                    non_threat_value_list = []

                    for sub_item in BBSIQ_dict[sessionid]:
                        if np.nan(BBSIQ_dict[sessionid][sub_item]) == False:
                            if sub_item in physical_list:
                                physical_value_list.append(BBSIQ_dict[sessionid][sub_item])
                            elif sub_item in non_physical_list:
                                non_physical_value_list.append(BBSIQ_dict[sessionid][sub_item])
                            elif sub_item in threat_list:
                                threat_value_list.append(BBSIQ_dict[sessionid][sub_item])
                            elif sub_item in non_threat_list:
                                non_threat_value_list.append(BBSIQ_dict[sessionid][sub_item])

                    if np.mean(non_physical_value_list) != 0.0:
                        single_e_feature_vector.append(
                            np.mean(physical_value_list) / np.mean(non_physical_value_list))
                    else:
                        single_e_feature_vector.append(0.0)

                    if np.mean(non_threat_value_list) != 0.0:
                        single_e_feature_vector.append(np.mean(threat_value_list) / np.mean(non_threat_value_list))
                    else:
                        single_e_feature_vector.append(0.0)

                else:
                    single_e_feature_vector.append(0.0)
                    single_e_feature_vector.append(0.0)

                feature_item_list.append(sessionid + '_' + sub_item + '_physical')
                feature_item_list.append(sessionid + '_' + sub_item + '_threat')

            elif sub_item == 'OASIS':
                if sessionid in OASIS_dict:
                    temp_OASIS_value_list = []
                    for sub_item in OASIS_dict[sessionid]:
                        if np.isnan(OASIS_dict[sessionid][sub_item]) == False:
                            temp_OASIS_value_list.append(OASIS_dict[sessionid][sub_item])

                    single_e_feature_vector.append(np.sum(temp_OASIS_value_list))
                else:
                    single_e_feature_vector.append(0.0)

                feature_item_list.append(sessionid + '_' + sub_item)

            elif sub_item == 'session_dwell_time':
                if sessionid in timeOnPage_dict:
                    single_e_feature_vector.append(timeOnPage_dict[sessionid])
                else:
                    single_e_feature_vector.append(0.0)
                feature_item_list.append(sessionid + '_' + sub_item)


    return single_e_feature_vector

def feature_vector_r01_overlap_with_templeton(credibility_dict, mental_dict, affect_dict, trial_dict, demographics_dict,
                                              participant_id_list):
    task_dict = {
        'preTest': ['credibility', 'demographic', 'mental'],
        'firstSession': ['affect_pre', 'trial', 'affect_post']
    }
    education_level = {'Prefer not to answer': 0, 'Elementary School': 1, 'Some High School': 2,
                       'High School Graduate': 3, 'Some College': 4, "Associate's Degree": 5, 'Some Graduate School': 6,
                       "Bachelor's Degree": 7, 'M.B.A.': 8, "Master's Degree": 9, 'Ph.D.': 10, 'J.D.': 11,
                       'M.D.': 12, 'Other': 13}
    income_level = {'Less than $5,000': 0, '$5,000 through $11,999': 1, '$12,000 through $15,999': 2,
                    '$16,000 through $24,999': 3, '$25,000 through $34,999': 4, '$35,000 through $49,999': 5,
                    '$50,000 through $74,999': 6, '$75,000 through $99,999': 7, '$100,000 through $149,999': 8,
                    '$150,000 through $199,999': 9, '$200,000 through $249,999': 10, '$250,000 or greater': 11,
                    'Other': 12, "Don't know": 13, 'Prefer not to answer': 14}

    single_e_feature_vector = []
    feature_item_list = []

    for sessionid in ['preTest', 'firstSession']:
        for sub_item in task_dict[sessionid]:
            if sub_item == 'demographic':
                demo_item_list = ['education', 'income']
                temp_demo_value_dict = {}
                if len(demo_item) > 0:
                    for demo_item in demo_item_list:
                        if demo_item not in demographics_dict:
                            demo_value == 'Other'
                        else:
                            demo_value = demographics_dict[demo_item]

                        if ('?' in demo_value) or (demo_value == '') or ('Other' in demo_value) or \
                                (demo_value == 'Junior High'):
                            temp_demo_value_dict[demo_item] = 'Other'
                        else:
                            temp_demo_value_dict[demo_item] = demo_value
                else:
                    for demo_item in demo_item_list:
                        temp_demo_value_dict[demo_item] = 'Other'
                    temp_demo_value_dict['timeonpage'] = 3600 * 24 * 2

                for demo_item in demo_item_list:
                    if demo_item == 'education':
                        single_e_feature_vector.append(education_level[temp_demo_value_dict[demo_item]])
                        feature_item_list.append(sessionid + '_' + sub_item + '_edu')
                    elif demo_item == 'income':
                        single_e_feature_vector.append(income_level[temp_demo_value_dict[demo_item]])
                        feature_item_list.append(sessionid + '_' + sub_item + '_income')
                    elif demo_item == 'timeonpage':
                        single_e_feature_vector.append(temp_demo_value_dict['timeonpage'])
                        feature_item_list.append(sessionid + '_' + sub_item + '_timeonpage')

            elif sub_item == 'credibility':
                if sessionid in credibility_dict:
                    single_e_feature_vector.append(credibility_dict[sessionid]['timeonpage'])
                else:
                    single_e_feature_vector.append(0)
                feature_item_list.append(sessionid + '_' + sub_item + '_timeonpage')

            elif sub_item == 'mental':
                if sessionid in mental_dict:
                    single_e_feature_vector.append(mental_dict[sessionid]['timeonpage'])
                else:
                    single_e_feature_vector.append(0)
                feature_item_list.append(sessionid + '_' + sub_item + '_timeonpage')

            elif sub_item == 'affect_pre':
                score_list = ['posFeelings', 'negFeelings']

                if sessionid in affect_dict:
                    if 'pre' in affect_dict[sessionid]:
                        for score_item in score_list:
                            single_e_feature_vector.append(affect_dict[sessionid]['pre'][score_item])

                        single_e_feature_vector.append(affect_dict[sessionid]['pre']['timeonpage'])
                    else:
                        for l in range(3):
                            single_e_feature_vector.append(0.0)
                else:
                    for l in range(3):
                        single_e_feature_vector.append(0.0)

                for score_item in score_list:
                    feature_item_list.append(sessionid + '_' + sub_item + score_item)
                feature_item_list.append(sessionid + '_' + sub_item + '_timeonpage')


            elif sub_item == 'affect_post':
                score_list = ['posFeelings', 'negFeelings']
                if sessionid in affect_dict:
                    if 'post' in affect_dict[sessionid]:
                        for score_item in score_list:
                            single_e_feature_vector.append(affect_dict[sessionid]['post'][score_item])
                        single_e_feature_vector.append(affect_dict[sessionid]['post']['timeonpage'])

                    else:
                        for l in range(3):
                            single_e_feature_vector.append(0.0)
                else:
                    for l in range(3):
                        single_e_feature_vector.append(0.0)


                for score_item in score_list:
                    feature_item_list.append(sessionid + '_' + sub_item + score_item)
                feature_item_list.append(sessionid + '_' + sub_item + '_timeonpage')

            elif sub_item == 'trial':
                if sessionid in trial_dict:
                    if 'first_try_correct' in trial_dict[sessionid]:
                        number_correctness = trial_dict[sessionid]['first_try_correct'].count('TRUE')
                        single_e_feature_vector.append(number_correctness)
                    else:
                        single_e_feature_vector.append(0)

                    if ('time_elapsed' in trial_dict[sessionid]) and (len(trial_dict[sessionid]['time_elapsed']) > 1):
                        time_elapsed_list = []
                        for i in range(len(trial_dict[sessionid]['time_elapsed'])):
                            if i == 0:
                                time_elapsed_list.append(trial_dict[sessionid]['time_elapsed'][i])
                            else:
                                time_elapsed_list.append(
                                    trial_dict[sessionid]['time_elapsed'][i] - trial_dict[sessionid]['time_elapsed'][i - 1])
                        if len(time_elapsed_list) > 0:
                            single_e_feature_vector.append(np.mean(time_elapsed_list))
                            single_e_feature_vector.append(np.std(time_elapsed_list))
                            single_e_feature_vector.append(
                                trial_dict[sessionid]['time_elapsed'][len(trial_dict[sessionid]['time_elapsed']) - 1])
                        else:
                            for l in range(3):
                                single_e_feature_vector.append(0)
                    else:
                        for l in range(3):
                            single_e_feature_vector.append(0)
                else:
                    for l in range(4):
                        single_e_feature_vector.append(0)

                feature_item_list.append(sessionid + '_' + sub_item + '_first_try_correct')
                feature_item_list.append(sessionid + '_' + sub_item + '_latency_time_mean')
                feature_item_list.append(sessionid + '_' + sub_item + '_latency_time_std')
                feature_item_list.append(sessionid + '_' + sub_item + '_timeonpage')


    return single_e_feature_vector