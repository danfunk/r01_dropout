import numpy as np

from feature_generation import file_read_and_feature_extract, mindtrails_feature_vector_generation, \
    templeton_feature_vector_generation

def model_prediction_testing():
    prediction_session_index = 2
    platform_list = ['mindtrails', 'templeton']

    for platform in platform_list:
        if platform == 'mindtrails':
            session_list = ['PRE', 'SESSION1', 'SESSION2', 'SESSION3', 'SESSION4', 'SESSION5', 'SESSION6', 'SESSION7',
                            'SESSION8']

            demographic_dict, QOL_dict, OASIS_dict, RR_dict, BBSIQ_dict, DASS21_AS_dict, DASS21_DS_dict, trial_dict, \
            dwell_time_dict, session_completion_dict, dropout_label, control_normal_dict = file_read_and_feature_extract(
                platform)

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

        elif platform == 'templeton':
            session_list = ['preTest', 'firstSession', 'secondSession', 'thridSession', 'fourthSession']

            demographic_dict, affect_dict, credibility_dict, mental_dict, whatibelieve_dict, relatability_dict, \
            expectancy_dict, phq4_dict, trial_dict, session_completion_dict, dropout_label = file_read_and_feature_extract(
                platform)

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
        print 'dataset ==>', platform
        print 'number of participants for predict session ==>', prediction_session, len(feature_vector)
        print 'feature dimension ==>', len(feature_vector[0]), len(feature_item_list)

        X = range(len(feature_vector))
        kf = KFold(n_splits=10, random_state=None, shuffle=True)
        kf.get_n_splits(X)

        f1_score_svm_list = []
        f1_score_lr_list = []
        f1_score_rf_list = []
        f1_score_multi_svm_list = []
        for train_index, test_index in kf.split(X):
            data_train = []
            truth_train = []
            data_test = []
            truth_test = []

            participant_list_train = []
            participant_list_test = []

            for i in train_index:
                data_train.append(feature_vector[i])
                truth_train.append(truth_vector[i])
                participant_list_train.append(participant_list[i])
            for i in test_index:
                data_test.append(feature_vector[i])
                truth_test.append(truth_vector[i])
                participant_list_test.append(participant_list[i])

            svm_model = svm.LinearSVC(C=1, tol=1e-3)
            svm_model.fit(data_train, truth_train)

            lr_model = linear_model.LogisticRegression()
            lr_model.fit(data_train, truth_train)

            rf_model = RandomForestClassifier(
                n_estimators=10, criterion="gini", max_features="auto", max_depth=2, min_samples_split=2,
                min_samples_leaf=1, random_state=0, bootstrap=True, min_weight_fraction_leaf=0.0,
                n_jobs=1, oob_score=False, verbose=0, warm_start=False
            )
            rf_model.fit(data_train, truth_train)


            svm_prediction = svm_model.predict(data_test)
            lr_prediction = lr_model.predict(data_test)
            rf_prediction = rf_model.predict(data_test)

            truth_prediction_svm = []
            truth_prediction_lr = []
            truth_prediction_rf = []
            truth_test_new = []
            for i in range(len(svm_prediction)):
                truth_prediction_svm.append(int(svm_prediction[i]))
                truth_prediction_lr.append(int(lr_prediction[i]))
                truth_prediction_rf.append(int(rf_prediction[i]))
                truth_test_new.append(int(truth_test[i]))

            fscore_svm = metrics.f1_score(truth_test_new, truth_prediction_svm, average='micro')
            fscore_lr = metrics.f1_score(truth_test_new, truth_prediction_lr,  average='micro')
            fscore_rf = metrics.f1_score(truth_test_new, truth_prediction_rf, average='micro')

            f1_score_svm_list.append(fscore_svm)
            f1_score_lr_list.append(fscore_lr)
            f1_score_rf_list.append(fscore_rf)

            multi_svm_model = svm_multi_task_training(data_train, truth_train, participant_list_train, demographic_dict)

            truth_prediction_multi_svm = []
            truth_test_new = []
            for i in range(len(data_test)):
                participant_id = participant_list_test[i]
                testing_feature_vector = data_test[i]
                prediction_value = svm_multi_task_prediction(testing_feature_vector, demographic_dict, participant_id, multi_svm_model)

                truth_prediction_multi_svm.append(prediction_value)
                truth_test_new.append(int(truth_test[i]))
            fscore_multi_svm = metrics.f1_score(truth_test_new, truth_prediction_multi_svm, average='micro')
            f1_score_multi_svm_list.append(fscore_multi_svm)


        mean_f1score_svm = np.mean(f1_score_svm_list)
        std_f1score_svm = np.std(f1_score_svm_list)

        mean_f1score_lr = np.mean(f1_score_lr_list)
        std_f1score_lr = np.std(f1_score_lr_list)

        mean_f1score_rf = np.mean(f1_score_rf_list)
        std_f1score_rf = np.std(f1_score_rf_list)

        mean_f1score_multi_svm = np.mean(f1_score_multi_svm_list)
        std_f1score_multi_svm = np.std(f1_score_multi_svm_list)

        print 'prediction_session ==>', prediction_session
        print 'SVM classifier ==>                ', 'f1 score mean', mean_f1score_svm, 'f1 score std', std_f1score_svm
        print 'Logisitc Regression classifier ==>', 'f1 score mean', mean_f1score_lr, 'f1 score std', std_f1score_lr
        print 'Random Forest classifier ==>      ', 'f1 score mean', mean_f1score_rf, 'f1 score std', std_f1score_rf
        print 'Multi-SVM classifier ==>          ', 'f1 score mean', mean_f1score_multi_svm, 'f1 score std', std_f1score_multi_svm
        print '\n'

model_prediction_testing()
