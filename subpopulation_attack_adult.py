# importing necessary packages

import pickle
import os
import numpy as np
import pandas as pd
from sklearn import neural_network, linear_model, cluster

np.random.seed(0) # seeding 0 for random number generation

def import_adults():
    # importing the adult dataset, it is split into train and test by default
    a = pd.read_csv('adult/adult.data', header=None,names=['age', 'workclass', 'fnlwgt', 'education',
                       'education-num', 'marital-status', 'occupation', 'relationship',
                       'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                       'native-country', 'income']) # dim: (32561 x 15)

    b = pd.read_csv('adult/adult.test', header=None,
                names=['age', 'workclass', 'fnlwgt', 'education',
                       'education-num', 'marital-status', 'occupation', 'relationship',
                       'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                      'native-country', 'income'])# dim: (16282, 15)
    return a, b

def preprocess_data(total):
    ### Data Preprocessing ###
    # converting the pandas dataframe to numpy array
    total_np = total.to_numpy()

    # D has 7841 samples, D_aux has 7841 samples, and D_test has 7692 samples

    # taking '>50K' income as y values
    y = (total_np[:, -1] + total_np[:, -2]).astype(np.float32) # last two columns are duplicate, so add them 
    y = np.delete(y, 32561, axis=0)                            # delete the 32561th row value for y as it has NaN values

    # taking rest of the data as x values, after deleting the last three columns i.e columns
    x = np.delete(total_np, [total_np.shape[1]-1, total_np.shape[1]-2, total_np.shape[1]-3], axis=1)
    x = np.delete(x, 32561, axis=0).astype(np.float32)         # delete the 32561th row value for x as it has NaN vlaues


    #separating the dataset into training and testing, training [0-32560] row, testing [32561-rest]
    train_x, train_y = x[:32561], y[:32561]     # dim: (32561 x 57)
    test_x, test_y = x[32561:], y[32561:]       # dim: (16281 x 57)

    # saving the index of samples where y is true(1) and false(0)
    train_zero_inds = np.where(train_y==0)[0]   # dim: (24720 x 1)
    train_one_inds = np.where(train_y==1)[0]    # dim: (7841 x 1)
    test_zero_inds = np.where(test_y==0)[0]     # dim: (12435 x 1)
    test_one_inds = np.where(test_y==1)[0]      # dim: (3846 x 1)

    # creating an array of random numbers, range[0-24720(train_zero_inds.shape[0])], of dimension (7841(train_one_inds.shape[0]), 1) 
    train_zeros = np.random.choice(train_zero_inds.shape[0], train_one_inds.shape[0], replace=False) # dim:(7841 x 1)

    # creating an array of random numbers, range[0-12435(test_zero_inds.shape[0])], of dimension (3846(test_one_inds.shape[0]), 1)
    test_zeros = np.random.choice(test_zero_inds.shape[0], test_one_inds.shape[0], replace=False) # dim: (3846 x 1)


    # concatenating random choices of zero indexed example with one indexed example to build a dataset with 
    # equal number of zero and one indexed samples
    train_x = np.concatenate((train_x[train_zeros], train_x[train_one_inds]), axis=0) # dim: (15682 x 57)
    train_y = np.concatenate((train_y[train_zeros], train_y[train_one_inds]), axis=0) # dim: (15682 x 1)


    test_x = np.concatenate((test_x[test_zeros], test_x[test_one_inds]), axis=0)      # dim: (7692 x 57)
    test_y = np.concatenate((test_y[test_zeros], test_y[test_one_inds]), axis=0)      # dim: (7692 x 1)


    # shuffle training data(row wise) by shuffling the index and getting the data from shuffled index
    train_shuffle = np.random.choice(train_x.shape[0], train_x.shape[0], replace=False) # dim: (15682 x 1)
    train_x, train_y = train_x[train_shuffle], train_y[train_shuffle]                   # dim: (15682 x 57) and (15682 x 1)

    train_size = train_x.shape[0]//2

    train_x, train_y, ho_x, ho_y = train_x[:train_size], train_y[:train_size], train_x[train_size:], train_y[train_size:]

    # D: (train_x, train_y)
    # D_aux : (ho_x, ho_y)
    # D_test: (test_x, test_y)
    print(train_x.shape, train_y.shape, ho_x.shape, ho_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, ho_x, ho_y, test_x, test_y

def run_simulation():
    # get adult dataset
    a, b = import_adults()

    # concatenate the data to preprocess the whole dataset according to our need
    total = pd.concat([a, b], axis=0) # dim: (488843, 15)

    # dropping education, native country, fnlwgt as mentioned in the paper (three columns)
    total = total.drop('education', axis=1)
    total = total.drop('native-country', axis=1)
    total = total.drop('fnlwgt', axis=1)

    # dim: (488843, 12)
    # print(total['income'])

    # adding new columns for columns with categorical variables
    for col in ['workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'income']:
        if '-' in col:
            prefix_col = col.split('-')[0]
        else:
            prefix_col = col
        
        # converting categorical varaibles into indicator variables
        total = pd.concat([total, pd.get_dummies(total[col], prefix=prefix_col, drop_first=True)], axis=1)
        total = total.drop(col, axis=1)

    # dim: (488843, 60)

    print(total.shape)
    print(list(total.columns))
    #total.tail()

    ### Data Preprocessing:
    pois_rates = [0.5, 1, 2]     # poison rates
    #pois_rates = [0.5, 1, 1.5, 2, 2.5, 3]
    
    train_x, train_y, ho_x, ho_y, test_x, test_y = preprocess_data(total)

    ### Train Clean LR and NN
    # training and testing the linear regression model
    lm = linear_model.LogisticRegression(max_iter = 5000)
    lm.fit(train_x, train_y)
    lm.score(test_x, test_y)

    # saving the trained model as a pickel file
    save_path = 'subpopulation/'

    with open(os.path.join(save_path,"lm.pickle"), "wb") as f:
        pickle.dump(lm,f)

    # # loading the trained model 
    # with open(os.path.join(save_path,"lm.pickle"), "rb") as f:
    #     test = pickle.load(f)

    # print(lm.classes_) # two classes '0' & '1'

    # predict the output probability for ho_x dataset. predict_proba gives probability for each class i.e a tuple 
    # in this case [0.1 0.9] probaibility for this class. np.eye(2) gets [1 0] when y = 0 and [0 1] when y = 1
    # which when muliplied singles out the probability
    lm_preds = np.multiply(lm.predict_proba(ho_x), np.eye(2)[ho_y.astype(np.int)]).sum(axis=1) # dim: (7841, 1)

    # could have simply done predict
    lm_pred_class = lm.predict(ho_x)                    # dim: (7841, 1)

    print(lm_preds) # print the probabilites but has no information on which class it belongs to
    print(lm_pred_class)  # print the predictions, gives classes

    # training and test the neural network with 10 hidden units
    nn = neural_network.MLPClassifier(hidden_layer_sizes=(10,))
    nn.fit(train_x, train_y)
    clean_test_acc = nn.score(test_x, test_y)
    print(clean_test_acc)

    # prediction for the nn
    # probability of belonging to a class

    nn_preds = np.multiply(nn.predict_proba(ho_x), np.eye(2)[ho_y.astype(np.int)]).sum(axis=1) 
    nn_pred_class = nn.predict(ho_x) # prediction class

    print(nn_preds)
    print(nn_pred_class)

    pred_mean = np.multiply(nn.predict_proba(test_x), np.eye(2)[test_y.astype(int)]).mean()*2
    print(pred_mean)

    ################################################
    ###### Feature Match ~ Derived from original paper
    ################################################

    # FeatureMatch Data Preprocessing
    np.random.seed(0)
    # concatenating education level and race for D_aux data
    protected = np.concatenate((ho_x[:, 12:27], ho_x[:, 52:57]), axis=1) # dim: (7841, 20)

    # concatenating education level and race for D_test data
    test_prot = np.concatenate((test_x[:, 12:27], test_x[:, 52:57]), axis=1) # dim: (7692, 20)

    # concatenating education level and race for D_train data
    train_prot = np.concatenate((train_x[:, 12:27], train_x[:, 52:57]), axis=1) # dim: (7841, 20)

    all_cols = list(total.columns)                # getting names of all columns
    prot_cols = all_cols[12:27] + all_cols[52:57] # getting names of protected columns

    subclasses, counts = np.unique(protected, axis=0, return_counts=True)# 122 unique examples in the protected data 
    # dim: (122 x 20), (122,1)
    # print(tuple(zip(subclasses, counts)))
    # print(subclasses[0:5], counts[0:5])

    hd_sbcl_conf = []                             # empty array
    hd_used = []                                  # empty array

    # declaring arrays for storing errors in the subclasses, i.e [subclasses, (clean_acc, collat, target), pois_ind]

    #  dim: (122 x 3 x len: pois rates) for a class (3 x pois len) dimensional array
    hd_lr_errs = np.zeros((len(subclasses), 3, len(pois_rates)))
    #  dim: (122 x 3 x len: pois rates) for a class (3 x 3 pois len) dimensional array
    hd_nn_errs = np.zeros((len(subclasses), 3, len(pois_rates)))

    # Feature Match: Initially there are 122 subclasses of common features, later if senctences filters the subclasses 
    # to number to 35

    for i, (subcl, count) in enumerate(zip(subclasses, counts)):  # for each subclass in subclasses
        if count > 10 and count < 100:                            # if the number of counts is more than 10 and less than 100
            hd_used.append((i, count))                            # mark if by apppend to hd_used
            
            print("\n")
            print("Subclass Index: %d, Subclass Count: %d " % (i, count)) # print subclass index and count
        
            # subtract the current subclass from test_prot data, find frobenius norm along columns and then get index of
            # data where the norm is still zero. This finds the samples in test_prot (D_test) which have identical features 
            # to current subclass
            test_sbcl = np.where(np.linalg.norm(test_prot - subcl, axis=1)==0) # 2D array of (indexes, datatypes)
            
            # same logic as above, finding the samples in protected data (D_aux) which have identical features to current subclass
            
            sbcl = np.where(np.linalg.norm(protected-subcl, axis=1)==0)   # 2D array of (indexes, datatypes) Note: number of index
                                                                        # should be equal to counts value
            
            # same logic as above, finding the samples in protected data (D_train) which have identical features to current subclass
            train_sbcl = np.where(np.linalg.norm(train_prot - subcl, axis=1)==0) # 2D array of (indexes, datatypes)
            
            # getting the samples with idential feature (to current subclass) from test data (D_test)
            p_t_x, p_t_y = test_x[test_sbcl], test_y[test_sbcl] 
            
            # getting the samples with idential feature (to current subclass) from auxiliary data (D_aux)
            # labelling it as poison data
            pois_x_base, pois_y_base = ho_x[sbcl], ho_y[sbcl]  
            
            # getting the prediction probability of identical samples from(D_aux), and finding their mean
            sc_lr_pred, sc_nn_pred = lm_preds[sbcl].mean(), nn_preds[sbcl].mean()
            print("Sc_lr_pred: %f, Sc_nn_pred: %f " % (sc_lr_pred, sc_nn_pred))
            
            train_ct = train_sbcl[0].shape[0] # number of identical samples in train_sbcl (D_train)
            test_ct = p_t_x.shape[0]          # number of identical samples in test_sbcl (D_test) 
            
            # multiplying prediction probability [x y] of a sample with [0 1] or [1 0] based on y values
            # taking their sum(mean) and multiplying with 2
            hd_sbcl_conf.append(2*np.multiply(lm.predict_proba(p_t_x), np.eye(2)[p_t_y.astype(int)]).mean())
            
            #all_errs = []

            # multiply poison rates with number of identical training samples, then take it as integer, dim same as pois rate, 1D array
            # pois_ct now has the number of poisoned samples which needs to be added to D_train
            # for each number(pois_ct) in the pois_rates 
            for j, pois_ct in enumerate([int(train_ct*pois_rate) for pois_rate in pois_rates]):

                # get the random indexes of 'pois_ct' number of samples from pois_x_base with replacement (repeated data is allowed)
                pois_inds = np.random.choice(pois_x_base.shape[0], pois_ct, replace=True)

                # get the x_values and flip the y values. Now we have the poison data
                pois_x, pois_y = pois_x_base[pois_inds], 1 - pois_y_base[pois_inds]

                # add the poisoned data to the training data
                total_x, total_y = np.concatenate((train_x, pois_x), axis=0), np.concatenate((train_y, pois_y), axis=0)

                # print the column name for which the value (v > 0.5) is 1 basically, then print the number of samples
                # in training, poisoned, and testing data. This finally prints which features we are poisoning
                
                print([prot_col for v, prot_col in zip(subcl, prot_cols) if v > 0.5], train_ct, pois_ct, test_ct)
                print("poison fraction:", pois_ct/train_ct) # printing the poison fraction

                # creating three Logistic regression model so that we could train three models and average their results
                lmps = [linear_model.LogisticRegression(solver='liblinear', max_iter=5000) for _ in range(3)] 
                for lmp in lmps:
                    lmp.fit(total_x, total_y)

                # creating three Logistic regression model so that we could train three models on the Poisoned Data (total_x, total_y)
                nnps = [neural_network.MLPClassifier(hidden_layer_sizes=(10,), max_iter=3000) for _ in range(3)]
                for nnp in nnps:
                    nnp.fit(total_x, total_y)

                # get the average accuracy score on test data (D_test) for three models of LR
                lmp_acc_colla = np.mean([lmp.score(test_x, test_y) for lmp in lmps])
                print("Poisoned lr test acc {:.3f}".format(lmp_acc_colla))

                # get the average accuracy score on test data (D_test) for three models of NN
                nnp_acc_colla = np.mean([nnp.score(test_x, test_y) for nnp in nnps])
                print("Poisoned nn test acc {:.3f}".format(nnp_acc_colla))

                # checking the score for identical samples(p_t_x, p_t_y) in test data on Clean LR model
                lmc_sbc = lm.score(p_t_x, p_t_y)
                print("lr clean sbc {:.3f}".format(lmc_sbc))

                # checking the score for identical samples(p_t_x, p_t_y) in test data on Poisoned LR model
                lmp_sbc_itest = np.mean([lmp.score(p_t_x, p_t_y) for lmp in lmps])
                print("lr poisoned sbc {:.3f}".format(lmp_sbc_itest))

                # checking the score for identical samples(p_t_x, p_t_y) in test data on Clean NN model
                nnc_sbc = nn.score(p_t_x, p_t_y)
                print("nn clean  sbc {:.3f}".format(nnc_sbc))

                # checking the score for identical samples(p_t_x, p_t_y) in test data on Clean NN model
                nnp_sbc_itest = np.mean([nnp.score(p_t_x, p_t_y) for nnp in nnps])
                print("nn poisoned sbc {:.3f}".format(nnp_sbc_itest))

                # storing the errors as (subclass, (row0: clean_acc, row1: collat, row2: target), (pois_ind))
                hd_lr_errs[i, 0, j] = lmc_sbc
                hd_lr_errs[i, 1, j] = lmp_acc_colla
                hd_lr_errs[i, 2, j] = lmp_sbc_itest

                hd_nn_errs[i, 0, j] = nnc_sbc
                hd_nn_errs[i, 1, j] = nnp_acc_colla
                hd_nn_errs[i, 2, j] = nnp_sbc_itest
    
    ################################################
    ###### Cluster Match ~ Derived from original paper
    ################################################
    #Cluster match Data preprocessing

    km = cluster.KMeans(n_clusters=100)   # KMeans with 100 clusters
    km.fit(ho_x)                          # fit the ho_x (Data_aux)

    # dim of cluster_centers_ : (100, 57)

    test_km = km.predict(test_x)          # predict the cluster centers for test dataset (D_test), (7692 x 1)

    train_km = km.predict(train_x)        # predict the cluster centers for test dataset (D_train), (7841 x 1)

    kd_sbcl_conf = []
    kd_used = []

    # declaring arrays for storing errors in the subclasses, i.e [subclasses, (clean_acc, collat, target), pois_ind]

    #  dim: (122 x 3 x len: pois rates) for a class (3 x pois len) dimensional array
    kd_lr_errs = np.zeros((len(subclasses), 3, len(pois_rates))) 

    #  dim: (122 x 3 x len: pois rates) for a class (3 x pois len) dimensional array
    kd_nn_errs = np.zeros((len(subclasses), 3, len(pois_rates))) 

    kmeans_designed = []
    cl_inds, cl_cnts = np.unique(km.labels_, return_counts=True) # cl_inds has cluster center index, and count in a cluster 
                                                                # given by cl_ctns
    print("cluster indexes: ", cl_inds)
    print("cluster count: ", cl_cnts)

    # ClusterMatch: Initially there are 100 clusters, later if sentence filters the clusters 28 
    for i, (cl_ind, cl_ct) in enumerate(zip(cl_inds, cl_cnts)): # for each cluster in D_aux data
        if cl_ct > 10 and cl_ct < 100:                          # select the cluster if it has more than 10 and less than 100 samples
            kd_used.append((i, cl_ct))                          # append the selected cluster
            
            print("\n")
            print("Cluster Index: %d, Cluster Count: %d, Test Samples: %d" % (cl_ind, cl_ct, np.where(test_km==cl_ind)[0].shape[0])) # print current cluster index, count and number of 
                                                                        # samples in test data belonging to current cluster index

            # getting the indexes of test samples that belong to current cluster index
            test_sbcl = np.where(test_km==cl_ind)

            # getting the indexes of aux data samples that belong to current cluster index
            sbcl = np.where(km.labels_==cl_ind)

            # getting the indexes of training samples that belong to current cluster index
            train_sbcl = np.where(train_km==cl_ind)

            # getting the test samples that belong to current cluster index
            p_t_x, p_t_y = test_x[test_sbcl], test_y[test_sbcl]

            # getting the aux samples that belong to current cluster index which is too be poisoned
            pois_x_base, pois_y_base = ho_x[sbcl], ho_y[sbcl]

            # getting the prediction probability of identical samples from(D_aux), and finding their mean
            sc_lr_pred, sc_nn_pred = lm_preds[sbcl].mean(), nn_preds[sbcl].mean()
            print(sc_lr_pred, sc_nn_pred)


            train_ct = train_sbcl[0].shape[0] # number of train samples that match current cluster index
            test_ct = p_t_x.shape[0]          # number of test samples that match current cluster index
            
            # multiplying prediction probability [x y] of a sample with [0 1] or [1 0] based on y values
            # taking their sum(mean) and multiplying with 2
            kd_sbcl_conf.append(2*np.multiply(lm.predict_proba(p_t_x),np.eye(2)[p_t_y.astype(int)]).mean())
            
            #all_errs = []
            
            # multiply poison rates with number of identical training samples, then take it as integer, dim same as pois rate, 1D array
            # pois_ct now has the number of poisoned samples which needs to be added to D_train
            # for each number(pois_ct) in the pois_rates 
            for j, pois_ct in enumerate([int(train_ct*pois_rate) for pois_rate in pois_rates]):
                
                # get the random indexes of 'pois_ct' number of samples from pois_x_base with replacement (repeated data is allowed)
                pois_inds = np.random.choice(pois_x_base.shape[0], pois_ct, replace=True)
                
                # get the x_values and flip the y values. Now we have the poison data
                pois_x, pois_y = pois_x_base[pois_inds], 1 - pois_y_base[pois_inds]
                
                # add the poisoned data to the training data
                total_x, total_y = np.concatenate((train_x, pois_x), axis=0), np.concatenate((train_y, pois_y), axis=0)
                
                print("poison fraction:", pois_ct/train_ct, train_ct, pois_ct, test_ct)
                
                # creating three Logistic regression model so that we could train three models and average their results
                lmps = [linear_model.LogisticRegression(solver='liblinear', max_iter=500) for _ in range(3)]
                for lmp in lmps:
                    lmp.fit(total_x, total_y)
                
                # creating three Logistic regression model so that we could train three models on the Poisoned Data (total_x, total_y)
                nnps = [neural_network.MLPClassifier(hidden_layer_sizes=(10,), max_iter=3000) for _ in range(3)]
                for nnp in nnps:
                    nnp.fit(total_x, total_y)
                    
                # get the average accuracy score on test data (D_test) for three models of LR
                lmp_acc_col = np.mean([lmp.score(test_x, test_y) for lmp in lmps])
                print("Poisoned lr test acc {:.3f}".format(lmp_acc_col))
                
                # get the average accuracy score on test data (D_test) for three models of NN
                nnp_acc_col = np.mean([nnp.score(test_x, test_y) for nnp in nnps])
                print("Poisoned nn test acc {:.3f}".format(nnp_acc_col))
                
                # checking the score for identical samples(p_t_x, p_t_y) in test data on Clean LR model
                lmc_sbc = lm.score(p_t_x, p_t_y)
                print("lr clean sbc {:.3f}".format(lmc_sbc))
                
                # checking the score for identical samples(p_t_x, p_t_y) in test data on Poisoned LR model
                lmp_sbc_itst = np.mean([lmp.score(p_t_x, p_t_y) for lmp in lmps])
                print("lr poisoned sbc {:.3f}".format(lmp_sbc_itst))
                
                # checking the score for identical samples(p_t_x, p_t_y) in test data on Clean NN model
                nnc_sbc = nn.score(p_t_x, p_t_y)
                print("nn cl sbc {:.3f}".format(nnc_sbc))
                
                # checking the score for identical samples(p_t_x, p_t_y) in test data on Clean NN model
                nnp_sbc_itst = np.mean([nnp.score(p_t_x, p_t_y) for nnp in nnps])
                print("nn sbc {:.3f}".format(nnp_sbc_itst))
                
                # storing the errors as (subclass, (row0: clean_acc, row1: collat, row2: target), (pois_ind))
                kd_lr_errs[i, 0, j] = lmc_sbc
                kd_lr_errs[i, 1, j] = lmp_acc_col
                kd_lr_errs[i, 2, j] = lmp_sbc_itst
                
                kd_nn_errs[i, 0, j] = nnc_sbc
                kd_nn_errs[i, 1, j] = nnp_acc_col
                kd_nn_errs[i, 2, j] = nnp_sbc_itst
    
    # number of filters for FeatureMatch and ClusterMatch
    print(len(hd_used), len(kd_used))
    hd_lr_errs  # storing the errors as (subclass, (row0: clean_acc, row1: collat, row2: target), (pois_ind))
    hd_nn_errs  # storing the errors as (subclass, (row0: clean_acc, row1: collat, row2: target), (pois_ind))

    # computing error rates for FetureMatch, indexing of hd_used and hd_targets are same
    hd_targets = np.zeros((len(hd_used), 3, 2)) # dim: (len(hd filter: 35) X 3(pois_ind) X 2). storing nn_errors

    for j, ((i, count), conf) in enumerate(zip(hd_used, hd_sbcl_conf)): # for each subclasses(filter) in FeatureMatch
        for pois_ind in range(3):
            #print(i, count, conf, pois_ind)
            this_err = hd_nn_errs[i, :, pois_ind] # pick pois_ind column of (3 x pois_ind) matrix
            clean_acc, collat, target = this_err[0], this_err[1], this_err[2]
            
            # target is poisoned acc - clean acc for identical test samples in a subclass
            hd_targets[j, pois_ind, 0] = clean_acc - target  
            hd_targets[j, pois_ind, 1] = collat  # collat is poisoned acc on test dataser(D_test)

            
    # computing error rates for ClusterMatch

    # input: clean_nn test accuracy, poisoned_nn collat, frac of test samples belonging to current 
    #        cluster, cleann_nn cluster test samples, poisoned_nn cluster test samples
    def compute_collat(acc_before, acc_after, subpop_frac, pre_subpop, post_subpop):
        other_acc_before = (acc_before - subpop_frac*pre_subpop)/(1-subpop_frac)
        other_acc_after = (acc_after - subpop_frac*post_subpop)/(1-subpop_frac)
        return other_acc_after - other_acc_before

    # indexing of kd_used and kd_targets are same
    kd_targets = np.zeros((len(kd_used), 3, 2)) # dim: (len(kd filter: 28) X 3 X 2)

    for j, ((i, count), conf) in enumerate(zip(kd_used, kd_sbcl_conf)): # for each cluster(filter) in ClusterMatch
        # get the indexes of test samples which have current cluster index as its cluster index
        this_inds = np.where(test_km==i)[0] 
        size = this_inds.size                 # number of test samples that belong to this cluster
        
        #     pre_acc = nn.score(test_x[this_inds], test_y[this_inds]) # score of test samples belonging to current cluster given
        #                                                              # clean nn
        #     print(pre_acc, size, pre_acc*size)    # print accuracy score, size, and their multiplication
        for pois_ind in range(3):
            #print(i, count, conf, pois_ind)
            this_err = kd_nn_errs[i, :, pois_ind]      # pick pois_ind column of (3 x pois_ind) matrix
            clean_acc, collat, target = this_err[0], this_err[1], this_err[2]
            
            # target is poisoned acc - clean acc for identical test samples in a cluster
            kd_targets[j, pois_ind, 0] = clean_acc - target
            # compute collat by giving clean_nn test accuracy, poisoned_nn collat, frac of test samples belonging to current 
            # cluster, cleann_nn cluster test samples, poisoned_nn cluster test samples
            kd_targets[j, pois_ind, 1] = compute_collat(clean_test_acc, collat, size/test_x.shape[0], clean_acc, target)
    
    ## print target metric for FeatureMatch
    # for i in range(len(hd_targets)):
    #     for poi_ind in range(3):
    #         if poi_ind == 0:
    #             print("Index: %d, Poison ratio: % d, target: %f " %(i, poi_ind, hd_targets[i, poi_ind, 0]))

    # sorting the error rate for FeatureMatch based on target metric

    for pois_ind in range(3): # for each pois_ind/pois frac
        print("\n")
        
        # print the poison index and poison rate
        print("Pois Index: %d, Pois fraction: %f of identical training samples " %( pois_ind, pois_rates[pois_ind]))
        
        # sort the target metric scores for current poison index
        sorted_hd_targets = np.argsort(hd_targets[:, pois_ind, 0])
        
        # take the last index of sorted_hd_targets as it is the index with highest target score for current poison index
        top1_index = sorted_hd_targets[-1]
        
        #     print("Hd error/used Index: ", top1_index) # print the top1_index

        # get the subclasses index, count for the feature with the highest target score using top1_index, 
        # the key is hd_target and hd_used have same indexing, so top1_index which is index in hd_targets is also the same index
        # in hd_used, then hd_used has the index and count of the subclass to which the highest target score belongs to
        subclasses_ind, count = hd_used[top1_index]
        print("Subclass Index: %d, Count: %d" % (subclasses_ind, count))
        
        subclass = subclasses[subclasses_ind] # get the subclass to which the highest score belongs to
        print([prot_col for v, prot_col in zip(subclass, prot_cols) if v > 0.5]) # print the subclass
        print(subclass)
            
        # print its target and collat values
        print("Target: %f, Collat: %f" % (hd_targets[top1_index, pois_ind, 0], hd_targets[top1_index, pois_ind, 1]))
    
    # # print target metric for ClusterMatch

    # for i in range(len(kd_targets)):
    #     for poi_ind in range(3):
    #         if poi_ind == 0:
    #             print("Index: %d, Poison ratio: % d, target: %f " %(i, poi_ind, kd_targets[i, poi_ind, 0]))

    # sorting the error rate for ClusterMatch based on target metric

    for pois_ind in range(3): # for each pois_ind/pois frac
        print("\n")
        
        # print the poison index and poison rate
        print("Pois Index: %d, Pois fraction: %f of identical training samples " %( pois_ind, pois_rates[pois_ind]))
        
        # sort the target metric scores for current poison index
        sorted_kd_targets = np.argsort(kd_targets[:, pois_ind, 0])
        
        # take the last index of sorted_kd_targets as it is the index with highest target score for current poison index
        top1_ind = sorted_kd_targets[-1]
        
    #     print("kd error/used Index: ", top1_index) # print the top1_index

        # get the cluster index, count for the feature with the highest target score using top1_index, 
        # the key is kd_target and kd_used have same indexing, so top1_index which is index in kd_targets is also the same index
        # in kd_used, then kd_used has the index and count of the cluster to which the highest target score belongs to
        cluster_ind, count = kd_used[top1_ind]
        print("Cluster Index: %d, Count: %d" % (cluster_ind, count))
        
        # get the index of training samples that belong to that cluster with highest target
        train_examp_ind = np.where(train_km==cluster_ind)[0]    # has indexes of such samples
        train_examp_rand_ind = np.random.choice(train_examp_ind.shape[0], 3, replace= False) # pick five random such samples
        for i in train_examp_rand_ind:
            example = train_x[i]        # print the examples in the cluster
            print(example)
            print([prot_col for v, prot_col in zip(example, all_cols) if v > 0.5]) # print the examples in cluster
            
        # print its target and collat values
        print("Target: %f, Collat: %f" % (kd_targets[top1_ind, pois_ind, 0], kd_targets[top1_ind, pois_ind, 1]))

if __name__ == "__main__":
    run_simulation()