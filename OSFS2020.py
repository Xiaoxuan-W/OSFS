import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
import random
from scipy.sparse import diags
from sklearn.neighbors import kneighbors_graph as kng

def compare(list1,list2):
    return len(set(list1) & set(list2))/len(list1)

def arrfs_score(X_in):
    M=np.array(X_in)
    features=X_in.columns.values
    MAD_list=[]
    Score_list=[]
    S=cosine_similarity(M.transpose())
    for i in range(0,M.shape[1]):      
        Score_list.append(MAD_list[i]/(np.sum(S[i])))
        
    zipped=zip(features, Score_list)
    sort_zipped = sorted(zipped,key=lambda x:(x[1]))
    result = zip(*sort_zipped)
    arrfs_f_rank, Score = [list(x) for x in result]
    Score.reverse()
    arrfs_f_rank.reverse()
    return arrfs_f_rank

def arrfs_produce(X,eta,st): #run osfs with arrfs
    # X: timeseries data set; eta: similarity threshold; st: random start point
    k_array=[4,16,64,256]
    for a in k_array:
        fea_temp1=arrfs_score(X[st:st+8])[0:a]
        fea_temp2=arrfs_score(X[st:st+16])[0:a]
        sim_temp=compare(fea_temp2,fea_temp1)
        tk_array=[32,64,128,256,512,1024]
        for b in tk_array:
            fea_now=arrfs_score(X[st:st+b])[0:a]
            sim_now=compare(fea_temp2,fea_now)
            if sim_now<sim_temp and sim_temp>eta:
                return fea_temp1,a,b/4
            elif sim_now>eta and b==1024:
                return fea_temp2,a,b/2
            else:
                fea_temp1=fea_temp2
                fea_temp2=fea_now
                sim_temp=sim_now
    return arrfs_score(X[st:st+1024])[0:256],256,1024

def Prepare(X_lp,tk):
    if tk<=16:
        kn=2
    elif tk<=128:
        kn=5
    else:
        kn=10        
    S=kng(X_lp, kn, mode='distance',metric='euclidean')
    S=S.A
    for i in range(0,len(X_lp)):
        for j in range(0,len(X_lp)):
            if(S[i][j]!=0):
                S[i][j]=math.exp(-math.pow(S[i][j],2))
    S=np.matrix(S)
    D = np.array(S.sum(axis=1))
    D = diags(np.transpose(D), [0])
    L=D.A-S.A
    D=D.A
    I_t=np.ones(len(X_lp))
    I= I_t.reshape(I_t.shape[0],1)
    return I,I_t,D,L    

def Laplacian_Score(f_col_t,I,I_t,D,L):   
    f_col=f_col_t.reshape(f_col_t.shape[0],1)
    F=f_col-np.matmul(f_col_t,np.matmul(D,I))/(np.matmul(I_t,np.matmul(D,I)))*I
    F_t= F.ravel()
    LS=(np.matmul(F_t,np.matmul(L,F)))/np.matmul(F_t,np.matmul(D,F))
    return LS[0]

def ls_score(X_in,tk):
    M=np.array(X_in) 
    features=X_in.columns.values
    I,I_t,D,L=Prepare(M,tk)
    ls_list=[]
    for i in range(0,M.shape[1]):
        ls_list.append(Laplacian_Score(M[:,i],I,I_t,D,L))
    zipped=zip(features, ls_list)
    sort_zipped = sorted(zipped,key=lambda x:(x[1]))
    result = zip(*sort_zipped)
    ls_f_rank, ls = [list(x) for x in result]
    return ls_f_rank

def ls_produce(X,leta,st):#run osfs with ls
    # X: timeseries data set; leta: similarity threshold; st: random start point
    k_array=[4,16,64,256]
    for a in k_array:
        fea_temp1=ls_score(X[st:st+8],8)[0:a]
        fea_temp2=ls_score(X[st:st+16],16)[0:a]
        sim_temp=compare(fea_temp2,fea_temp1)
        tk_array=[32,64,128,256,512,1024]
        for b in tk_array:
            fea_now=ls_score(X[st:st+b],b)[0:a]
            sim_now=compare(fea_temp2,fea_now)
            if sim_now<sim_temp and sim_temp>leta:
                return fea_temp1,a,b/4
            elif sim_now>leta and b==1024:
                return fea_temp2,a,b/2
            else:
                fea_temp1=fea_temp2
                fea_temp2=fea_now
                sim_temp=sim_now
    return ls_score(X[st:st+1024],1024)[0:256],256,1024

def tb_score(X_in,Y_in):
    features=X_in.columns.values
    clf = RandomForestRegressor(n_estimators=100)
    clf = clf.fit(X_in, Y_in)
    feature_importance=clf.feature_importances_
    rf_zipped=zip(features,feature_importance)
    sort_rf_zipped = sorted(rf_zipped,key=lambda x:(x[1]))
    rf_result = zip(*sort_rf_zipped)
    rf_f_rank, fea_im = [list(x) for x in rf_result]
    rf_f_rank.reverse()
    return rf_f_rank

def tb_produce(X,Y,teta,st):#run osfs with tree based selection, supervised method
    # X: timeseries data set; Y: timeseries target; teta: similarity threshold; st: random start point
    k_array=[4,16,64,256]
    for a in k_array:
        fea_temp1=tb_score(X[st:st+8],Y[st:st+8])[0:a]
        fea_temp2=tb_score(X[st:st+16],Y[st:st+16])[0:a]
        sim_temp=compare(fea_temp2,fea_temp1)
        tk_array=[32,64,128,256,512,1024]
        for b in tk_array:
            fea_now=tb_score(X[st:st+b],Y[st:st+b])[0:a]
            sim_now=compare(fea_temp2,fea_now)
            if sim_now<sim_temp and sim_temp>teta:
                return fea_temp1,a,b/4
            elif sim_now>teta and b==1024:
                return fea_temp2,a,b/2
            else:
                fea_temp1=fea_temp2
                fea_temp2=fea_now
                sim_temp=sim_now
    return tb_score(X[st:st+1024],Y[st:st+1024])[0:256],256,1024