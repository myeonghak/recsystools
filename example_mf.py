import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings(action='ignore')

import matplotlib.pyplot as plt
import sklearn

from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

from recsystools import *

print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"scipy version: {scipy.__version__}")
print(f"sklearn version: {sklearn.__version__}")
print(f"bottleneck version: {bn.__version__}")


user_purch_list=pd.read_csv("ecommerce_data.csv", encoding = 'ISO-8859-1')

user_code="CustomerID"
item_code="StockCode"
idx_colname="InvoiceNo"

user_purch_list = user_purch_list.loc[user_purch_list['Quantity'] > 0]
user_purch_list = user_purch_list.loc[user_purch_list['UnitPrice'] > 0]
user_purch_list = user_purch_list.dropna(subset=[user_code])
user_purch_list[user_code]=user_purch_list[user_code].astype(int)

# 필터링을 마친 새로운 raw data와, 유저의 활동 로그, 아이템의 판매 기록을 각각 저장
raw_data, user_activity, item_popularity = filter_triplets(user_purch_list)

# 새로운 raw data로 interaction matrix 계산
pivot=pd.pivot_table(raw_data, values=idx_colname,index=[user_code],columns=[item_code],aggfunc="count",fill_value=0)

# sparsity는?
sparsity = 1-( 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0]))

print("After filtering, there are %d buying events from %d users and %d items (sparsity: %.3f%%)" % 
      (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))



# 유니크한 유저 아이디를 저장
unique_uid = user_activity.index

# 균일하게 섞어주기 위해 shuffle
np.random.seed(34)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]


# 유저의 수를 나누어, train/test를 split함
# 500명의 유저를 hold, 추천 모델 성능 test 용으로 사용

n_users = unique_uid.size
n_heldout_users = 500

tr_users = unique_uid[:(n_users - n_heldout_users)]
te_users = unique_uid[(n_users - n_heldout_users):]


tr_df=raw_data[raw_data[user_code].isin(tr_users)]
te_df=raw_data[raw_data[user_code].isin(te_users)]

te_tr,te_te=split_train_test_proportion(te_df)
holdout_user_list=te_te[user_code].unique()

# 전체 train 데이터
train_data=pd.concat([tr_df,te_tr])

train_pivot=pd.pivot_table(train_data, values=idx_colname,index=[user_code],columns=[item_code],aggfunc="count",fill_value=0)


matrix=train_pivot.values

#demean mean

user_ratings_mean=np.mean(matrix,axis=1)
ui_matrix=matrix-user_ratings_mean.reshape(-1,1)


start=time.time()

U, sig , Vt =svds(ui_matrix,k=200)

print(f"training time: {time.time()-start}s")

sig=np.diag(sig)

svd_user_predicted_ratings=np.dot(np.dot(U,sig),Vt) + user_ratings_mean.reshape(-1,1)


df_svd_preds=pd.DataFrame(svd_user_predicted_ratings, columns=train_pivot.columns).T
df_svd_preds.columns=train_pivot.index


df_svd_preds_exclude_purchase=df_svd_preds-(1e+10*(train_pivot.T))

pred_svd=np.array(df_svd_preds_exclude_purchase.T[df_svd_preds_exclude_purchase.columns.isin(holdout_user_list)])


holdout_svd=scipy.sparse.csr_matrix(pivot[pivot.T.columns.isin(holdout_user_list)].values)

# k=200
print(f"NDCG at 10, k = 200: " , NDCG_binary_at_k_batch(pred_svd,holdout_svd,10).mean())
print(f"Recall at 10, k = 200 :",Recall_at_k_batch(pred_svd,holdout_svd,10).mean())



