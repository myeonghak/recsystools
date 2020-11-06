# 소수 샘플 제거하기
# 유니크 고객 및 상품 정리
# sparsity check
# pivot 생성하기


# train-test split
# hold-out
# demeaning

import pandas as pd
import numpy as np

import pandas as pd
import os


os.chdir("/home/aithe/Documents/LEE/recsys_package/my_package")


cols = ["user_id", "item_id", "rating", "timestamp"]
movie_data = pd.read_csv(
    "../ml-100k/u.data", names=cols, sep="\t", usecols=[0, 1, 2], engine="python")


pd.pivot_table(values=)



class preproInteractions():

    def __init__(self, data, item_col, user_col):
        self.data_original=data
        self.item_col=item_col
        self.user_col=user_col

        # for computational efficiency, cut off columns that are not necessary for the matrix factorization process.
        self.data=data[[item_col,user_col,]]

        self.user_list=self.data[item_col].unique().tolist()
        self.item_list=self.data[user_col].unique().tolist()



    # 해당 칼럼의 그룹 사이즈를 출력
    def get_count_per(self, data, id_):
        count_groupbyid = data[[id_]].groupby(id_, as_index=False)
        count = count_groupbyid.size()
        return count


    def filter_rare_cases(self, min_user_cnt=5, min_item_cnt=5):
        # filter out items or users that have less interactions than min_user_cnt and min_item_cnt
        data=self.data.copy()

        self.user_cnt=self.get_count_per(data, self.user_col)
        self.item_cnt=self.get_count_per(data, self.item_col)

        if min_user_cnt > 0 :
            data = data[data[self.user_col].isin(self.user_cnt.index[self.user_cnt >= min_user_cnt])]

        if min_item_cnt > 0 :
            data = data[data[self.item_col].isin(self.item_cnt.index[self.item_cnt >= min_item_cnt])]

        print(f"updated data after filtering out items with less than {min_item_cnt} and users with less than {min_user_cnt}.")
        print(f"original shape of the data was {self.data.shape}, now it is {data.shape}")

        self.data=data

        return True


    def check_sparsity(self):
        shape=self.data.shape
        total_interactions=self.data.sum().sum()
        matrix_shape=shape[0]*shape[1]

        print(f"sparsity is: {round((total_interactions/matrix_shape)*100,2)} %")

        return True


    def make_matrix(self):
        # convert the interaction records into a matrix that has shape of (m x n), where m is the number of users and n is the number of items.
        # only for small matrices that has less than 1m interactions
        
        data=self.data.copy()
        pivot=pd.pivot_table(data,values=None,index=[self.user_col],columns=[self.item_col],aggfunc="count",fill_value=0)
        self.matrix=pivot.values

        return True

    def 


        



    def 


