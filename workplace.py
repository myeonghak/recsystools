import pandas as pd

import os
os.listdir(".")
os.chdir("Documents/LEE/recsys_package/my_package")

cols = ["user_id", "item_id", "rating", "timestamp"]
movie_data = pd.read_csv(
    "../ml-100k/u.data", names=cols, sep="\t", usecols=[0, 1, 2], engine="python")

movie_data.head(10)







