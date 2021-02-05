import copy
import pandas as pd 
from IPython.display import display

sub_3  = pd.read_csv("submissions/submission_id_3.txt",  sep = ',', names = ["id", "lat", "long"])
sub_6  = pd.read_csv("submissions/submission_id_6.txt",  sep = ',', names = ["id", "lat", "long"])
sub_7  = pd.read_csv("submissions/submission_id_7.txt",  sep = ',', names = ["id", "lat", "long"])
sub_8  = pd.read_csv("submissions/submission_id_8.csv",  sep = ',', names = ["id", "lat", "long"])
sub_9  = pd.read_csv("submissions/submission_id_9.txt",  sep = ',', names = ["id", "lat", "long"])
sub_10 = pd.read_csv("submissions/submission_id_10.txt", sep = ',', names = ["id", "lat", "long"])
sub_11 = pd.read_csv("submissions/submission_id_11.txt", sep = ',', names = ["id", "lat", "long"])
sub_12 = pd.read_csv("submissions/submission_id_12.txt", sep = ',', names = ["id", "lat", "long"])
sub_13 = pd.read_csv("submissions/submission_id_13.txt", sep = ',', names = ["id", "lat", "long"])
sub_14 = pd.read_csv("submissions/submission_id_14.txt", sep = ',', names = ["id", "lat", "long"])
sub_15 = pd.read_csv("submissions/submission_id_15.txt", sep = ',', names = ["id", "lat", "long"])
sub_16 = pd.read_csv("submissions/submission_id_16.txt", sep = ',', names = ["id", "lat", "long"])

sub_3  = sub_3[1:]
sub_6  = sub_6[1:]
sub_7  = sub_7[1:]
sub_8  = sub_8[1:]
sub_9  = sub_9[1:]
sub_10 = sub_10[1:]
sub_11 = sub_11[1:]
sub_12 = sub_12[1:]
sub_13 = sub_13[1:]
sub_14 = sub_14[1:]
sub_15 = sub_15[1:]
sub_16 = sub_16[1:]


latitude = pd.DataFrame()
latitude["submission_id_3"]  = sub_3["lat"].astype(float)
latitude["submission_id_6"]  = sub_6["lat"].astype(float)
latitude["submission_id_7"]  = sub_7["lat"].astype(float)
latitude["submission_id_8"]  = sub_8["lat"].astype(float)
latitude["submission_id_9"]  = sub_9["lat"].astype(float)
latitude["submission_id_10"] = sub_10["lat"].astype(float)
latitude["submission_id_11"] = sub_11["lat"].astype(float)
latitude["submission_id_12"] = sub_12["lat"].astype(float)
latitude["submission_id_13"] = sub_13["lat"].astype(float)
latitude["submission_id_14"] = sub_14["lat"].astype(float)
latitude["submission_id_15"] = sub_15["lat"].astype(float)
latitude["submission_id_16"] = sub_16["lat"].astype(float)

longitude = pd.DataFrame()
longitude["submission_id_3"]  = sub_3["long"].astype(float)
longitude["submission_id_6"]  = sub_6["long"].astype(float)
longitude["submission_id_7"]  = sub_7["long"].astype(float)
longitude["submission_id_8"]  = sub_8["long"].astype(float)
longitude["submission_id_9"]  = sub_9["long"].astype(float)
longitude["submission_id_10"] = sub_10["long"].astype(float)
longitude["submission_id_11"] = sub_11["long"].astype(float)
longitude["submission_id_12"] = sub_12["long"].astype(float)
longitude["submission_id_13"] = sub_13["long"].astype(float)
longitude["submission_id_14"] = sub_14["long"].astype(float)
longitude["submission_id_15"] = sub_15["long"].astype(float)
longitude["submission_id_16"] = sub_16["long"].astype(float)

display(latitude.corr())
display(longitude.corr())

blending = copy.deepcopy(sub_3)
blending["lat"]  = sub_16["lat"].astype(float)  * 0.7 + sub_3["lat"].astype(float)  * 0.3 + sub_6["lat"].astype(float)  * 0.0 + sub_7["lat"].astype(float)  * 0.0
blending["long"] = sub_16["long"].astype(float) * 0.7 + sub_3["long"].astype(float) * 0.3 + sub_6["long"].astype(float) * 0.0 + sub_7["long"].astype(float) * 0.0


blending.to_csv("blending_18.txt", index = False)