print(__doc__)

from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

random_seed = 1225

#set data path
train_path = "../data/train_raw.csv"
test_path = "../data/test_raw.csv"

#load data
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

train_X = df_train.drop(["label", "uid"], axis=1).values
train_y = df_train["label"].values
test_X = df_test.drop(["label", "uid"], axis=1).values
test_y = df_test["label"].values

X, val_X, y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=1)#random_state is of big influence for val-auc

#xgboost start here
dtest = xgb.DMatrix(test_X)
dval = xgb.DMatrix(val_X,label=val_y)
dtrain = xgb.DMatrix(X, label=y)

params={
	'booster':'gbtree',
	'objective': 'binary:logistic',
	'early_stopping_rounds':100,
	'scale_pos_weight': 1500.0/13458.0,
    'eval_metric': 'auc',
	'gamma':0.1,#0.2 is ok
	'max_depth':8,
	'lambda':550,
    'subsample':0.7,
    'colsample_bytree': 0.3,
    'min_child_weight': 2.5,
    'eta': 0.007,
	'seed':random_seed,
	'nthread':7
    }

# watchlist = [(dtrain,'train'),(dval,'val')] #The early stopping is based on last set in the evallist
model = xgb.train(params, dtrain, num_boost_round=50000)
model.save_model('./model/xgb.model')
print "best best_ntree_limit",model.best_ntree_limit   #did not save the best,why?

#predict test set (from the best iteration)
pred_y = model.predict(dtest, ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=["uid","score"])
test_result.uid = df_test.uid
test_result.score = pred_y
auc_score = roc_auc_score(test_y, pred_y)
print("auc score is:", auc_score)
test_result.to_csv("xgb.csv",index=None,encoding='utf-8')  #remember to edit xgb.csv , add ""


#save feature score and feature information:  feature,score,min,max,n_null,n_gt1w
feature_score = model.get_fscore()
for key in feature_score:
    feature_score[key] = [feature_score[key]]+feature_info[key]+[features_type[key]]

feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1},{2},{3},{4},{5},{6}\n".format(key,value[0],value[1],value[2],value[3],value[4],value[5]))

with open('feature_score.csv','w') as f:
    f.writelines("feature,score,min,max,n_null,n_gt1w\n")
    f.writelines(fs)


def load_dataset():
    df_train_rank = pd.read_csv




