#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import bokeh
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from scipy import interp
mpl.style.use('ggplot')
from scipy.stats import pearsonr

font_size = 8 # 字体大小
fig_size = (4, 3) # 图表大小
mpl.rcParams['font.size'] = font_size   # 更改默认更新字体大小
mpl.rcParams['figure.figsize'] = fig_size

features = {
    'MP0045048': u"最近第5个月被叫联系人数",
    'MP0041080': u"最近1个月02至03点通话次数",
    'MP0110001': u"检查电话是否长时间关机",
    'MS0050011': u" ",
    'MP0050048': u'最近4个月异常时间（23~3点）通话次数占比',
    'mp_district_13': u"",
    'MP0045115': u'最近第5个月高频号码（top5）本地通话次数',
    'MP0110037': u'用户在异地的联系人数',
    'MP0042110': u'最近第2个月的单日最高通话次数',
    'MP0110027': u'通话记录中双向通话的平均时间间隔',
    'MP0110022': u'通话记录中单向被叫号码的个数',
    'MP0110010': u'与p2p小贷机构的通话次数',
    'MP0050003': u'最近1星期所在城市数量（排除常住地',
    'MP0110026': u'通话记录中双向通话时长',
    'mp_district_41': u'手机归属地为河南',
    'MP0050047': u'最近4个月手机月均单日最高通话率',
    'MP0110012': u'是否经常夜间活动',
    'MP0045110': u'最近第5个月的单日最高通话次数',
    'MP0110002': u'与110的通话次数',
    'MP0110044': u'平均通话时间大于2分钟的号码的个数',
    'MP0050028': u'',
    'MP0110024': u'通话记录中单向主叫通话次数',
    'MP0110045': u'平均通话时间小于1分钟的号码的个数',
    'MP0110009': u'与银行的通话时长',
    'MP0041053': u'最近1个月03至04点本地被叫次数',
    'MP0042067': u'最近第2个月03至04点漫游被叫次数',
    'MP0050018': u'最近1个月漫游主叫次数占比',
    'MP0110006': u'与澳门的通话次数',
    'MP0110004': u'与120的通话次数',
    'MP0110020': u'通话记录中双向通话号码的个数',
    'MP0110029': u'p2p类型电话双向通话次数',
    'MP0110011': u'与p2p小贷机构的通话时长',
    'MP0110023': u'通话记录中双向通话次数',
    'MS0050001': u'',
    'MP0110030': u'p2p类型电话单向主叫通话次数',
    'MP0110046': u'工作时间主叫电话个数',
    'MP0110021': u'通话记录中单向主叫号码个数'

}

# def get_data_curve(df):

def save_feature_csv():
    train_path = "../data/origin/train.csv"
    df = pd.read_csv(train_path)
    df_curve = get_feature_curve(df)
    df_curve.to_csv("../result/visual/feature/feature.csv", encoding="utf-8")
    # columns = list(df.drop(["uid", "label"], axis=1).columns)


def save_raw_feature_image():
    train_path = "../data/origin/train.csv"
    df = pd.read_csv(train_path)
    df_curve = get_feature_curve(df)
    df_ = df_curve[(df_curve["most_ratio"]<0.9)&(abs(df_curve["corr"])>0.01)]
    names = list(df_.index)
    if not os.path.exists("../result/visual/feature/image"):
        os.mkdir("../result/visual/feature/image")

    for name in names:
        plt.title(features.get(name) + ":")
        plt.ylim(ymax=df_.loc[name, "max"]+ df.shape[0]/4, ymin=0)

        plt_ = df[name].hist(bins=50, color="red")
        fig = plt_.get_figure()
        fig.savefig("../result/visual/feature/image/%s_raw.jpg"%name)


def save_discrete_feature_image():
    path = "../data/extend/M1/train_discrete.csv"
    df = pd.read_csv(path)
    df_curve = get_feature_curve(df)
    df_ = df_curve[(df_curve["most_ratio"] < 0.9) & (abs(df_curve["corr"]) > 0.01)]
    names = list(df_.index)
    if not os.path.exists("../result/visual/feature/image"):
        os.mkdir("../result/visual/feature/image")
    sd = float(sum(df["label"]==1.0))
    sg = float(sum(df["label"] == 0))
    df[names] = df[names].astype(int)
    for name in names:
        f_name = name.split("_")[0]
        plt.title(features.get(f_name) + ":")

        df1 = df[[name, "label"]].groupby(name)["label"]\
            .agg({"bad": lambda x: round(sum(x == 1)/sd, 4),
                  "good": lambda x: round(sum(x == 0)/sg, 4)
                  })
        df1 = df1.rename(columns={"good": u"未逾期占比", "bad": u"逾期占比"})
        plt_ = df1[[u"未逾期占比", u"逾期占比"]].plot.bar()
        fig = plt_.get_figure()
        fig.savefig("../result/visual/feature/image/%s_discrete.jpg" % name)


def save_evaluate_image():
    train_dir = "../M1/pred/train"
    test_dir = "../M1/pred/test"
    train_preds = sorted([os.path.basename(fp) for fp in os.listdir(train_dir)])
    test_preds = sorted([os.path.basename(fp) for fp in os.listdir(test_dir)])
    figure_num = 1
    for train_fp, test_fp in zip(train_preds, test_preds):
        data_name = train_fp.split("_")[0]
        clf_name = train_fp.split("_")[1]
        # if data_name != "raw" or data_name != "discrete":
        #     continue
        df_train = pd.read_csv(os.path.join(train_dir, train_fp))
        df_test = pd.read_csv(os.path.join(test_dir, test_fp))
        figure_num += 1
        save_roc_image(df_train, df_test, data_name, clf_name, figure_num)
        figure_num += 1
        save_ks_image(df_train, df_test, data_name, clf_name, figure_num)
        figure_num += 1
        save_badrate_image(df_train, df_test, data_name, clf_name, figure_num)




def save_roc_image(df_train, df_test, data_name, clf_name, figure_num):
    plt.figure(figure_num, figsize=(3,2))
    df_train_roc = get_auc_curve(df_train)
    df_test_roc= get_auc_curve(df_test)
    train_auc = round(auc(df_train_roc["fpr"], df_train_roc["tpr"]), 2)
    test_auc = round(auc(df_test_roc["fpr"], df_test_roc["tpr"]), 2)
    plt.plot([0, 1], [0, 1], '--', color="black")
    plt.plot(df_train_roc["fpr"].values, df_train_roc["tpr"].values, label="train auc:"+str(train_auc))
    plt.plot(df_test_roc["fpr"].values, df_test_roc["tpr"].values, label="test auc:"+str(test_auc))
    plt.legend(loc="lower right")
    plt.savefig("../result/visual/train/{0}_{1}_roc.jpg".format(data_name, clf_name))


def save_ks_image(df_train, df_test, data_name,clf_name, figure_num):
    plt.figure(figure_num, figsize=(3,2))
    df_train_ks = get_ks_curve(df_train, n_bins=10)
    max_train_ks = round(max(abs(df_train_ks["diff"].values)), 2)
    # print list(df_train_ks["bin"].values)

    print list(df_train_ks["diff"].values)
    df_test_ks = get_ks_curve(df_test, n_bins=10)
    max_test_ks = round(max(abs(df_test_ks["diff"]).values), 2)
    plt.plot(list(df_train_ks["bin"].values), list(df_train_ks["diff"].values), label="max train ks value:"+str(max_train_ks))
    plt.plot(list(df_test_ks["bin"].values), list(df_test_ks["diff"].values), label="max test ks value:" + str(max_test_ks))
    plt.legend(loc="lower left")
    plt.savefig("../result/visual/train/{0}_{1}_ks.jpg".format(data_name, clf_name))


def save_badrate_image(df_train, df_test, data_name, clf_name, figure_num):
    plt.figure(figure_num, figsize=(3,2))
    df_train_br = get_badrate_curve(df_train, n_bins=10)
    df_test_br = get_badrate_curve(df_test, n_bins=10)
    plt.plot(list(df_train_br["bin"].values), list(df_train_br["ratio"].values), label="train bad rate:")
    plt.plot(list(df_test_br["bin"].values), list(df_test_br["ratio"].values), label="test bad rate:")
    plt.legend(loc="upper right")
    plt.savefig("../result/visual/train/{0}_{1}_badrate.jpg".format(data_name, clf_name))



def get_feature_curve(df, id="uid", label="label"):
    feat_names = list(df.drop([id, label], axis=1).columns)
    df = pd.DataFrame(
        data={name: get_feature_info(df, name, label) for name in feat_names}
    )
    return df.T



def get_feature_info(df, feat_name, label_name):
    result = {}
    df_feat = df[feat_name]
    df_label = df[label_name]
    dc = df_feat.describe().to_dict()
    result["count"] = dc.get("count")
    result["std"] = round(dc.get("std"), 2)
    result["min"] = round(dc.get("min"), 2)
    result["max"] = round(dc.get("max"), 2)
    result["mean"] = round(dc.get("mean"), 2)
    result["null"] = 1 - round(df_feat.count()/float(df_feat.shape[0]), 2)
    md = df_feat.value_counts().head(1).to_dict().items()[0]
    result["most_value"] = round(md[0], 2)
    result["most_ratio"] = round(md[1]/float(df_feat.shape[0]), 2)
    df_feat = df_feat.fillna(df_feat.median())
    result["corr"] = round(pearsonr(df_feat.values, df_label.values)[0], 2)
    result["status"] = get_status_by_corr(result["corr"])
    result["pvalue"] = round(pearsonr(df_feat, df_label)[1], 2)
    result["description"] = features.get(feat_name)
    return result


def get_status_by_corr(corr):
    if 0.8 < corr <= 1.0:
        return u"极强相关"
    elif 0.6 < corr <= 0.8:
        return u"强相关"
    elif 0.4 < corr <= 0.6:
        return u"中等程度相关"
    elif 0.2 < corr <= 0.4:
        return u"弱相关"
    else:
        return u"无相关"


def get_data_visual():
    train_path = "../data/origin/train.csv"
    df = pd.read_csv(train_path)
    print df[df['mp_district_13']==0].count()


# 特征分布可视化
def visual_feature():
    pass


# 绘制auc曲线
def get_auc_curve(df):
    label = df["label"].values
    pred = df["score"].values
    df_auc = pd.DataFrame(columns=["fpr", "tpr", "threshold"])
    fpr, tpr, thresholds = roc_curve(label, pred)
    df_auc.fpr = fpr
    df_auc.tpr = tpr
    df_auc.threshold = thresholds
    return df_auc


# 绘制ks值得曲线
def get_ks_curve(df, n_bins):
    labels = n_bins - np.arange(n_bins)
    df["bin"] = pd.qcut(df["score"], n_bins, labels=labels)
    good = df[df["label"] == 0].shape[0]
    bad = df[df["label"] == 1].shape[0]
    df1 = df.groupby("bin")["label"].agg({
                                "tpr": lambda group: sum(group == 1)/float(bad),
                                "fpr": lambda group: sum(group == 0)/float(good)
                               }).sort_index(ascending=False)\
                                .cumsum()\
                                .reset_index()
    df1["diff"] = df1["tpr"] - df1["fpr"]
    df1["diff"] = df1["diff"].apply(lambda x: round(x,  5))
    return df1


# 绘制badrate的曲线
def get_badrate_curve(df, n_bins):
    labels = n_bins - np.arange(n_bins)
    df["bin"] = pd.qcut(df["score"], n_bins, labels=labels)

    df_rate = df.groupby("bin").apply(lambda x: sum(x["label"] == 1) / float(x["label"].shape[0]))\
        .to_frame(name="ratio").sort_index(ascending=False).reset_index()
    print("df rate is:")
    print df_rate
    return df_rate


if __name__ == "__main__":
    from scipy.stats import boxcox

    from pandas.tools.plotting import table
    test_pred_path = "../M1/pred/test/discrete:raw_xgb_1.csv"
    df = pd.read_csv(test_pred_path)
    label = df.label
    pred = df.score

    # df_bdr = get_badrate_curve(df, n_bins=10)
    # df_bdr_ = df_bdr[["bin", "ratio"]].set_index("bin")
    # plot_ = df_bdr_.plot()
    # fig = plot_.get_figure()
    # fig.savefig("./bad.jpg")
    # plt.show()

    # df_ks = get_ks_curve(df, n_bins=10)
    # df_ks_ = df_ks[["diff", "bin"]].set_index("bin")
    # df_ks_.plot()
    # plt.show()
    # train_path = "../data/extend/M1/train_raw.csv"
    train_path = "../data/origin/train.csv"
    df = pd.read_csv(train_path)
    # df_ = df[['MP0045048', 'MP0110027']]
    # df_ = df[df["label"]==0]['MP0110011']
    # print((df_.value_counts()/float(df.shape[0])).to_dict())
    # print sum(df[df['MP0110011']==0]["label"]==0)/float(sum(df_==0))
    # print df_.value_counts().head(1)
    # feat_names = list(df.drop(['label', 'uid'], axis=1).columns)
    # df[feat_names] = df[feat_names].diff()
    # df = df.fillna(df.median())
    # df_result = get_feature_curve(df)
    # df[feat_names].plot(subplots=True, layout=(2, 3), sharex=False)
    # print df_result
    # print(df_result["corr"].max())
    # df_result["most_ratio"].plot.kde()
    # df_result["corr"].hist(bins=50, label="most ratio is")
    # plt.show()
    # sel_cols =  list(df_result[df_result["most_ratio"]>0.9].index)
    # for col in sel_cols:
    #     print features.get(col)
    # plot_ = df_.diff().hist(bins=100)
    # plt.show()
    # nrows = 4
    # ncols = int(len(feat_names) / nrows) + 1
    #
    # for i in range(nrows):
    #     fig, axes = plt.subplots(nrows, ncols)
    #     for j in range(ncols):
    #         name_ = feat_names[i*4 +j]
    #         df[name_].hist(ax=axes[i,j]); axes[i,j].set_title(name_)
    #
    # plt.show()
    # save_feature()
    # save_discrete_feature_image()
    save_evaluate_image()
