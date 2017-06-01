#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import reportlab.rl_config
reportlab.rl_config.warnOnMissingFontGlyphs = 0
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas
import calendar
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import fonts,colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate,Table,Image,PageTemplate,Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus.doctemplate import *

import reportlab.rl_config
reportlab.rl_config.warnOnMissingFontGlyphs = 0
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.ttfonts import TTFontFile, TTFError
from reportlab.pdfgen import canvas
import copy
from reportlab.platypus import Paragraph, SimpleDocTemplate, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import fonts
import os

pdfmetrics.registerFont(TTFont('song', '/Users/wangwei/workplace/font/SURSONG.TTF'))
pdfmetrics.registerFont(TTFont('hei', '/Users/wangwei/workplace/font/simhei.ttf'))

fonts.addMapping('song', 0, 0, 'song')
fonts.addMapping('song', 0, 1, 'song')
fonts.addMapping('song', 1, 0, 'hei')
fonts.addMapping('song', 1, 1, 'hei')

stylesheet=getSampleStyleSheet()
normalStyle = copy.deepcopy(stylesheet['Normal'])
normalStyle.fontName ='song'
normalStyle.fontSize = 12
story = []
story.append(Paragraph('<b>你好</b>,中文', normalStyle))
doc = SimpleDocTemplate('hello.pdf')
doc.build(story)


class ModelReport(object):
    class MyPageTemp(PageTemplate):
        '''
        定义一个页面模板
        '''
        def __init__(self):
            F6 = Frame(x1=0.5 * inch, y1=0.5 * inch, width=7.5 * inch, height=2.0 * inch, showBoundary=0)
            F5 = Frame(x1=0.5 * inch, y1=2.5 * inch, width=7.5 * inch, height=2.0 * inch, showBoundary=0)
            F4 = Frame(x1=0.5 * inch, y1=4.5 * inch, width=7.5 * inch, height=2.0 * inch, showBoundary=0)
            F3 = Frame(x1=0.5 * inch, y1=6.5 * inch, width=7.5 * inch, height=2.0 * inch, showBoundary=0)
            F2 = Frame(x1=0.5 * inch, y1=8.5 * inch, width=7.5 * inch, height=2.0 * inch, showBoundary=0)
            F1 = Frame(x1=0.5 * inch, y1=10.5 * inch, width=7.5 * inch, height=0.5 * inch, showBoundary=0)
            PageTemplate.__init__(self, "MyTemplate", [F1, F2, F3, F4, F5, F6])

        def beforeDrawPage(self, canvas, doc):  # 在页面生成之前做什么画logo
            pass

    def __init__(self, filename=None):
        self.filename = filename
        self.objects = []  ##story
        self.doc = BaseDocTemplate(self.filename)
        self.doc.addPageTemplates(self.MyPageTemp())
        self.Style = getSampleStyleSheet()
        # 设置中文,设置四种格式
        # self.nor=self.Style['Normal']
        self.cn = self.Style['Normal']
        self.cn.fontName = 'song'
        self.cn.fontSize = 9
        self.t = self.Style['Title']
        self.t.fontName = 'song'
        self.t.fontSize = 15
        self.h = self.Style['Heading1']
        self.h.fontName = 'song'
        self.h.fontSize = 10
        self.end = copy.deepcopy(self.t)
        self.end.fontSize = 7

    # 创建特征分析报告
    def _create_feature_report(self, df, corr_threshold=0.01, most_threshold=0.01):
        cols = ["Unnamed: 0", 'description', 'corr', "status",'max', 'min', 'mean', 'null', 'most_ratio', 'most_value', 'pvalue']
        story = []
        title = u"特征分析报告"
        story.append(Paragraph(title, self.t))
        table_data = []
        df = df[cols].rename(columns={"Unnamed: 0": "feature"})
        table_data.append(list(df.columns))
        for d in list(df.values):
            table_data.append(d)
        t = Table(table_data,
                  style=[('GRID', (0, 0), (-1, -1), 0.25, colors.black),
                        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                                    ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                                      ('FONTNAME', (0, 0), (-1, -1), 'song'),  # 字体
                                      ('FONTSIZE', (0, 0), (-1, -1), 8),
                                      ('BACKGROUND', (0, 0), (-1, 0), colors.lightskyblue),
                                      ('ALIGN', (-1, 0), (-2, 0), 'RIGHT')
                                     ],
                  )

        story.append(t)
        story.append(Spacer(0, 0))

        # 添加各特征的分析图
        story.append(Paragraph(u"数据特征的分布", self.t))
        df_image = self.load_feature_image()
        df_image = df_image.rename(columns={"feature":u"特征标识", "raw":u"原数据", "bin": u"离散数据"})
        table_images = []
        table_images.append(list(df_image.columns))
        for v in df_image.values:
            table_images.append(list(v))

        t = Table(table_images,
                  style=[('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                  ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                  ('FONTNAME', (0, 0), (-1, -1), 'song'),  # 字体
                  ('FONTSIZE', (0, 0), (-1, -1), 8),
                  ('BACKGROUND', (0, 0), (-1, 0), colors.lightskyblue),
                  ('ALIGN', (-1, 0), (-2, 0), 'RIGHT')]
                  )
        story.append(t)
        story.append(FrameBreak())
        # 添加模型的评估分析
        story.append(Paragraph(u"模型在原始数据的评估分析", self.t))

        # 原数据模型评估
        df_raw_image = self.load_evaluate_image()
        t = self.get_table_image(df_raw_image, fontsize=12, fontname="hei")
        story.append(t)


        # 离散数据模型评估
        story.append(FrameBreak())
        story.append(Paragraph(u"模型在离散数据的评估分析", self.t))
        story.append(Spacer(0, 0))
        df_raw_image = self.load_evaluate_image(data_name="discrete")
        t = self.get_table_image(df_raw_image, fontsize=12, fontname="hei")
        story.append(t)

        # 融合模型评估
        story.append(FrameBreak())
        story.append(Paragraph(u"融合模型评估分析", self.t))
        story.append(Spacer(0, 0))
        df_raw_image = self.load_evaluate_image(data_name="discrete:raw")
        t = self.get_table_image(df_raw_image, fontsize=12, fontname="hei")
        story.append(t)

        doc = SimpleDocTemplate('feature.pdf')
        doc.build(story)

    def get_table_image(self, df_image, fontsize=12, fontname="song"):
        table_images = []
        table_images.append(list(df_image.columns))
        for v in df_image.values:
            table_images.append(list(v))

        table = Table(table_images,
                  style=[('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                         ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                         ('FONTNAME', (0, 0), (-1, -1), fontname),  # 字体
                         ('FONTSIZE', (0, 0), (-1, -1), fontsize),
                         ('BACKGROUND', (0, 0), (-1, 0), colors.lightskyblue),
                         ('ALIGN', (-1, 0), (-2, 0), 'RIGHT'),
                         ('VALIGN', (0,-1),(-2,-1), 'MIDDLE')
                         ]
                  )

        return table


    def load_feature_image(self, data_name=None, feature_name=None):
        result = {}
        image_dir = "../result/visual/feature/image"

        for fp in os.listdir(image_dir):
            d = {}
            base_name = os.path.basename(fp)
            file_name, _ = os.path.splitext(base_name)
            feature_name_ = file_name.split("_")[0]
            feature_kind_ = file_name.split("_")[1]
            if feature_name and feature_name_ != feature_name:
                continue
            if not result.get(feature_name_):
                result[feature_name_] = {}
            img = Image(os.path.join(image_dir, base_name))
            img.drawHeight = 190
            img.drawWidth = 240
            d[feature_kind_] = img
            result[feature_name_].update(d)
        df = pd.DataFrame(result).T.reset_index().rename(columns={"index":"feature"})
        return df


    def load_evaluate_image(self, data_name="raw"):
        result = {}
        image_dir = "../result/visual/train"
        for fp in os.listdir(image_dir):
            d = {}
            base_name = os.path.basename(fp)
            file_name, _ = os.path.splitext(base_name)
            data_name_ = file_name.split("_")[0]
            clf_name_ = file_name.split("_")[1]
            eval_name = file_name.split("_")[2]
            if data_name_ != data_name:
                continue
            if not result.get(clf_name_):
                result[clf_name_] = {}
            img = Image(os.path.join(image_dir, base_name))
            img.drawHeight = 180
            img.drawWidth = 160
            d[eval_name] = img
            result[clf_name_].update(d)
        df = pd.DataFrame(result).T.reset_index().rename(columns={"index":"model"})
        print df.columns
        return df

if __name__ == "__main__":
    path = "../result/visual/feature/feature.csv"
    df = pd.read_csv(path)
    mr = ModelReport()
    mr._create_feature_report(df=df)
    # mr.load_feature_image()