#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/20 19:36
# @Author  : Ken
# @Software: PyCharm
import joblib
import pandas as pd
import sklearn
import shap
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import os



# 标题,居中
# st.markdown("<h1 style='text-align: center; color: green;'>Predicting the risk of heart failure after non-cardiac surgery in patients</h1", unsafe_allow_html=True)
st.title('Predicting the risk of heart failure after non-cardiac surgery in adult patients')

warning = 'You have entered an extreme value. Please confirm whether the value for this feature is correct.'
is_pass = True
left, right = st.columns(2)
with left:
    B = st.number_input('Albumin（g/L）', value=20)
    if B<10 or B>100:
        st.error(warning)
        is_pass = False
    A = st.number_input('Age（years）', value=18)
    if A<18 or A>100:
        st.error(warning)
        is_pass = False
    E = st.number_input('eGFR（mL/min/1.73m²）', value=0.5)
    if E<=0:
        st.error(warning)
        is_pass = False
    C = st.number_input('Neutrophil-to-lymphocyte ratio (NLR)', value=0.5)
    if C<0.1 or C>25:
        st.error(warning)
        is_pass =False
    D = st.number_input('Serum creatinine（μmol/L）', value=10)
    if D<10 or D>1500:
        st.error(warning)
        is_pass =False
    J = st.number_input('International Normalized Ratio（INR）', value=0.5)
    if J<0.5 or J>10:
        st.error(warning)
        is_pass =False

with right:
    G = st.number_input('Absolute lymphocyte count（× 10⁹/L）', value=0.1)
    if G <0.1 or G>20:
        st.error(warning)
        is_pass =False
    H = st.number_input('Pulse rate（beats/min）', value=20)
    if H <20 or H>200:
        st.error(warning)
        is_pass =False
    F = st.number_input('Blood glucose（mmol/L）', value=1.0)
    if F<1 or F>35:
        st.error(warning)
        is_pass =False
    L = st.number_input('MCHC（g/L）', value=20)
    if L<20 or L>500:
        st.error(warning)
        is_pass =False
    I = st.number_input('Absolute neutrophil count（× 10⁹/L）', value=0.5)
    if I<0.2 or I>150:
        st.error(warning)
        is_pass =False
    K = st.number_input('Diastolic blood pressure（mmHg）', value=20)
    if K<20 or K>150:
        st.error(warning)
        is_pass =False

features_adult = ['AGE', '白蛋白-加', '中性粒细胞与淋巴细胞比值', '肌酐—加', 'eGFR', '血糖-加', '淋巴细胞绝对值',
                  '脉搏',
                  '中性粒细胞绝对值', '国际标准化比率', '血压Low', '平均血红蛋白浓度']

features_adult_en = ['Age', 'Albumin', 'NLR', 'Serum creatinine', 'eGFR', 'Blood glucose', 'Absolute lymphocyte count',
                  'Pulse',
                  'Absolute neutrophil count', 'INR', 'Diastolic blood pressure', 'MCHC']

input_features = [A, B, C, D, E, F, G, H, I, J, K, L]

sample = np.array([x for x in input_features]).reshape(1, -1)

if is_pass:
    if st.button('Predict'):
        model = joblib.load('model_audlt.pkl')
    
        explainer = shap.Explainer(model)
        shap_values = explainer(sample)
    
        force_plot_data = shap.force_plot(explainer.expected_value[1], shap_values.values[0, :, 1], features_adult_en)
        f_x = force_plot_data.data['outValue']
        print("force_plot 中的 f(x):", f_x)
    
        prediction_proba = model.predict_proba(sample)
        prediction_value = model.predict(sample)[0]
        st.subheader("The risk of heart failure in this patient after non-cardiac surgery is " + str(round(f_x,3)))
        # st.write("Probability（class 0, class 1）：", prediction_proba)
    
        st.subheader("SHAP Explanation")
        plt.figure()
        # shap.initjs()
        shap.plots.force(
            explainer.expected_value[1],  # 类别 1 的基准值
            shap_values.values[0, :, 1],  # 类别 1 的 SHAP 值
            sample[0, :],  # 输入样本
            feature_names=features_adult_en,  # 特征名称
            matplotlib=True,
            show=False  # 不自动显示图形
        )
        st.pyplot(plt.gcf())

st.write(
        "**Tips:**  \nThe remaining numerical variable were the patient's most recent laboratory test before non-cardic surgery.")




