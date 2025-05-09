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
st.title('Predicting the  risk of heart failure after non-cardiac surgery in geriatric patients')

warning = 'You have entered an extreme value. Please confirm whether the value for this feature is correct.'
is_pass = True
left, right = st.columns(2)
with left:
    E = st.selectbox('Coronary heart disease (Yes or No)', ['No', 'Yes'])
    if E == 'Yes':
        E = 1
    else:
        E = 0
    D = st.number_input('Neutrophil-to-lymphocyte ratio (NLR)', value=0.5)
    if D < 0.1 or D > 25:
        st.error(warning)
        is_pass = False
    A = st.number_input('Age（years）', value=65)
    if A < 65 or A > 100:
        st.error(warning)
        is_pass = False
    B = st.number_input('Albumin（g/L）', value=20)
    if B < 10 or B > 100:
        st.error(warning)
        is_pass = False
    C = st.number_input('Pulse rate（beats/min）', value=20)
    if C < 20 or C > 200:
        st.error(warning)
        is_pass = False
    F = st.number_input('International Normalized Ratio（INR）', value=0.5)
    if F < 0.5 or F > 10:
        st.error(warning)
        is_pass = False
    G = st.number_input('Diastolic blood pressure（mmHg）', value=20)
    if G < 20 or G > 150:
        st.error(warning)
        is_pass = False

with right:
    H = st.number_input('eGFR（mL/min/1.73m²）', value=0.5)
    if H <= 0:
        st.error(warning)
        is_pass = False
    M = st.selectbox('Severe anemia (Yes or No)', ['No', 'Yes'])
    if M == 'Yes':
        M = 1
    else:
        M = 0
    J = st.number_input('Serum creatinine（umol/L）', value=50)
    if J < 10 or J > 1500:
        st.error(warning)
        is_pass = False
    K = st.number_input('BMI（kg/m²）', value=5)
    if K < 5 or K > 50:
        st.error(warning)
        is_pass = False
    L = st.number_input('White blood cell count（× 10⁹/L）', value=0.5)
    if L < 0.5 or L > 150:
        st.error(warning)
        is_pass = False
    I = st.number_input('Systolic blood pressure（mmHg）', value=40)
    if I < 40 or I > 250:
        st.error(warning)
        is_pass = False
    N = st.number_input('MCHC（g/l）', value=50)
    if N < 20 or N > 500:
        st.error(warning)
        is_pass = False


features_adult = ['AGE', '白蛋白-加', '脉搏','中性粒细胞与淋巴细胞比值','冠心病','国际标准化比率','血压Low', 'eGFR', '血压high','肌酐—加','BMI','白细胞计数', '严重贫血', '平均血红蛋白浓度']

features_adult_en = ['Age', 'Albumin', 'Pulse', 'NLR','Coronary heart disease', 'INR', 'Diastolic blood pressure', 'eGFR','Systolic blood pressure','Serum creatinine', 'BMI','White blood cell count', 'Severe anemia', 'MCHC']

input_features = [A, B, C, D, E, F, G, H, I, J, K, L,M,N]


sample = np.array([x for x in input_features]).reshape(1, -1)

if is_pass:
    if st.button('Predict'):
        model = joblib.load(r'/demo20250509/model_old2.pkl')

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        # shap_values = explainer(sample)

        force_plot_data = shap.force_plot(explainer.expected_value[1], shap_values[0, :, 1], features_adult_en)
        f_x = force_plot_data.data['outValue']
        # print("force_plot 中的 f(x):", f_x)

        # prediction_proba = model.predict_proba(sample)
        # prediction_value = model.predict(sample)[0]
        st.subheader("The risk of heart failure in this  geriatric patient（≥65） after non-cardiac surgery is " + str(round(f_x, 3)))
        # st.write("Probability（class 0, class 1）：", prediction_proba)

        st.subheader("SHAP Explanation")
        plt.figure()
        # shap.initjs()
        shap.plots.force(
            explainer.expected_value[1],  # 类别 1 的基准值
            shap_values[0, :, 1],  # 类别 1 的 SHAP 值
            sample[0, :],  # 输入样本
            feature_names=features_adult_en,  # 特征名称
            matplotlib=True,
            show=False  # 不自动显示图形
        )
        st.pyplot(plt.gcf())

st.write(
    "**Tips:**  \n1.Severe anemia is defined as hemoglobin < 60 g/L  \n2. The remaining numerical variable were the patient's most recent laboratory test before non-cardic surgery.")