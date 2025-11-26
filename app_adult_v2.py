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
import pickle

# 标题,居中
# st.markdown("<h1 style='text-align: center; color: green;'>Predicting the risk of heart failure after non-cardiac surgery in patients</h1", unsafe_allow_html=True)
st.title('Predicting the risk of heart failure after non-cardiac surgery in adult patients')
is_ok =  True
left, right = st.columns(2)
with left:
    A = st.number_input('Age（years）', value=18)
    if A <18 or A>100:
        st.warning("You have entered an extreme value. Please confirm whether the value for this feature is correct.")
        is_ok =False
    B = st.number_input('Albumin（g/L）',value=100)
    if B<0 or B>200:
        st.warning("You have entered an extreme value. Please confirm whether the value for this feature is correct.")
        is_ok = False
    C = st.number_input('Neutrophil-to-lymphocyte ratio (NLR)', value=25)
    if C<0 or C>50:
        st.warning("You have entered an extreme value. Please confirm whether the value for this feature is correct.")
        is_ok = False
    F = st.number_input('Blood glucose（mmol/L）', value=25)
    if F<0 or F>50:
        st.warning("You have entered an extreme value. Please confirm whether the value for this feature is correct.")
        is_ok = False
    J = st.number_input('International Normalized Ratio（INR）', value=25)
    if J<0 or J>50:
        st.warning("You have entered an extreme value. Please confirm whether the value for this feature is correct.")
        is_ok = False
with right:
    H = st.number_input('Pulse rate（beats/min）', value=100)
    if H<0 or H>250:
        st.warning("You have entered an extreme value. Please confirm whether the value for this feature is correct.")
        is_ok = False
    L = st.number_input('MCHC（g/L）', value=500)
    if L<0 or L>1000:
        st.warning("You have entered an extreme value. Please confirm whether the value for this feature is correct.")
        is_ok = False
    D = st.number_input('Serum creatinine（μmol/L）', value=300)
    if D<0 or D>500:
        st.warning("You have entered an extreme value. Please confirm whether the value for this feature is correct.")
        is_ok = False
    E = st.number_input('eGFR（mL/min/1.73m²）', value=300)
    if E<0 or E>500:
        st.warning("You have entered an extreme value. Please confirm whether the value for this feature is correct.")
        is_ok = False
    K = st.number_input('Diastolic blood pressure（mmHg）', value=100)
    if K<1 or K>200:
        st.warning("You have entered an extreme value. Please confirm whether the value for this feature is correct.")
        is_ok = False




features_adult = ['AGE', '白蛋白-加', '中性粒细胞与淋巴细胞比值', '肌酐—加', 'eGFR', '血糖-加',
                  '脉搏','国际标准化比率', '血压Low', '平均血红蛋白浓度']

features_adult_en = ['Age', 'Albumin', 'NLR', 'Serum creatinine', 'eGFR', 'Blood glucose',
                  'Pulse',
                  'INR', 'Diastolic blood pressure', 'MCHC']

input_features = [A, B, C, D, E, F, H, J,K,L]

sample = np.array([x for x in input_features]).reshape(1, -1)
if is_ok:
    if st.button('Predict'):
        model = joblib.load('pt_st/medicalPredict20250320/model_audlt_v2.pkl')

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
        shap.initjs()
        shap.plots.force(
            explainer.expected_value[1],  # 类别 1 的基准值
            shap_values.values[0, :, 1],  # 类别 1 的 SHAP 值
            sample[0, :],  # 输入样本
            feature_names=features_adult_en,  # 特征名称
            matplotlib=True,
            show=False  # 不自动显示图形
        )
        st.pyplot(plt.gcf())
else:
    st.button('Predict',disabled=True)

st.write(
        "**Tips:**  The remaining numerical variable were the patient's most recent laboratory test before non-cardic surgery.")




