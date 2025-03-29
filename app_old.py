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


# 标题,居中
# st.markdown("<h1 style='text-align: center; color: green;'>Predicting the risk of heart failure after non-cardiac surgery in patients</h1", unsafe_allow_html=True)
st.title('Predicting the risk of heart failure after non-cardiac surgery in geriatric patients')

warning = 'You have entered an extreme value. Please confirm whether the value for this feature is correct.'
is_pass = True
left, right = st.columns(2)
with left:
    E = st.selectbox('Coronary heart disease (Yes or No)', ['No', 'Yes'])
    if E == 'Yes':
        E = 1
    else:
        E = 0
    A = st.number_input('Age（years）',value=65)
    if A<65 or A>100:
        st.error(warning)
        is_pass = False
    F = st.number_input('Albumin（g/L）', value =20)
    if F<10 or F>100:
        st.error(warning)
        is_pass = False
    D = st.number_input('Neutrophil-to-lymphocyte ratio (NLR)', value = 0.5)
    if D<0.1 or D>25:
        st.error(warning)
        is_pass = False
    H = st.number_input('International Normalized Ratio（INR）', value = 0.5)
    if H<0.5 or H>10:
        st.error(warning)
        is_pass = False
    B = st.number_input('Pulse rate（beats/min）', value = 20)
    if B<20 or B>200:
        st.error(warning)
        is_pass = False
        
with right:
    G = st.number_input('eGFR（mL/min/1.73m²）', value = 5)
    if G<=0:
        st.error(warning)
        is_pass = False
    C = st.number_input('Absolute neutrophil count（× 10⁹/L）', value=0.5)
    if C<0.2 or C>150:
        st.error(warning)
        is_pass = False
    J = st.number_input('Diastolic blood pressure（mmHg）', value =20)
    if J<20 or J>150:
        st.error(warning)
        is_pass = False
    I = st.number_input('Serum creatinine（μmol/L）', value=10)
    if I<10 or I>1500:
        st.error(warning)
        is_pass = False
    K = st.number_input('BMI kg/m²', value=5)
    if K<5 or K>50:
        st.error(warning)
        is_pass = False
    L = st.number_input('Systolic blood pressure（mmHg）', value=40)
    if L<40 or L>250:
        st.error(warning)
        is_pass = False

features_old = ['AGE', '脉搏', '中性粒细胞绝对值', '中性粒细胞与淋巴细胞比值', '冠心病', '白蛋白-加', 'eGFR',
                '国际标准化比率',
                '肌酐-加', '血压Low', 'BMI', '血压high']

features_old_en = ['Age', 'Pulse', 'Absolute neutrophil count', 'NLR', 'Coronary heart disease', 'Albumin', 'eGFR',
                   'INR',
                   'Serum creatinine', 'Diastolic blood pressure', 'BMI', 'Systolic blood pressure']

input_features = [A, B, C, D, E, F, G, H, I, J, K, L]

sample = np.array([x for x in input_features]).reshape(1, -1)
print(sample)
if is_pass:
    if st.button('Predict'):
        model = joblib.load('model_old.pkl')
    
        explainer = shap.Explainer(model)
        shap_values = explainer(sample)
    
        force_plot_data = shap.force_plot(explainer.expected_value[1],shap_values.values[0, :, 1], features_old_en)
        f_x = force_plot_data.data['outValue']
        print("force_plot 中的 f(x):", f_x)
    
        prediction_proba = model.predict_proba(sample)
        prediction_value = model.predict(sample)[0]
        print(prediction_proba)
        result = prediction_proba[:, 1]
        st.subheader("The risk of heart failure in this geriatric patient（≥65）after non-cardiac surgery is "+str(round(f_x,3)))
        # st.write("Probability（class 0, class 1）：", prediction_proba)
    
        st.subheader("SHAP Explanation")
        plt.figure()
        # shap.initjs()
        shap.plots.force(
            explainer.expected_value[1],  # 类别 1 的基准值
            shap_values.values[0, :, 1],  # 类别 1 的 SHAP 值
            sample[0, :],  # 输入样本
            feature_names=features_old_en,  # 特征名称
            matplotlib=True,
            show=False  # 不自动显示图形
        )
        st.pyplot(plt.gcf())

st.write(
    "**Tips:**  \nThe remaining numerical variable were the patient's most recent laboratory test before non-cardic surgery.")
