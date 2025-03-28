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

left, right = st.columns(2)
with left:
    E = st.selectbox('Coronary heart disease (Yes or No)', ['No', 'Yes'])
    if E == 'Yes':
        E = 1
    else:
        E = 0
    A = st.number_input('Age（years）', max_value=100, min_value=65)
    F = st.number_input('Albumin（g/L）', max_value=200, min_value=0)
    D = st.number_input('Neutrophil-to-lymphocyte ratio (NLR)', max_value=50, min_value=0)
    H = st.number_input('International Normalized Ratio（INR）', max_value=50, min_value=0)
    B = st.number_input('Pulse rate（beats/min）', max_value=250, min_value=0)
with right:
    G = st.number_input('eGFR（mL/min/1.73m²）', max_value=500, min_value=0)
    C = st.number_input('Absolute neutrophil count（× 10⁹/L）', max_value=50.0, min_value=0.1)
    J = st.number_input('Diastolic blood pressure（mmHg）', max_value=200, min_value=1)
    I = st.number_input('Serum creatinine（μmol/L）', max_value=500, min_value=0)
    K = st.number_input('BMI kg/m²', max_value=100, min_value=1)
    L = st.number_input('Systolic blood pressure（mmHg）', max_value=250, min_value=1)

features_old = ['AGE', '脉搏', '中性粒细胞绝对值', '中性粒细胞与淋巴细胞比值', '冠心病', '白蛋白-加', 'eGFR',
                '国际标准化比率',
                '肌酐-加', '血压Low', 'BMI', '血压high']

features_old_en = ['Age', 'Pulse', 'Absolute neutrophil count', 'NLR', 'Coronary heart disease', 'Albumin', 'eGFR',
                   'INR',
                   'Serum creatinine', 'Diastolic blood pressure', 'BMI', 'Systolic blood pressure']

input_features = [A, B, C, D, E, F, G, H, I, J, K, L]

sample = np.array([x for x in input_features]).reshape(1, -1)
print(sample)
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
