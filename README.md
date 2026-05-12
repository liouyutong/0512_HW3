# 紅酒品質預測分析專案 (Red Wine Quality Prediction)

本專案為資料科學作業，目標是透過紅酒的理化特徵，利用**多元線性回歸 (Multiple Linear Regression)** 模型預測紅酒的品質評分，並嚴格遵循 **CRISP-DM** 流程進行分析。

## 🚀 專案摘要
*   **學號：** 4112056006
*   **分析方法：** 多元線性回歸、特徵選擇 (SelectKBest)、資料標準化
*   **核心技術：** Python (Pandas, Scikit-learn, Seaborn, Statsmodels)
*   **研究對象：** Kaggle 經典資料集 - [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

## 📊 CRISP-DM 流程成果
1.  **Business Understanding:** 建立客觀預測模型，輔助釀酒產線品質管理。
2.  **Data Understanding:** 透過相關性分析發現酒精濃度 (Alcohol) 為品質的最強預測因子。
3.  **Data Preparation:** 執行資料標準化與特徵選擇，篩選出 8 個關鍵特徵。
4.  **Modeling & Evaluation:** 
    *   **MSE:** 0.3892
    *   **R-squared:** 0.4032
    *   生成預測圖與殘差圖，並計算 95% 信賴區間。

## 🤖 AI 協助與研究
*   **GPT 輔助：** 協助專案架構規劃、程式碼撰寫與數據視覺化。
*   **NotebookLM 研究：** 本專案利用 NotebookLM 針對 IEEE 及 ScienceDirect 相關學術論文進行研究。
    *   **NotebookLM 筆記連結：** [點此查看研究筆記](https://notebooklm.google.com/notebook/7ee6c392-bb28-4004-903e-da57660d909d)

## 📂 檔案清單
*   `4112056006_hw3.py`: 主分析程式碼。
*   `4112056006_hw3_report.pdf`: 正式分析報告。
*   `winequality-red.csv`: 原始數據集。
*   `*.png`: 資料視覺化圖檔（熱圖、預測圖、殘差圖）。
*   `README.md`: 專案總結說明。

---
