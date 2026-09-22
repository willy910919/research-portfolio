# 統計與資料科學作品集｜王邱鈺為

這份作品集收錄我參與的生醫資料分析系統、產業預測研究，以及統計課程學習成果。程式區以實際專案的可公開核心為主，附來源對照、執行說明、測試及方法限制。合成資料只用來展示程式行為，不代表原始研究結果。

| 類別 | 作品與入口 | 可以閱讀／執行的內容 |
|---|---|---|
| 研究專題 | [生醫研究資料與分析計畫核心](projects/biomed/README.md) | 12 個實際模組：資料匯入、資料異動檢視、需求契約、計畫編譯、分析圖與執行稽核；58 項原測試及合成資料示範。 |
| 研究專題 | [產業預測與可靠度方法](projects/industry/README.md) | 真實估計器、歷史績效集成、長期選模、轉折指標、季度可靠度運算器，以及 2026-09-22 增率表純函式。 |
| 課堂學習 | [抽樣方法模擬](coursework/sampling/README.md) | 原始 R 程式；固定有限母體、3,000 次模擬，比較 SRS、比率估計、分層與群集抽樣。 |
| 課堂學習 | [多變量貝氏](coursework/bayesian/README.md) | 本人負責數學原理及實作代碼說明；方法、分工、學習重點與報告入口。 |
| 課堂學習 | [MLE 與 MAP](coursework/mle-map/README.md) | 個人作品；共軛常態推導、樣本資訊與先驗敏感度的學習記錄。 |
| 課堂學習 | [德里氣溫時間序列](coursework/delhi/README.md) | 幾乎全份本人完成的早期學習專題；季節模型、診斷與驗證限制。 |

完整圖文文件依兩類整理在 Google Drive：[作品集總入口](https://drive.google.com/drive/folders/1q2K9JcR0G2FkhXVXVrWJTfOdCSEAyUi0)、[研究專題](https://drive.google.com/drive/folders/1XyPpjNNa1_XnkvsAOfRAHnjBfKDWGdp4)、[課堂學習](https://drive.google.com/drive/folders/1Jz69LXrlcab4hEv5CXLoq1vukDnb714n)。本 repository 不收錄原教材整檔、原始工作表或私人資料。

## 研究專題快覽

![生醫研究系統架構](assets/biomed_architecture.png)

生醫完整系統架構示意。本 repository 收錄資料與契約核心；圖中的完整執行器、LLM、記憶與部署系統不包含在公開子集中。

![五大產業模型的彙整績效比較](assets/industry_metrics.png)

既有研究報告的彙整比較圖：長期選模期與近期開發診斷分開呈現。近期區間不能視為完全未參與開發的外部測試；RMSLE 下降也不保證方向判斷同步改善。這張圖來自報告，不是本 repository 合成示例重新產生的結果；原始逐期資料未附。

## 快速開始

Python 核心不需要 API key、LLM 服務或帳號。建議建立獨立 Python 3.12+ 環境；先安裝 [requirements.txt](requirements.txt)。R 運算只用 base R，使用 R 4.5.0 驗證。

```sh
python -m pip install -r requirements.txt
cd projects/biomed
python -B run_demo.py
python -B run_plan_demo.py
python -B -m unittest discover -s tests -v
cd ../industry
python -B run_demo.py
python -B -m unittest discover -s tests -v
Rscript R/run_synthetic_checks.R
```

抽樣完整實驗的獨立執行方式見[抽樣說明](coursework/sampling/README.md)。本次整理只做 R 語法檢查，未重跑其 3,000 次完整實驗。Windows 的 R 若繼承不支援的 UTF-8 locale，使用各專題 README 提供的明確 locale 指令。

## 作品與本人貢獻

本人負責方法實作、流程整合、結果檢查與文件化；原有套件、合作內容及第三方方法依來源標示。兩項研究以實際開發／分析工作區的程式版本呈現，具體課堂分工分別寫在各作品頁。

公開整理增加了 README、合成示例、部分獨立測試及封裝。原始核心和新增示例已在 [PROVENANCE.json](PROVENANCE.json) 區分。沒有加入新的開源授權；原作者、共同作者及第三方材料的權利不因整理而改變。

## 可重現範圍

- 可以重現：核心函式與契約行為、20 個標準依賴估計器的合成訓練／預測、選模與集成規則、R 增率算術及季度可靠度的時間順序檢查。
- 不附：逐筆真實研究資料、原工作表、模型檔、資料字典實例、執行紀錄、伺服器、機密設定、帳密、第三方 vendor 目錄。沒有將機構或個人資料轉成「範例」公開。
- 因此不能以此 repository 重算原研究的全部數值、原驗證成績或完整 UI／LLM 工作流。生醫核心的契約測試不等於臨床效度；產業合成示例不等於實際預測績效。

[架構與資料邊界](docs/architecture.md) · [驗證結果](docs/validation.md) · [公開檔案清單與 SHA-256](public_manifest.json)
