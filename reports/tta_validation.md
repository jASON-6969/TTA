# test_v2 修正與驗證

日期：2026-10-06。此報告檢查本地 TTA 組合框架、GUI 與驗證工具；
沒有重新訓練來源模型，也不宣稱是各作者原版方法的完整復現。

## 已修正問題

- 每次參數更新恰好執行一次 `after_commit()`。三步適應原本呼叫四次，
  現在是三次；DLTTA 每步加入一筆特徵記憶。
- VPTTA 在整張影像適應與預測期間使用同一份歷史 prompt memory，
  最終預測成功後才加入一筆新記憶；失敗不發布該 prompt。
- 每張圖的日誌增加 `iterations`，保存每步 loss、梯度範數、學習率、
  更新參數數量與時長。頂層保持最後一步欄位相容性；頂層時長為整張圖，
  `method_metrics` 仍為最後一次迭代。
- GUI 與 CLI 可設定並保存 `adaptation_steps`，拒絕零、負數與非整數。
- 驗證腳本補上 DLTTA 單方法，修正方法名稱、缺少結果檔、部分失敗、
  全部失敗的處理與退出碼，移除硬編碼舊結果及未驗證的提升承諾。
- CLI 拒絕覆寫已包含實驗產物的輸出目錄，避免新失敗混入舊成功摘要。
- 每次新實驗保存適應程式 SHA-256 快照到 `provenance.json` 和 `audit.json`。
- 文件修正 confidence 篩選方向、熵公式等價性、來源與目標分數差的解釋。
  TestFit 偽標籤交叉熵並不等價於預測熵最小化。

## 回歸檢查

- 完整入口 `Paradigm.tta_composition.tests.run_checks`：142/142 通過。
- 包含 131 個 unittest 與 11 個函式測試；32 種方法子集各自測 1/3/5 步。
- 覆蓋逐 commit 的 EMA、DLTTA 記憶內容與回呼數量、prompt 成功及失敗
  生命週期、完整迭代日誌、GUI 傳遞、結果來源分類及覆寫保護。
- mypy 使用 `--check-untyped-defs --follow-imports=skip --ignore-missing-imports`：
  9 個修改程式檔通過。此結果不代表所有第三方套件也經過靜態型別驗證。

從 test_v2 執行：

```powershell
..\.venv\Scripts\python.exe -X utf8 -B -m Paradigm.tta_composition.tests.run_checks
..\.venv\Scripts\python.exe -X utf8 -B verify_tta_improvements.py --seed 42 --max-cases 138
..\.venv\Scripts\python.exe -X utf8 -B -m Paradigm.tta_composition.run_catalog
```

## 完整實驗

批次：`composition_runs/verify_batch_20261006_232500_610034/verification_summary.json`。
六組全部成功。每組為 Montgomery 138 張、seed 42、256x256、z-score、
CUDA 與每張圖三次更新。每組 138 筆日誌含 414 次更新，數值均為有限值；
病例、標註與程式雜湊相同，CSV 重算與保存的 JSON 平均值一致。

Checkpoint SHA-256 與原本 test_v1 相同：
`d08d23748329fdfaef5842c6257ee418b55d2ec8a083790c9c8b2eff3d67cfb2`。

| 方法 | Mean Dice | Mean HD95，像素 | 相對 baseline，百分點 | Dice 改善/退化/不變病例 |
|---|---:|---:|---:|---:|
| Source-only | 0.9395699204 | 2.6404206776 | 0 | 0/0/138 |
| VPTTA | 0.9395685629 | 2.6403965497 | -0.000136 | 5/9/124 |
| DLTTA | 0.9413109838 | 2.5136628584 | +0.174106 | 125/7/6 |
| TestFit | 0.9404412523 | 2.6106540837 | +0.087133 | 106/25/7 |
| GraTa | 0.9414993318 | 2.5045299661 | +0.192941 | 122/10/6 |
| SmaRT | 0.9400225191 | 2.6166634900 | +0.045260 | 73/47/18 |
| 五方法組合 | 0.9408018767 | 2.5769638204 | +0.123196 | 111/23/4 |

## 結果來源

`composition_runs/run_catalog.json` 保存來源分類，沒有改寫或刪除歷史结果。
本次索引含 25 個從 test_v1 複製的歷史 run、7 個未保存適應程式快照的 run，
以及本次 6 個與目前適應程式相符的驗證 run。其中 17 個複製 run 含舊 comparison.json。
舊設定不能由目前預設值反推，缺失的歷史程式快照也不能事後補成當時證據。

## 尚存限制

- VPTTA 仍無有效增益；其微小負增益不是這次回呼修復能解決的問題。
- 五方法組合平均增益低於 GraTa 單方法，不能假定各方法效果相加。
- SmaRT 有 47 張、TestFit 有 25 張相對 baseline 退化；正平均增益不代表
  每張圖都改善。進一步調整需要獨立驗證資料與逐項消融。
- 本批只有一個 seed 和一個目標資料集，沒有多 seed、資料順序或外域驗證。
  沒有宣稱統計顯著、最佳超參數或部署可用性。
