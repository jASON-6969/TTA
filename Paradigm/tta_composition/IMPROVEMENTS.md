# TTA Composition 改善記錄

## 修正日期：2026-10-06

本文件保留早期調參提案的參數表，不代表 test_v2 的目前預設值。
調參理由是待驗證假設；目前實作與驗證方式見 README.md，實測見批次摘要。

## 🐛 Bug 修正

### 1. VPTTA 熵計算公式簡化
**位置**：`methods/vptta.py` 第 37-39 行

**問題**：
```python
# 原式：log_softmax().exp() 與 softmax() 在數學上等價
entropy = -(context.student_logits.softmax(dim=1) * 
            context.student_logits.log_softmax(dim=1).exp().clamp_min(1e-6).log()).sum(dim=1).mean()
```

**修正**：
```python
# 正確：與其他方法一致的熵計算
probabilities = context.student_logits.softmax(dim=1)
entropy = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=1).mean()
```

**影響**：
- 修正前：VPTTA Dice 改善 -0.000000437（退化）
- 改寫簡化計算，不能據此推論梯度方向原本錯誤或必定改善。

---

## ⚙️ 超參數調整

### 2. 學習率提升
**位置**：`config.py` 第 16 行

| 方法 | 原學習率 | 新學習率 | 提升倍數 | 理由 |
|------|----------|----------|----------|------|
| VPTTA | 1e-3 | **5e-3** | 5× | Prompt 更新過慢 |
| DLTTA | 1e-4 | **5e-4** | 5× | 分佈自適應需要更大步長 |
| TestFit | 1e-5 ⚠️ | **1e-4** | **10×** | 極小學習率導致 90 cases 變化但改善僅 +0.00026 |
| GraTa | 1e-4 | **5e-4** | 5× | BatchNorm 自適應需要更大步長 |
| SmaRT | 1e-4 | **5e-4** | 5× | 風格編碼器學習過慢 |

---

### 3. TTA 強度增強
**位置**：`config.py` 第 54-65 行

| 參數 | 原值 | 新值 | 提升倍數 | 理由 |
|------|------|------|----------|------|
| `vptta_prompt_strength` | 0.02 | **0.05** | 2.5× | 加性擾動係數，並非影像修改比例 |
| `testfit_feature_weight` | 0.10 | **0.30** | 3× | 特徵對齊權重不足 |
| `grata_feature_weight` | 0.10 | **0.30** | 3× | 特徵一致性權重不足 |
| `grata_noise_std` | 0.03 | **0.05** | 1.67× | 增強噪聲強度增加魯棒性 |
| `smart_structure_weight` | 0.10 | **0.30** | 3× | 結構約束過弱 |
| `smart_consistency_weight` | 0.10 | **0.30** | 3× | EMA 一致性約束過弱 |

---

## 📊 問題根因分析

### Base Model 已經非常強
```json
{
  "source_validation_dice": 0.9562,
  "target_test_dice": 0.9396,
  "validation_target_score_difference": 0.0166
}
```

**分析**：
- Source-only model 在 Montgomery 測試集達到 93.96% Dice
- 兩個資料集分數差約 1.66 個百分點，不是 domain shift 的直接量測。
- 高平均 Dice 不代表已過擬合目標域或沒有漏分割病例。

### 原始 TTA 結果

| 方法 | Dice 改善 | 預測改變 cases | 平均像素變化率 |
|------|-----------|----------------|----------------|
| VPTTA | -0.000000437 ❌ | 7/138 | 0.000000885 |
| TestFit | +0.000259 ✅ | 90/138 | 0.000117 |
| GraTa | +0.000001794 | 2/8 | 0.0000038 |

**結論**：
1. TestFit 有改善但幅度極小（90 cases 變化但 Dice 僅 +0.00026）
2. VPTTA 的微小負增益原因未由熵公式改寫證明。
3. 表中 GraTa 為 8 張小樣本；另有 138 張完整歷史結果，須分開標示。

---

## ✅ 修正後預期效果

原先列出的提升區間沒有實驗或統計依据，已撤回。
各方法與組合可能改善或退化，只能引用保存的完整實測結果。

---

## 🔬 驗證建議

### A. 完整重測（推薦）
```powershell
# 從 test_v2 執行
$env:PYTHONPATH = (Resolve-Path .).Path
..\.venv\Scripts\python.exe -m Paradigm.tta_composition.run `
    --methods VPTTA,DLTTA,TestFit,GraTa,SmaRT `
    --seed 42 `
    --output .\composition_runs\fixed_params_full
```

### B. 消融測試
```powershell
# 僅測試修正後的 VPTTA
..\.venv\Scripts\python.exe -m Paradigm.tta_composition.run `
    --methods VPTTA --seed 42 `
    --output .\composition_runs\vptta_fixed

# 僅測試提升 LR 的 TestFit
..\.venv\Scripts\python.exe -m Paradigm.tta_composition.run `
    --methods TestFit --seed 42 `
    --output .\composition_runs\testfit_10x_lr
```

### C. 更具挑戰性的測試集
高平均分數不能證明飽和，外域檢驗可考慮：
1. 其他未參與來源訓練的資料；SZ-CXR 本來是來源域，不能稱為更遠域。
2. 人工添加噪聲/模糊/對比度變化
3. 使用其他醫院的胸片數據

---

## 📝 技術細節

### 熵計算標準化
熵最小化分支使用下列公式；TestFit 的高置信度偽標籤交叉熵是另一個目標：
```python
probabilities = logits.softmax(dim=1)
entropy = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=1).mean()
```

**驗證**：
- ✅ VPTTA：公式簡化
- ✅ DLTTA：正確
- ✅ TestFit：使用偽標籤 cross_entropy，不等價於預測熵
- ✅ GraTa：正確
- ✅ SmaRT：正確

### 生命週期正確性
- ✅ VPTTA 的 `after_prediction` 在預測成功後更新 prompt memory
- ✅ DLTTA 的 `after_commit` 在參數更新後立即更新 feature memory
- ✅ 所有梯度提案在參數更新前完成
- ✅ 參數更新前檢查梯度範數有限性

---

## ⚠️ 注意事項

1. **學習率敏感性**：新的學習率可能在某些 case 上過大，需監控訓練穩定性
2. **過擬合風險**：更大的 TTA 強度可能導致某些簡單 case 退化
3. **計算成本**：增加更新步數會增加運算；調高係數本身不增加步數
4. **改善上限**：不能由來源驗證與目標測試分數差推導 TTA 理論上限

---

## 📚 參考

- Base model checkpoint: `base_model/benchmark_cxr_base_no_source_aug/source_checkpoint.pth`
- Baseline Dice: 0.9396 (93.96%)
- Dataset: SZ-CXR (source) → Montgomery (target), 138 test cases
