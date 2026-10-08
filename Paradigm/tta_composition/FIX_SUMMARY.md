# TTA 組合框架修正總結

本文件為早期調參提案的歷史摘要，參數表不是目前 test_v2 預設值。
未验证的效果預測已撤回；目前程式契約、來源索引與批次驗證見 README.md。

## 🎯 修正目標

檢驗 TTA 平均增益較小的原因，區分程式錯誤與調參假設。

---

## 📝 修正清單

### ✅ 1. 簡化 VPTTA 熵計算

**檔案**：`Paradigm/tta_composition/methods/vptta.py`

**修正前（第 38 行）**：
```python
entropy = -(context.student_logits.softmax(dim=1) * 
            context.student_logits.log_softmax(dim=1).exp().clamp_min(1e-6).log()).sum(dim=1).mean()
```

**問題**：
- `log_softmax().exp()` 會將對數機率轉回機率
- 然後再 `clamp_min(1e-6).log()` 又轉回對數
- `log_softmax().exp()` 與 `softmax()` 在數學上等價；沒有證據顯示原梯度方向錯誤。

**修正後（第 37-39 行）**：
```python
probabilities = context.student_logits.softmax(dim=1)
entropy = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=1).mean()
```

**影響**：
- 修正前：VPTTA Dice = -0.000000437（退化）
- 改寫簡化計算，不保證正向改善。

---

### ✅ 2. 提升學習率

**檔案**：`Paradigm/tta_composition/config.py` 第 16 行

| 方法 | 修正前 | 修正後 | 倍數 | 理由 |
|------|--------|--------|------|------|
| **VPTTA** | 1e-3 | **5e-3** | 5× | Prompt 更新過慢 |
| **DLTTA** | 1e-4 | **5e-4** | 5× | 分佈自適應需要更大步長 |
| **TestFit** | 1e-5 | **1e-4** | **10×** | 90 cases 變化但僅 +0.00026 改善 |
| **GraTa** | 1e-4 | **5e-4** | 5× | BatchNorm 參數需要更大步長 |
| **SmaRT** | 1e-4 | **5e-4** | 5× | 風格編碼器學習過慢 |

**依據**：
- TestFit 有 90/138 cases 預測改變，但 Dice 僅改善 +0.000259
- 這個平均分數不足以確認梯度方向或學習率是根因，需要獨立消融實驗。

---

### ✅ 3. 增強 TTA 強度

**檔案**：`Paradigm/tta_composition/config.py` 第 54-65 行

| 參數 | 修正前 | 修正後 | 倍數 |
|------|--------|--------|------|
| `vptta_prompt_strength` | 0.02 | **0.05** | 2.5× |
| `testfit_feature_weight` | 0.10 | **0.30** | 3× |
| `grata_feature_weight` | 0.10 | **0.30** | 3× |
| `grata_noise_std` | 0.03 | **0.05** | 1.67× |
| `smart_structure_weight` | 0.10 | **0.30** | 3× |
| `smart_consistency_weight` | 0.10 | **0.30** | 3× |

**理由**：
- 原設計過於保守，避免破壞已經很好的 base model
- 但導致 TTA 訊號過弱，無法有效自適應

---

## 📊 問題根因分析

### 根因 1：Base Model 已經非常強

```json
{
  "source_val_dice": 0.9562,    // 源域驗證集
  "target_test_dice": 0.9396,   // 目標域測試集
  "validation_target_score_difference": 0.0166
}
```

**分析**：
- Montgomery 測試集 Dice 已達 93.96%
- 兩個資料集分數差約 1.66 個百分點，不能直接當成 domain shift 大小。
- 高平均 Dice 不代表已過擬合目標域或改善空間的理論上限。

### 根因 2：學習率過小

| 方法 | 原 LR | 效果 |
|------|-------|------|
| TestFit | 1e-5 | 90 cases 變化，Dice 僅 +0.00026 |
| VPTTA | 1e-3 | 7 cases 變化，Dice -0.0000004（退化） |
| GraTa | 1e-4 | 2/8 cases 變化，幾乎無改變 |

### 已排除的說法：VPTTA 熵公式錯誤

原式與改寫在數學上等價，不能用來解釋微小退化。

---

## 🎯 預期改善效果

原先提升區間沒有實測依據，已撤回。實際結果只引用來源可核查的完整實驗。

---

## ✅ 驗證方法

### A. 快速驗證（8 cases）

```powershell
cd test_v2
$env:PYTHONPATH = (Resolve-Path .).Path

# 測試目前 VPTTA 設定
..\.venv\Scripts\python.exe -m Paradigm.tta_composition.run `
    --methods VPTTA --max-cases 8 --seed 42

# 測試 TestFit（10× LR）
..\.venv\Scripts\python.exe -m Paradigm.tta_composition.run `
    --methods TestFit --max-cases 8 --seed 42
```

### B. 完整驗證（138 cases）

```powershell
# 執行驗證腳本（自動測試所有方法）
..\.venv\Scripts\python.exe verify_tta_improvements.py
```

### C. GUI 測試

```powershell
..\.venv\Scripts\python.exe -m Paradigm.tta_composition.gui
```

選擇 Montgomery 數據集，全部影像，測試各方法。

---

## 📁 修正檔案清單

1. ✅ `Paradigm/tta_composition/methods/vptta.py` - 簡化熵計算
2. ✅ `Paradigm/tta_composition/config.py` - 提升學習率和 TTA 強度
3. ✅ `verify_tta_improvements.py` - 驗證腳本
4. ✅ `Paradigm/tta_composition/IMPROVEMENTS.md` - 詳細改善記錄

---

## ⚠️ 注意事項

### 1. 可能的副作用

- **過大的學習率**可能在某些 case 上導致不穩定
- **更強的 TTA**可能在簡單 case 上過擬合

### 2. 監控指標

執行實驗時，注意觀察：
- 梯度範數（`gradient_norms` 在 update 紀錄中）
- 損失值變化（`losses` 在 update 紀錄中）
- 預測變化率（`prediction_change_fraction_mean`）

### 3. 回退方案

如果新超參數導致不穩定：

**方案 A：中等激進**
```python
DEFAULT_LRS = {
    "VPTTA": 3e-3,    # 3× instead of 5×
    "TestFit": 5e-5,  # 5× instead of 10×
    "GraTa": 3e-4,    # 3× instead of 5×
    "SmaRT": 3e-4,    # 3× instead of 5×
}

vptta_prompt_strength = 0.035  # 1.75× instead of 2.5×
testfit_feature_weight = 0.20  # 2× instead of 3×
```

**方案 B：歷史參數對照**
```python
# 歷史學習率；需保存獨立 config.json 與實驗結果
DEFAULT_LRS = {"VPTTA": 1e-3, "DLTTA": 1e-4, "TestFit": 1e-5, "GraTa": 1e-4, "SmaRT": 1e-4}
```

---

## 📚 技術驗證

### 適應目標檢查

| 方法 | 熵計算公式 | 狀態 |
|------|-----------|------|
| VPTTA | `-(p * p.clamp_min(1e-6).log()).sum(dim=1).mean()` | ✅ 已修正 |
| DLTTA | `-(p * p.clamp_min(1e-6).log()).sum(dim=1).mean()` | ✅ 正確 |
| TestFit | 高置信度偽標籤 `F.cross_entropy(...)`，非熵最小化 | ✅ 正確 |
| GraTa | `-(p * p.clamp_min(1e-6).log()).sum(dim=1).mean()` | ✅ 正確 |
| SmaRT | `-(p * p.clamp_min(1e-6).log()).sum(dim=1).mean()` | ✅ 正確 |

### 生命週期檢查

- ✅ `propose()` → 所有方法提出梯度更新
- ✅ `_commit()` → 統一檢查並應用更新
- ✅ `after_commit()` → DLTTA 更新 feature memory
- ✅ `after_prediction()` → VPTTA 更新 prompt memory

---

## 🚀 下一步

1. **執行驗證**：運行 `verify_tta_improvements.py`
2. **觀察結果**：檢查 Dice 改善是否達到預期
3. **調整超參數**：如果需要，使用回退方案
4. **完整測試**：在所有 138 cases 上測試組合方法
5. **撰寫報告**：記錄改善幅度和發現

---

**修正日期**：2026-10-06  
**修正者**：Claude (Opus 5.5)  
**狀態**：歷史提案；目前修正驗證見 `reports/tta_validation.md`
