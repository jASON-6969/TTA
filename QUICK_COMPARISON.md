# test_v1 vs test_v2 快速对比

## 核心问题
test_v1 的部分历史实验平均增益较小，但确实执行了参数更新。
学习率、权重和步数是待验证的调参因素，不能仅凭像素变化量判定根因。
本表的旧参数来自历史 config.json，不代表 test_v1 当前代码默认值。

---

## 5个修复方向 ✅

### 1️⃣ 学习率提升 (5-50倍)
```
GraTa:   1e-4 → 5e-4  (5x)
TestFit: 1e-5 → 5e-4  (50x) ⭐ 最大提升
DLTTA:   1e-4 → 5e-4  (5x)
SmaRT:   1e-4 → 5e-4  (5x)
VPTTA:   1e-3 → 5e-3  (5x)
```

### 2️⃣ Confidence阈值提升
```
TestFit confidence: 0.80 → 0.95
只保留更高置信度的预测作为伪标签监督，排除低置信度像素
```

### 3️⃣ 特征权重提升 (5倍)
```
TestFit/GraTa feature_weight: 0.10 → 0.50
SmaRT structure/consistency weights: 0.10 → 0.50
```

### 4️⃣ 多次迭代
```
adaptation_steps: 1 → 3
每张图像执行3次梯度更新（而非1次）
```

### 5️⃣ 增强adaptation强度
```
VPTTA prompt_strength: 0.02 → 0.05 (2.5x)
GraTa noise_std: 0.05 → 0.08 (1.6x)
```

---

## 修改的文件 (7个)

### Methods (5个)
- ✅ `methods/grata.py`
- ✅ `methods/testfit.py`
- ✅ `methods/dltta.py`
- ✅ `methods/smart.py`
- ✅ `methods/vptta.py`

### Core (2个)
- ✅ `config.py`
- ✅ `engine.py`

---

## 实测结果与来源

不预设 Dice 或像素变化的提升幅度。完整修正验证见
`reports/tta_validation.md`、批次 `verification_summary.json` 和
`composition_runs/run_catalog.json`。来源索引区分复制的 test_v1 结果、
没有适应程式快照的历史结果，以及与目前代码匹配的实测。

---

## 如何运行

```bash
cd test_v2

# 单个方法
python -m Paradigm.tta_composition.run --methods GraTa --output .\composition_runs\fresh_grata

# 所有方法
python verify_tta_improvements.py
```

⚠️ **注意**：运行时间预计增加约3倍（因为3次迭代）

---

## 下一步

1. 运行test_v2实验
2. 对比 `test_v2/composition_runs/` 和 `test_v1/composition_runs/`
3. 检查 `comparison.csv` 中的 `changed_pixels` 和 `delta_Dice`
4. 分别比较学习率、损失权重和步数；不得用测试集标签选择最终配置。
