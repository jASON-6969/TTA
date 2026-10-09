# TTA方法改进总结

## 问题分析

本文参数对比对应 test_v1 历史运行配置。学习率、权重和迭代次数的
改动属于实验设置，不保证提升；完整修正验证以批次摘要和来源索引为准。

从test_v1的实验结果发现：
1. **部分预测不变**：离散标签不变不代表参数没有更新，需看完整逐病例数据
2. **Base model平均表现较好**：Dice = 93.96%, PPV = 99.2%，仍需分析漏分割与困难病例
3. **更新幅度待分析**：参数确有更新，离散预测可能没有跨过分类边界
4. **损失尺度待分析**：0.1 的系数不等于损失或梯度贡献只有 10%
5. **迭代次数待验证**：单步与多步需要消融比较，多步也可能放大错误

## 实施的5个修复方向

### ✅ 修复1: 提高学习率 (5-50倍)

#### 修改的文件：
- `Paradigm/tta_composition/methods/grata.py`
- `Paradigm/tta_composition/methods/testfit.py`
- `Paradigm/tta_composition/methods/dltta.py`
- `Paradigm/tta_composition/methods/smart.py`
- `Paradigm/tta_composition/methods/vptta.py`
- `Paradigm/tta_composition/config.py`

#### 具体变化：
| 方法 | 原学习率 | 新学习率 | 提升倍数 |
|------|---------|---------|---------|
| GraTa | 1e-4 | 5e-4 | 5倍 |
| TestFit | 1e-5 | 5e-4 | 50倍 |
| DLTTA | 1e-4 | 5e-4 | 5倍 |
| SmaRT | 1e-4 | 5e-4 | 5倍 |
| VPTTA | 1e-3 | 5e-3 | 5倍 |

---

### ✅ 修复2: 调整Confidence阈值

#### 修改的文件：
- `Paradigm/tta_composition/methods/testfit.py`
- `Paradigm/tta_composition/config.py`

#### 具体变化：
```python
# TestFit
confidence: 0.80 → 0.95
```

**实际作用**：代码采用 `confidence > threshold`，提高到 0.95 会保留更高置信度伪标签、排除低置信度像素。PPV 是基于真实标注的评估指标，不能直接用于选择 softmax confidence 门槛。

---

### ✅ 修复3: 增加特征和正则化权重 (5倍)

#### 修改的文件：
- `Paradigm/tta_composition/methods/grata.py`
- `Paradigm/tta_composition/methods/testfit.py`
- `Paradigm/tta_composition/methods/smart.py`
- `Paradigm/tta_composition/config.py`

#### 具体变化：
| 方法 | 参数 | 原值 | 新值 | 提升倍数 |
|------|-----|------|------|---------|
| GraTa | feature_weight | 0.10 | 0.50 | 5倍 |
| TestFit | feature_weight | 0.10 | 0.50 | 5倍 |
| SmaRT | structure_weight | 0.10 | 0.50 | 5倍 |
| SmaRT | consistency_weight | 0.10 | 0.50 | 5倍 |

---

### ✅ 修复4: 增加adaptation迭代步数

#### 修改的文件：
- `Paradigm/tta_composition/config.py`
- `Paradigm/tta_composition/engine.py`

#### 具体变化：
```python
# config.py新增参数
adaptation_steps: int = 3  # 每张图像执行3次adaptation迭代

# engine.py修改step方法
原来：每张图像只执行1次梯度更新
现在：每张图像执行3次梯度更新（可配置）
```

**实现细节**：
- 在`CompositionEngine.step()`方法中添加循环
- 对每张图像重复执行：context生成 → proposals → commit
- 每次参数更新后恰好调用一次 `after_commit`；DLTTA 每步记录一次特征，SmaRT 每步更新 EMA
- 最后一次预测成功后才发布 VPTTA prompt memory，每张图像一次
- 日志的 `iterations` 保存每一步 loss、梯度、学习率和时长
- 在metrics中记录实际的迭代次数

---

### ✅ 修复5: 增加adaptation强度

#### 修改的文件：
- `Paradigm/tta_composition/methods/vptta.py`
- `Paradigm/tta_composition/config.py`

#### 具体变化：
```python
# VPTTA
prompt_strength: 0.02 → 0.05  (2.5倍)

# GraTa (间接增强)
grata_noise_std: 0.05 → 0.08  (1.6倍)
```

**原因**：增加prompt的影响强度和augmentation的扰动强度，使adaptation对模型产生更明显的影响。

---

## 修改文件清单

### Python代码文件 (7个)
1. `Paradigm/tta_composition/methods/grata.py` - GraTa方法
2. `Paradigm/tta_composition/methods/testfit.py` - TestFit方法
3. `Paradigm/tta_composition/methods/dltta.py` - DLTTA方法
4. `Paradigm/tta_composition/methods/smart.py` - SmaRT方法
5. `Paradigm/tta_composition/methods/vptta.py` - VPTTA方法
6. `Paradigm/tta_composition/config.py` - 配置文件
7. `Paradigm/tta_composition/engine.py` - 主引擎

### 关键参数对比表

| 参数类别 | 参数名 | test_v1 | test_v2 | 变化 |
|---------|-------|---------|---------|------|
| **学习率** | GraTa lr | 1e-4 | 5e-4 | ↑ 5x |
| | TestFit lr | 1e-5 | 5e-4 | ↑ 50x |
| | DLTTA lr | 1e-4 | 5e-4 | ↑ 5x |
| | SmaRT lr | 1e-4 | 5e-4 | ↑ 5x |
| | VPTTA lr | 1e-3 | 5e-3 | ↑ 5x |
| **阈值** | TestFit confidence | 0.80 | 0.95 | ↑ 18.75% |
| **权重** | GraTa feature_weight | 0.10 | 0.50 | ↑ 5x |
| | TestFit feature_weight | 0.10 | 0.50 | ↑ 5x |
| | SmaRT structure_weight | 0.10 | 0.50 | ↑ 5x |
| | SmaRT consistency_weight | 0.10 | 0.50 | ↑ 5x |
| **强度** | VPTTA prompt_strength | 0.02 | 0.05 | ↑ 2.5x |
| | GraTa noise_std | 0.05 | 0.08 | ↑ 1.6x |
| **迭代** | adaptation_steps | 1 | 3 | ↑ 3x |

---

## 实测验证

原先提升幅度与像素数量预测没有依据，已撤回。实际结果可能退化，须报告
逐病例改善和退化，区分复制的旧结果、未版本化结果和当前代码验证。
完整修正验证见 `reports/tta_validation.md` 与批次 `verification_summary.json`。

---

## 如何运行test_v2

```bash
cd "C:/Users/jason/Documents/cource/y4sem1/fyp/test_v2"

# 运行单个方法（例如GraTa）
python -m Paradigm.tta_composition.run --methods GraTa --output .\composition_runs\fresh_grata

# 运行所有5个方法
python verify_tta_improvements.py

# 使用自定义配置
python -m Paradigm.tta_composition.run --config your_config.json
```

---

## 注意事项

1. **计算时间**：由于增加到3次迭代，总运行时间预计增加约3倍
2. **GPU内存**：每步重新构建并释放计算图；学习率大小本身不会增加模型内存
3. **过拟合风险**：对单张图像执行3次更新可能导致过拟合，需要观察验证
4. **超参数调优**：这些值是根据经验设定的，可能需要进一步调整

---

## 后续实验建议

如果test_v2的效果仍然不理想，可以尝试：

1. **学习率消融**：在独立验证集比较多个学习率
2. **迭代次数消融**：比较 1、3、5 步并检查逐病例退化
3. **使用更强的augmentation**：增加更多的数据增强策略
4. **调整损失函数**：考虑添加更多的正则化项
5. **分析失败案例**：找出baseline表现最差的图像，针对性优化

---

生成时间：2026-10-06
版本：test_v2
