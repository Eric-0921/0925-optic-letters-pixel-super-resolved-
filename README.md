# HSSA 无透镜片上显微复现(仿真)

复现对象:Z. Li et al., "High-fidelity pixel-super-resolved lensless on-chip microscopy via height scanning and synthetic aperture", *Opt. Lett.* **50**, 3257 (2025),以及其 Supplement 1。论文未公开代码与数据,本仓库用物理仿真 + 按论文描述实现的算法来复现其主要结论。

- 复现计划:[`PLAN.md`](PLAN.md)
- 复现报告(结果、与论文的对照、差异分析):[`REPORT.md`](REPORT.md)
- 结果表格:[`results/summary.md`](results/summary.md);图:`results/*.png`

## 代码结构

| 路径 | 内容 |
|---|---|
| `hssa/optics.py` | 角谱传播(带限)、倾斜谱核 H(f+f0)、亚像素平移、像素积分 |
| `hssa/frft.py` | 离散分数傅里叶变换(Ozaktas 快速算法) |
| `hssa/preprocess.py` | ADFrFT 自动对焦(Opt. Lasers Eng. 175, 107991)与回传后傅里叶互相关配准 |
| `hssa/simulate.py` | 前向仿真:针孔球面波斜照明(精确角谱)、4 倍细网格、像素积分、平台抖动、背景照明不均匀、泊松/读出噪声、10 bit 量化 |
| `hssa/usaf.py` | USAF 1951 靶(第 5–9 组,精确面积覆盖渲染)与分辨率判据 |
| `hssa/recon.py` | HSSA(Algorithm 1)、MFAP、LISA,以及扩展:由配准位移估计倾角的倾斜谱核 |
| `hssa/metrics.py` | 与真值对照的 MSE、分辨率、相位误差 |
| `experiments/` | 各实验脚本(见下) |
| `tests/` | FrFT 与光学约定的单元测试 |

## 运行

```bash
pip install -r requirements.txt
python -m pytest -q tests
python experiments/run_usaf.py              # Fig. 2 / Fig. 3 对应实验 + 消融(seed 1,约 25 分钟,4 核 CPU)
python experiments/run_usaf.py --seed 2     # 其他随机种子
python experiments/run_usaf.py --ideal      # 平面波、无背景的理想化仿真
python experiments/run_phase.py             # Supplement Note 2 相位仿真
python experiments/run_angles.py            # 最大照明角 2°/4°/8° 扫描
python experiments/summarize.py             # 汇总为 results/summary.md
python experiments/make_figures.py          # 生成图
```

## 与论文一致的参数

λ = 532 nm;像素 1.34 µm;针孔–样品 70 cm;针孔横移 ≤ 10 cm(照明角 ≤ 8°,随机);最近样品–传感器距离 0.975 mm;3 个高度 × 3 个角度共 9 帧;HSSA 参数 α = 0.2、β = 0.7、γ = 0.05、T = 50;初始化 φ0 = x0 = √I1、p0 = 1。

## 论文未给出、由本复现自行选定的参数

- 上采样倍数 2(重建像素 0.67 µm)。
- 高度:HSSA 用 0.975 / 1.175 / 1.375 mm;MFAP 用 0.975–1.375 mm、步长 50 µm 共 9 个高度。
- 仿真传感器窗口 512×512 像素(686 µm),而非全幅 4608×3456。
- 分辨判据:线对间隙相对相邻线条的 Michelson 对比度 ≥ 0.1(两个方向都满足)。
- MFAP 采用与 HSSA 相同的并行平均投影(即 HSSA 去掉 p 分离)。
