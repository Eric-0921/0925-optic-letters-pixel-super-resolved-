# 复现计划:Li et al., Opt. Lett. 50(10), 3257 (2025)
**High-fidelity pixel-super-resolved lensless on-chip microscopy via height scanning and synthetic aperture (HSSA)**
哈工大 刘正君课题组。无公开代码、无公开数据("available upon reasonable request")。

---

## 1. 论文核心内容拆解

### 1.1 光路与采集参数(正文给出)
| 项目 | 数值 |
|---|---|
| 光源 | 532 nm 光纤激光 + 针孔 |
| 针孔到样品距离 d0 | ≈ 70 cm(样品面近似平面波) |
| 样品到传感器距离 dn | < 2 mm,手动位移台做轴向扫描 |
| 传感器 | MV-GE1600C,像素 1.34 µm,全视场 28.6 mm² |
| 斜照明 | 手动横向移动针孔,角度在 ±8° 内随机,**不需要标定角度** |
| 采集帧数 | 主对比 9 帧(3 高度 × 3 角度);最少 6 帧(3 高度 × 2 角度)即可分辨 8-6 |
| 样品 | USAF 1951 振幅靶、未染色胃溃疡切片(相位)、蜜蜂翅膀、松茎切片(补充材料) |

### 1.2 预处理(Fig. 1b)
1. 低分辨全息图上采样,记录下采样算子 D(像素分箱)。上采样倍数正文未给(推测 2 倍,见 §4 待确认项)。
2. ADFrFT 自动对焦(该组 Opt. Lasers Eng. 175, 107991, 2024)求每帧的 zn,得到自由空间传播算子 An(角谱法)。
3. 先把各帧回传到样品面,再用 Guizar-Sicairos 傅里叶互相关配准(Opt. Lett. 33, 156)求亚像素位移,得到平移算子 Tn。斜照明在传感器上表现为全息图整体平移,因此不需要角度先验。

### 1.3 前向模型与 HSSA 迭代(Algorithm 1)
损失:L(x) = 1/(2N) Σn ‖ |D An Tn (p·x)| − √In ‖²
- x:样品面复振幅(待求),p:照明/背景波前(待分离),φ = p·x。
- 每次迭代对 N 帧分别做:Sn = D An Tn;yn = Sn φt;φ'n = Sn⁻¹[√In · yn/|yn|](模约束 + 逆变换)。
- 对 N 帧的 φ'n 取平均得 φ'。
- 用 rPIE 型(Maiden, Optica 4, 736, 2017)更新物体与照明:
  - x ← x + p*(φ'−φ) / ((1−α)|p|² + α·max|p|²),α = 0.2
  - p ← p + γ·x*(φ'−φ) / ((1−β)|x|² + β·max|x|²),β = 0.7,γ = 0.05
- φ ← p·x;初始化 φ0 = x0 = √I1,p0 = 1;T = 50 次迭代。

### 1.4 论文要复现的定量结论
- Fig. 2:高度数(1/2/3)× 角度数(1/2/3)九宫格,以二值化的中心区重建作 GT 算 MSE;3×2 时 MSE 首次 < 0.02(0.0197),3×3 为 0.0190。
- Fig. 3:同为 9 帧时,LISA(纯角度扫描)伪影严重;MFAP(纯高度扫描)分辨到 8-5(线宽 1.23 µm);HSSA 分辨到 8-6(1.10 µm)。
- "1.26 倍"是相对像素 1.34 µm 的奈奎斯特极限(≈ 8-4,线宽 1.38 µm):1.38/1.10 ≈ 1.26。
- Fig. 4/5:相位样品上 HSSA 孪生像更少、相位不卷绕。

---

## 2. 可借鉴的开源资源与领域通用做法

| 资源 | 用途 | 备注 |
|---|---|---|
| [THUHoloLab/pixel-super-resolution-phase-retrieval](https://github.com/THUHoloLab/pixel-super-resolution-phase-retrieval) | PSR 相位恢复的统一优化框架(Gao & Cao, Opt. Express 2021;Cells 2022),含仿真数据 | MIT,MATLAB;正是本文 ref [9],D/A/T 算子建模方式可直接对照 |
| [PyHoloscope](https://github.com/MikeHughesKent/PyHoloscope) | 角谱传播、同轴全息重建、自动对焦(GPU) | Python,可用作传播与对焦基线 |
| [pyDHM](https://github.com/catrujilla/pyDHM) | 数值传播器、相位补偿 | Python |
| PtyPy / PtychoShelves / [ptycho 文献中的 rPIE 实现](https://doi.org/10.1364/OPTICA.4.000736) | rPIE + momentum 更新公式 | 本文 ref [26],步 7–8 直接来自它 |
| scikit-image `phase_cross_correlation` | Guizar-Sicairos 亚像素配准 | 本文 ref [25] 的标准实现 |
| Luo et al., LSA 4, e261 (2015) LISA;Greenbaum/Ozcan 多高度 MFAP | 两个对比基线的算法描述 | 无代码,按论文复现 |
| [OpenLM](https://github.com/xuwimming/OpenLM) | 开源 3D 打印无透镜显微镜 + PSR(Lab Chip 2025) | 若需自己搭硬件采集可参考 |
| 领域通用:USAF 1951 靶标仿真、角谱法、Tamura/ToG 等自动对焦判据、泊松+高斯噪声模型 | 仿真与评价 | |

公开的"多高度 + 多角度"原始无透镜全息数据集目前没有找到;单高度同轴全息示例数据(PyHoloscope 自带、UC2 项目 Zenodo)只能用来验证管线的传播/对焦部分。

---

## 3. 实施步骤

### Phase 1 仿真前向模型(不依赖任何外部数据)
- `sim/usaf.py`:按 USAF 1951 标准尺寸(group 4–9)在细网格(如 0.335 µm = 像素/4)上生成振幅靶;`sim/phase_sample.py` 用开源细胞/组织图像生成相位样品(可配合胃溃疡切片的量级)。
- `optics/propagate.py`:角谱法传播(带带限处理),倾斜平面波照明。
- `sim/sensor.py`:像素积分(分箱)到 1.34 µm、随机亚像素位移、z 抖动、泊松 + 读出噪声、非均匀背景照明(用来检验 p 的分离效果)。
- 输出与论文一致的数据集:3 高度 × 3 角度,±8° 随机角度,dn ≈ 0.5–2 mm。

### Phase 2 预处理
- 上采样(D⁻¹)、自动对焦(ToG / Tamura / 稀疏度判据,替代未开源的 ADFrFT;仿真中另有 z 真值可对比)、回传后傅里叶互相关配准得 Tn。

### Phase 3 算法
- `recon/hssa.py`:严格按 Algorithm 1 实现,参数 α=0.2, β=0.7, γ=0.05, T=50;支持 NumPy / CuPy(可选 PyTorch)。
- `recon/mfap.py`:多高度交替投影基线(无照明分离)。
- `recon/lisa.py`:纯角度扫描基线(移位叠加 PSR + 迭代相位恢复)。

### Phase 4 评价与对照论文
- 复现 Fig. 2 九宫格:MSE(对二值化 GT)随高度数 × 角度数的变化,核对 3×2 时是否首次 < 0.02。
- 复现 Fig. 3:三种算法在 9 帧下的可分辨组/元素(按第 7/8 组线对剖面的对比度判据)与剖面曲线。
- 相位样品:相位 RMSE、卷绕情况;记录运行时间。
- 所有结果输出到 `results/`,并写一份"论文值 vs 复现值"对照表。

### Phase 5 实验数据(视可得性)
- 首选:按 "available upon reasonable request" 向通讯作者 zjliu@hit.edu.cn 索取 USAF 数据。
- 次选:用户自有无透镜平台按 §1.1 参数采集(3 高度 × 3 角度即可)。
- 兜底:用公开单高度同轴全息数据验证传播/对焦/配准部分。

---

## 4. 需要用户提供 / 确认的内容

1. **Supplement 1 PDF**(Note 1 合成孔径分析、Note 2 相位仿真、Note 3 松茎、Note 4 计时)。本环境无法访问 Optica 与 figshare,请上传到仓库。
2. 同组的两篇相关论文(可选但很有帮助):ref [13] Opt. Lett. 50, 1085 (2025) DCNS;ref [24] Opt. Lasers Eng. 175, 107991 (2024) ADFrFT 自动对焦。它们很可能给出上采样倍数、预处理细节。
3. **上采样倍数**:正文未写。默认按 2 倍(重建像素 0.67 µm,足以表示 1.10 µm 线宽);若你知道实际值请告知。
4. 是否有自己的无透镜采集平台或原始数据;若有,请一并给出波长、像素尺寸、大致 dn、每帧的高度/角度标签。
5. 算力:仿真规模(如 2048×2048 上采样网格 × 9 帧 × 50 迭代)CPU 可跑;若有 GPU 会更快,但不是必需。

以上 1–5 都不阻塞 Phase 1–4,可先做仿真复现,再等实验数据。
