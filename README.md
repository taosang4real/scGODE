# 基于图自编码器与神经ODE/CNF的单细胞动态预测模型

## 1. 项目目标

本项目旨在构建一个深度学习模型，用于预测单细胞群体基因表达谱随时间的动态变化。模型的核心是学习细胞在不同时间点之间的状态转换规律，从而能够根据一个时间点 ($t_0$) 的细胞群体基因表达特征，预测该群体在未来某个时间点 ($t_1$) 的基因表达分布。

## 2. 核心架构

模型采用了灵活的多阶段设计，集成了多种先进技术，可通过配置文件进行切换和组合：

* **变分图自编码器 (VGAE)**:
    * **编码器 (Encoder)**: 使用图神经网络 (GNN) 和图池化操作，将基于蛋白质-蛋白质相互作用 (PPI) 图的细胞基因表达数据，编码到一个结构化的、低维的**潜空间分布** (由均值$\mu$和对数方差$\log\sigma^2$参数化)。
    * **解码器 (Decoder)**: 从潜空间表示中，通过反池化和GNN层，重构出原始高维基因表达谱的**概率分布**（概率化解码器）。

* **潜空间动力学模型 (可配置)**:
    * **神经ODE (NeuralODE)**: 学习潜状态的**确定性**演化轨迹。
    * **连续归一化流 (CNF)**: 学习潜状态**概率密度**的精确变换，直接对分布到分布的演化进行建模。

* **损失函数**:
    * **重构损失**: 基于概率解码器的负对数似然 (NLL) 损失。
    * **KL散度**: (VGAE) 用于正则化潜空间。
    * **CNF NLL损失**: (CNF) 用于优化动力学流模型的似然。
    * **最优传输 (OT) 损失**: 作为辅助损失，用于衡量潜空间和表达空间中预测分布与真实分布的差异。

## 3. 代码工程结构

```
scGODE/
├── configs/
│   └── main_config.yaml
├── data/
│   ├── ALL_cells_HVG_expr.csv
│   ├── ALL_cells_meta.csv
│   └── model_input_adj_mat.npy
├── notebooks/
│   └── 03_results_visualization_and_interpretation.ipynb
├── results/
│   └── [experiment_name]/
│       ├── checkpoints/
│       ├── logs/
│       └── umap_plots/
├── src/
│   ├── __init__.py
│   ├── models/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── losses.py
│   ├── trainer.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## 4. 安装依赖

在开始之前，请确保您已安装所有必需的Python库。建议在虚拟环境中进行安装。

```bash
pip install -r requirements.txt
```requirements.txt` 文件内容:
```text
torch
torchvision
torchaudio
torch_geometric
pandas
numpy
PyYAML
omegaconf
tqdm
matplotlib
umap-learn
geomloss
scipy
torchdiffeq
```

## 5. 配置模型

所有实验的参数都通过 `configs/main_config.yaml` 文件进行控制。在运行前，请仔细检查并根据您的需求修改此文件。

**关键配置项**:

* **`data_params.data_dir`**: 设置为您的数据文件所在的**绝对路径**。
* **`model_params.encoder.is_variational`**: 设置为 `true` 以使用VGAE编码器，`false` 则使用确定性GAE。
* **`model_params.decoder.distribution`**: 设置为 `"gaussian"` 使用概率解码器，或 `"deterministic"` 使用确定性解码器。
* **`model_params.node.type`**: 设置为 `"CNFIntegrator"` 使用连续归一化流，或 `"NeuralODE"` 使用标准神经ODE。
* **`training_params.joint_training_mode`**: 设置为 `"latent_only"` 以专注于潜空间动力学训练，或 `"end_to_end"` 进行完整的端到端训练。
* **`loss_weights`**: 调整各个损失项（如`L_KL_latent`, `L_CNF_NLL`, `L_OT_latent`等）的权重以平衡训练目标。
* **`experiment_name`**: 为您的每次实验设置一个唯一的名称，相关的检查点和日志将保存在 `results/[experiment_name]` 目录下。

## 6. 运行模型

所有操作都通过 `main.py` 脚本启动。

### 6.1 GAE 预训练

此步骤用于训练变分图自编码器（VGAE），以获得一个高质量的潜空间表示。

```bash
python3 main.py --config configs/main_config.yaml --mode pretrain_gae
```
训练好的模型检查点将保存在 `results/[experiment_name]/checkpoints/` 目录下。

### 6.2 联合训练

此步骤用于训练动力学模型（NODE或CNF）。它可以从一个预训练好的GAE检查点开始。

**模式一：潜空间动力学训练 (推荐先进行此模式)**
确保 `main_config.yaml` 中的 `training_params.joint_training_mode` 设置为 `"latent_only"`。

```bash
# 从一个预训练好的GAE检查点开始
python3 main.py --config configs/main_config.yaml --mode train_joint \
--checkpoint_path results/experiment_vgae_prob_cnf_ready/checkpoints/gae_pretrain_best.pt
```

**模式二：端到端微调**
确保 `main_config.yaml` 中的 `training_params.joint_training_mode` 设置为 `"end_to_end"`。

```bash
# 从一个经过潜空间训练的检查点开始微调
python3 main.py --config configs/main_config.yaml --mode train_joint \
--checkpoint_path results/experiment_vgae_prob_cnf_ready/checkpoints/joint_train_best_latent_only.pt
```

### 6.3 评估模型

此步骤用于评估一个训练好的模型检查点，并生成UMAP图。

```bash
python3 main.py --config configs/main_config.yaml --mode evaluate \
--checkpoint_path /path/to/your/best_checkpoint.pt \
--eval_type all # 'gae', 'joint', or 'all'
```
评估结果（包括量化指标和UMAP图）将保存在 `evaluation_params.output_dir` 指定的目录中。

---

至此，我们已经完成了整个代码工程框架的更新。您可以以此为基准进行后续的实验和