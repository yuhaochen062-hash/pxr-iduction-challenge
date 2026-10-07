# Track 1 复现流程

本文档说明如何在远程 Linux GPU 服务器上复现 OpenADMET PXR Blind Challenge
Track 1 结果。流程按照“先复现原始方案，再做 ablation”组织。

参考结果：

| 阶段 | 排名 | MAE | RAE | R2 | Spearman | Kendall |
|---|---:|---:|---:|---:|---:|---:|
| Phase 1 | 4 | 0.4059 | 0.5359 | 0.6496 | 0.8343 | 0.6459 |
| Phase 2 | 4 | 0.4113 | 0.5703 | 0.6008 | 0.8161 | 0.6225 |

## 复现范围

仓库包含源代码和轻量数据包，但最终方案使用的以下运行时产物不在 Git 中：

- 大型 embedding 表；
- 模型 checkpoint；
- Boltz-2 输出；
- 本地实验数据库；
- 提交账号和 API 状态。

复现分为两层：

1. 重建环境、数据库，并验证 Git 中的数据。
2. 重新生成大型特征和模型产物，再重跑最终 ensemble 和 calibration。

全新 clone 不能直接得到最终 CSV，必须先完成第二层。

## 1. 环境

原始开发环境使用 Python 3.12、PostgreSQL 18 + RDKit cartridge，以及 RTX
5080。当前 `pixi.toml` 声明了 CUDA 13.1 和 GPU 版 PyTorch。

```bash
cd ~/projects/pxr-iduction-challenge
git switch reproduce                 # 使用非 main/master 分支
pixi install
pixi run python --version
pixi run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

生成的数据、checkpoint、embedding 和 submission 应留在 Git 忽略目录中。不要
提交或恢复 `track1_activity/scripts/api.py`、`submit.py`，它们可能包含账号信息。

## 2. 验证数据

```bash
sha256sum --check data/MANIFEST.sha256
pixi run python - <<'PY'
import pandas as pd
from pathlib import Path

for name in ["default_train.parquet", "default_test.parquet",
             "train_activity_db.parquet", "test_activity_db.parquet"]:
    p = Path("data") / name
    df = pd.read_parquet(p)
    print(f"{p}: {len(df)} rows, columns={len(df.columns)}")
PY
```

上游 `default_train.parquet` 和历史本地数据库的训练行数可能不同。连接公开的
RDKit descriptor 表时，应使用 `train_activity_db.parquet` 和
`test_activity_db.parquet`，不要按行号连接新旧数据。

## 3. 获取源数据

仓库已经包含一份轻量数据。需要从官方 Hugging Face 数据集刷新时执行：

```bash
pixi run python download_data.py
```

记录本次实验使用的数据日期或 revision，不要把新数据和旧数据库按行位置混用。

## 4. 初始化 PostgreSQL 和 RDKit

项目默认使用 PostgreSQL 端口 `5433`，Unix socket 为 `/tmp`。

`db-start` 只会启动已有的 PostgreSQL cluster，不会自动创建 `db/pgdata`。
第一次使用时先初始化一次；以后直接从 `pixi run db-start` 开始。确认目录中
已有重要数据库后，不要重复执行 `initdb`。

```bash
mkdir -p db/pgdata
pixi run initdb -D db/pgdata --auth-local=trust --auth-host=trust
pixi run db-start
pixi run db-status
pixi run createdb -h /tmp -p 5433 pxr_challenge 2>/dev/null || true

# 先确认服务器提供 CREATE EXTENSION rdkit，再执行 schema
# pixi run db-psql -c "CREATE EXTENSION IF NOT EXISTS rdkit;"
pixi run db-psql -f db/schema.sql
pixi run db-psql -f db/experiments_schema.sql
pixi run db-psql -f db/lb_submissions_schema.sql
pixi run db-psql -f db/add_std_columns.sql
```

`db/schema.sql` 使用 RDKit 生成列和 GiST 分子索引，普通 PostgreSQL 实例不够。
加载和标准化源数据：

```bash
pixi run python db/load_data.py
pixi run python db/standardize_compounds.py
```

计算 descriptor 前，`compounds` 表必须已经有 `std_smiles` 和 `std_mol`。

复现 Phase 2 本地检查时，再加载已发布标签和 HTChem 数据：

```bash
pixi run python db/load_test_activity_phase1_labels.py
pixi run python db/load_htchem_activity.py
```

## 5. 生成基础特征

先生成可重复的 2D 特征：

```bash
pixi run install-jazzy
pixi run python db/compute_rdkit_descriptors_full.py
pixi run python db/compute_mordred.py
pixi run python db/compute_jazzy.py
```

### 最终方案需要：CheMeleon

最终 tabular member 的特征名包含 `cheme_2d_full_boltz...`，其中包含
CheMeleon 指纹。因此完整复现最终 ensemble 时需要执行：

```bash
pixi run python db/compute_chemeleon.py
```

这个脚本会从 Zenodo 下载 checkpoint 到 `~/.chemprop/chemeleon_mp.pt`，读取
数据库中 `std_mol IS NOT NULL` 的化合物，在 CPU 上生成约 300 维指纹，并清空
重建 `compound_chemeleon` 表。运行完成后检查覆盖率和维度：

```bash
pixi run db-psql -c "SELECT count(*) FROM compounds WHERE std_mol IS NOT NULL;"
pixi run db-psql -c "SELECT count(*), array_length(embedding, 1) FROM compound_chemeleon GROUP BY 2;"
```

### 可选探索：Hugging Face encoder embedding

`compute_embeddings.py` 用于 ChemBERTa、MoLFormer 等通用预训练模型，不是当前
canonical ensemble 的直接前置条件。测试某个 embedding 时只指定一个模型：

```bash
pixi run python db/compute_embeddings.py chemberta-77m-mlm
```

不带参数会依次处理脚本内的全部模型，通常没有必要。脚本会下载权重并清空对应
目标表后重建；表名和 feature 名称见 `run_train.py` 的 `EMBEDDING_TABLES`。

### 可选探索：ChemFM

ChemFM 目前不是最终白名单的必要输入。只有明确要测试 ChemFM 时才执行：

```bash
pixi run db-psql -f db/compound_chemfm_schema.sql
pixi run python db/compute_chemfm.py --size 1b
# 或者：pixi run python db/compute_chemfm.py --size 3b
```

ChemFM 脚本使用 bf16 和针对 RTX 5080 设定的 batch size；RTX 2080 Ti 不应在未
调整 dtype 和 batch size 前直接运行。

### 运行边界

先完成 RDKit、Mordred、Jazzy 和 CheMeleon，并确认数据库行数与特征维度正确，再
安排 Boltz-2 等耗时任务。ChemBERTa、MoLFormer、ChemFM 和其他 frozen low-fidelity
embedding 属于独立实验轴，不要因为脚本存在就全部运行。以上脚本都会写入或重建
目标表，重跑前先确认不会覆盖需要保留的结果。

### CheMeleon 之后的实际执行顺序

完成 CheMeleon 后，先检查覆盖率和维度：

```bash
pixi run db-psql -c "SELECT count(*) FROM compounds WHERE std_mol IS NOT NULL;"
pixi run db-psql -c "SELECT count(*), array_length(embedding, 1) FROM compound_chemeleon GROUP BY 2;"
```

接下来按以下顺序继续。这里的目标是先得到第一个可训练的最终特征组合，
不要同时运行所有探索性 foundation model。

#### 1. 训练 ChemProp 低保真模型

该模型预测 `log2FC @ 8.25 μM` 和 `log2FC @ 33 μM`，之后作为两个特征输入
TabPFN。先训练默认 seed=42 的 checkpoint：

```bash
pixi run python track1_activity/scripts/run_chemprop_pretrain.py \
  --seed 42
```

默认输出为：

```text
track1_activity/checkpoints/chemprop_pretrain/pretrain.pt
```

用 checkpoint 为 train/test 化合物生成预测：

```bash
pixi run python track1_activity/scripts/run_chemprop_predict_log2fc.py
```

默认输出为 `data/chemprop_pretrain_log2fc_predictions.parquet`。检查输出：

```bash
pixi run python - <<'PY'
import pandas as pd

df = pd.read_parquet("data/chemprop_pretrain_log2fc_predictions.parquet")
print(df.shape)
print(df.columns.tolist())
print(df.head())
PY
```

输出必须包含 `log2fc_8p25_pred` 和 `log2fc_33_pred` 两列。

#### 2. 运行 Boltz-2 smoke test

`2d_full_boltz` 特征依赖 Boltz-2 输出。先创建数据库表：

```bash
pixi run db-psql -f db/boltz2_schema.sql
```

Boltz 输入 YAML 使用预计算的 PXR 多序列比对文件 `pxr.a3m`。该文件不在
Git 中。作者在历史 `CLAUDE.md` 中记录的做法是：从 AlphaFold Database 获取
`AF-O75469-F1-msa_v6.a3m`，先保存到本地 Windows 下载目录，再复制到：

```text
structures/boltz2/msa/pxr.a3m
```

项目当前没有保留该文件，也没有记录一个仍然有效的下载地址。不要继续使用旧的
失效 URL；应从 AlphaFold Database 的当前数据入口或作者保存的原始文件中获取，
并在实验记录中保存文件的来源和校验值。确认输入确实引用该文件：

```bash
grep -R "pxr.a3m" structures/boltz2/inputs_smoke/*.yaml | head
```

先只生成 10 个化合物的输入，验证外部 Boltz-2 安装和 GPU：

```bash
pixi run python track1_activity/boltz2/scripts/boltz2_build_inputs.py \
  --smoke

# 使用项目外部安装的 Boltz-2；参数与 full_run.sh 的 R1 设置一致
boltz predict track1_activity/structures/boltz2/inputs_smoke \
  --out_dir track1_activity/structures/boltz2/outputs_smoke \
  --use_potentials \
  --diffusion_samples 1 \
  --recycling_steps 3 \
  --output_format mmcif \
  --accelerator gpu \
  --devices 1 \
  --num_workers 2
```

Boltz-2 使用项目外部的 `uv tool` 安装，不要默认在 Pixi 环境中直接导入或运行。
项目中的 Pixi 环境负责数据库、特征处理和训练脚本；独立的 Boltz 环境负责
`boltz predict` 推理。这样可以避免 Boltz-2 与 Pixi 中的 PyTorch、CUDA、Triton
和 `cuequivariance` 版本互相冲突。

先确认服务器上的外部命令可用：

```bash
which boltz
boltz --help
```

如果 `boltz` 不存在，应先按照 Boltz-2 的安装说明配置 `uv tool` 环境，不要直接
在 Pixi 环境中执行 `pip install boltz` 作为替代。仓库的全量脚本也会直接调用
外部的 `boltz` 命令，并设置需要的 CUDA 动态库路径。

smoke test 成功后再生成完整输入并启动全量任务：

```bash
pixi run python track1_activity/boltz2/scripts/boltz2_build_inputs.py
bash track1_activity/boltz2/scripts/boltz2_full_run.sh
```

完整任务预计需要数天，脚本支持中断后恢复。完成后将结果写入数据库：

```bash
pixi run python track1_activity/boltz2/scripts/boltz2_postprocess.py --db
pixi run db-psql -c \
"SELECT count(*), count(*) FILTER (WHERE preprocessing_failed = false) FROM compound_boltz2;"
```

#### 3. 生成和训练第一个 TabPFN member

完成 RDKit/Mordred/Jazzy、CheMeleon、ChemProp prediction 和 Boltz-2 基础输出后，
先训练最小的最终特征 member：

```bash
pixi run python track1_activity/scripts/run_train.py \
  --model tabpfn \
  --feature cheme_2d_full_boltz_log2fc_pred \
  --split umap \
  --umap-seed 42 \
  --umap-clusters 50 \
  --trials 0
```

该步骤必须成功生成 OOF 和 test prediction，并在 `experiments` 与
`experiment_oof_predictions` 中留下记录。确认这个 member 能正常运行后，再继续
多 seed、top500、ChemProp frozen embedding 和其他 ensemble member。

此阶段暂时不要运行 `compute_embeddings.py` 的全部模型或 `compute_chemfm.py`；
它们不是当前 canonical ensemble 的必要前置步骤。

## 6. 固定切分和特征约定

生产方案使用五折 UMAP split：

- Morgan radius 2、2048 bit；
- Jaccard distance；
- UMAP seed `42`；
- 先聚类为 50 个 KMeans cluster，再分配到 5 个 fold。

Scaffold split 只用于诊断。训练和测试加载必须保持数据库中的 `ORDER BY t.id`
顺序。实现位于 `track1_activity/src/splits.py`，特征入口位于
`track1_activity/src/features.py` 和 `track1_activity/scripts/run_train.py`。

## 7. 生成低保真 `pred log2FC`

最终方案中的 `pred log2FC` 不是最终 pEC50 模型，而是低保真辅助信号：

```text
SMILES -> ChemProp -> 8.25 μM / 33 μM 的 log2FC 预测
       -> 作为两个特征输入 TabPFN -> pEC50
```

相关入口：

```bash
pixi run python track1_activity/scripts/run_chemprop_pretrain.py --help
pixi run python track1_activity/scripts/run_chemprop_pretrain_optuna.py --help
pixi run python track1_activity/scripts/run_chemprop_predict_log2fc.py --help
```

具体超参数、checkpoint 和多 seed 配置参考
`docs/track1_explain/model_inventory.md`、GitHub issue #100 和 #208。

最终 pool 还使用冻结的 ChemProp 等 encoder embedding，以及 Boltz-2 trunk 表示。
这些特征需要单独生成并注册到 PostgreSQL 后，才能训练对应的 ensemble member。
Boltz-2 全量运行应使用 `track1_activity/boltz2/scripts/` 中支持恢复的脚本。

## 8. 训练单模型 member

使用统一入口训练单个模型：

```bash
pixi run python track1_activity/scripts/run_train.py \
  --model tabpfn \
  --feature <feature-name> \
  --split umap \
  --umap-seed 42 \
  --umap-clusters 50 \
  --trials 0
```

先查看当前支持的 feature 名称：

```bash
pixi run python track1_activity/scripts/run_train.py --help
```

每个准备加入 ensemble 的 member 都必须保存 OOF prediction、测试集 prediction，
并在 `experiments` 和 `experiment_oof_predictions` 中记录。缺少 OOF 行、行顺序
不同或混用不同 split，都会使 ensemble 比较失效。

`run_all_models.sh` 是历史 GPU 流程，不等于最终 2026 ensemble。使用前应根据
目标实验记录检查脚本和 trial 数量：

```bash
bash track1_activity/scripts/run_all_models.sh
```

## 9. 重建 canonical ensemble

权威 member 白名单是 `track1_activity/scripts/run_ensemble.py` 中的
`ENSEMBLE_MODELS`。不要自动查询数据库中的所有实验，旧实验或错误 split 可能
污染 pool。

确认白名单中的 member 都已生成且 OOF/test 输出对齐后运行：

```bash
pixi run python track1_activity/scripts/run_ensemble.py
```

脚本会生成多个候选文件，其中包括 `ens_caruana_bag20.csv`。当前默认方案是带
bagging 的 Caruana forward-selection ensemble，但不要只因为 OOF 最低就认定它
一定会提升公开 leaderboard。

如果 pool 有实质变化，重新运行两个 calibration：

```bash
pixi run python track1_activity/scripts/run_ensemble_calibrate.py
pixi run python track1_activity/scripts/run_ensemble_calibrate_importance.py
```

## 10. 本地验证和已发布标签检查

候选 CSV 先相对于可信 anchor 执行 preflight：

```bash
pixi run python track1_activity/scripts/submission_preflight.py \
  --candidate track1_activity/submissions/<candidate>.csv \
  --anchor track1_activity/submissions/ens_caruana_bag20.csv
```

检查预测均值和方差、最大移动、整体 shift，以及异常化合物的移动。Phase 1/Phase
2 已发布标签只能用于 answer-check 和验证设计；它们只覆盖部分测试化合物，不能
替代原始 blind evaluation。记录标签子集，并计算 MAE、RAE、R2、Spearman 和
Kendall。

不要只因为 OOF 有很小提升就提交新版本。项目历史中多次出现 OOF 和公开 LB
方向相反的情况，尤其是高度相关的 ensemble member 增加操作。

## 11. 提交

只有需要提交时才配置本地提交客户端。按照 challenge 客户端说明完成认证，先运行
preflight，再提交 CSV。不要提交 token、`api.py` 或 `submit.py`。

为每次候选保存文件名、实验 ID、ensemble 权重、calibration 方法、preflight 结果
和提交备注。回看排名时使用 `docs/leaderboards/activity/` 下最新快照，以及
PostgreSQL 中的 `lb_submissions` 和 `lb_submission_history`。

## 12. 首个 ablation：去掉 predicted `log2FC`

复现原始方案后，再进行 controlled SWAP：

1. 保持 UMAP folds、随机种子、TabPFN 版本和训练行完全一致。
2. 只从 tabular feature matrix 中移除两个 predicted `log2FC` 列。
3. 使用新的实验名称训练相同 member，并保存新的 OOF prediction。
4. 比较 OOF、prediction correlation 和 Phase 1/AS1 指标。
5. 使用独立 ensemble pool，不覆盖 canonical allow-list。

相关入口：

```bash
pixi run python track1_activity/scripts/run_admet_ai_tabpfn_no_log2fc_top500.py --help
```

通用 member 应在 `run_train.py` 的特征组装路径中明确排除这两列，并在实验
metadata 中记录 `no_log2fc`。同时检查移除前后的特征数量和训练/测试化合物顺序。

## 13. 复现完成标准

达到以下状态后，才算完成有意义的复现 checkpoint：

- 数据库中的 train/test 行和 descriptor 已确定；
- 至少一个 UMAP-split TabPFN member 生成 OOF/test prediction；
- `experiments` 和 `experiment_oof_predictions` 有对应记录；
- `run_ensemble.py` 能加载白名单中的全部 member；
- `ens_caruana_bag20.csv` 和 calibration 输出通过 preflight；
- 指标、命令、软件版本和生成产物位置已记录。

详细长期操作规则见 `AGENTS.md`，历史实验取舍见 GitHub issue #100 和 #208。
