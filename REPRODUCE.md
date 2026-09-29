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

```bash
pixi run db-start
pixi run db-status
createdb -h /tmp -p 5433 pxr_challenge 2>/dev/null || true

# 先确认服务器提供 CREATE EXTENSION rdkit，再执行 schema
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

后续按需生成 foundation model 特征：

```bash
pixi run python db/compute_chemeleon.py
pixi run python db/compute_embeddings.py
pixi run python db/compute_chemfm.py
```

这些任务可能需要大量磁盘、内存、GPU 时间和模型下载。每次生成后保存版本和
行数。应先验证 2D 特征和数据库连接，再开始 Boltz-2 全量任务。

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
