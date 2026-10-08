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

Boltz 输入 YAML 原本使用预计算的 PXR 多序列比对文件 `pxr.a3m`。该文件不在
Git 中。作者在历史 `CLAUDE.md` 中记录的做法是：从 AlphaFold Database 获取
`AF-O75469-F1-msa_v6.a3m`，先保存到本地 Windows 下载目录，再复制到：

```text
structures/boltz2/msa/pxr.a3m
```

项目当前没有保留该文件，旧的 AlphaFold URL 也可能已经失效。现在可以使用
Boltz 官方的 MSA server 代替本地文件，不需要重新生成 `pxr.a3m`。官方文档说明，
`--use_msa_server` 默认调用 `https://api.colabfold.com` 自动生成 MMseqs2 MSA。

使用 MSA server 的推荐方式是只查询一次。先生成一个单化合物的 seed 输入：

```bash
rm -rf structures/boltz2/inputs_smoke
pixi run python track1_activity/boltz2/scripts/boltz2_build_inputs.py \
  --smoke --limit 1 --use-msa-server
```

如果服务器上的脚本仍然报 `unrecognized arguments: --use-msa-server`，说明代码
版本还没有同步到支持该选项的提交。可以先确认版本：

```bash
git status --short --branch
git log -1 --oneline
```

推荐将仓库更新到包含该选项的版本；如果暂时不能更新，可用旧脚本生成输入，再
用 PyYAML 删除每个 YAML 中的 `msa` 字段：

```bash
rm -rf structures/boltz2/inputs_smoke
pixi run python track1_activity/boltz2/scripts/boltz2_build_inputs.py --smoke
pixi run python - <<'PY'
from pathlib import Path
import yaml

root = Path("structures/boltz2/inputs_smoke")
for path in root.glob("*.yaml"):
    data = yaml.safe_load(path.read_text())
    for entry in data.get("sequences", []):
        protein = entry.get("protein")
        if protein:
            protein.pop("msa", None)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
print(f"updated {len(list(root.glob('*.yaml')))} YAML files")
PY
```

检查没有残留 MSA 路径：

```bash
grep -R "msa:" structures/boltz2/inputs_smoke || true
```

先只对这个 seed 输入调用一次 MSA server：

```bash
# 这一步只处理 1 个化合物，用来生成共享的 PXR MSA CSV
boltz predict structures/boltz2/inputs_smoke \
  --out_dir structures/boltz2/outputs_smoke \
  --use_potentials \
  --use_msa_server \
  --diffusion_samples 1 \
  --recycling_steps 3 \
  --output_format mmcif \
  --accelerator gpu \
  --devices 1 \
  --num_workers 2
```

Boltz 会把 server 生成的 MSA 写到输出树的 `msa/*_A.csv`。复制它并让完整输入
全部引用同一个文件：

```bash
rm -rf structures/boltz2/inputs
pixi run python track1_activity/boltz2/scripts/boltz2_build_inputs.py \
  --use-msa-server

pixi run python track1_activity/boltz2/scripts/boltz2_prepare_shared_msa.py \
  --server-output structures/boltz2/outputs_smoke \
  --inputs structures/boltz2/inputs \
  --shared-msa structures/boltz2/msa/pxr_server.csv
```

检查所有 YAML 都引用同一个共享 MSA：

```bash
grep -R "msa:" structures/boltz2/inputs | head
find structures/boltz2/inputs -name '*.yaml' -print0 \
  | xargs -0 grep -h "msa:" \
  | sort -u
```

之后的 4652 个化合物运行必须去掉 `--use_msa_server`，因为 YAML 已经提供本地
共享 MSA。这样 Boltz 不会为每个化合物重复访问 MSA server 或生成临时 MSA：

```bash
nohup boltz predict structures/boltz2/inputs \
  --out_dir structures/boltz2/outputs \
  --use_potentials \
  --diffusion_samples 1 \
  --recycling_steps 3 \
  --output_format mmcif \
  --accelerator gpu \
  --devices 1 \
  --num_workers 2 \
  > boltz_full.log 2>&1 &
echo $! > boltz_full.pid
```

Seed 运行完成后先检查是否真的产生了 1 个预测目录和 MSA CSV：

```bash
find structures/boltz2/outputs_smoke -type f | head
find structures/boltz2/outputs_smoke/boltz_results_inputs_smoke/predictions \
  -mindepth 1 -maxdepth 1 -type d | wc -l
find structures/boltz2/outputs_smoke/boltz_results_inputs_smoke/msa \
  -name '*_A.csv' -type f
```

如果需要验证数据库写入链路，可以先处理 smoke 结果：

```bash
pixi run python track1_activity/boltz2/scripts/boltz2_postprocess.py \
  --smoke --db
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

如果不使用上面的共享 MSA 方案，才需要让全量脚本通过
`USE_MSA_SERVER=1` 对每个输入查询 server：

```bash
rm -rf structures/boltz2/inputs
pixi run python track1_activity/boltz2/scripts/boltz2_build_inputs.py \
  --use-msa-server
nohup env USE_MSA_SERVER=1 \
  bash track1_activity/boltz2/scripts/boltz2_full_run.sh \
  > boltz_full.log 2>&1 &
echo $! > boltz_full.pid
```

查看后台日志和进程：

```bash
tail -f boltz_full.log
ps -p "$(cat boltz_full.pid)" -o pid,etime,stat,cmd
```

如果当前 checkout 的输入目录是 `track1_activity/structures/boltz2/inputs`，且
全量脚本仍然固定使用根目录 `structures/boltz2/inputs`，直接提交 Boltz 命令：

```bash
nohup boltz predict track1_activity/structures/boltz2/inputs \
  --out_dir track1_activity/structures/boltz2/outputs \
  --use_msa_server \
  --use_potentials \
  --diffusion_samples 1 \
  --recycling_steps 3 \
  --output_format mmcif \
  --accelerator gpu \
  --devices 1 \
  --num_workers 2 \
  > track1_activity/structures/boltz2/boltz_full.log 2>&1 &
echo $! > track1_activity/structures/boltz2/boltz_full.pid
```

旧版脚本的全量兼容方式是先用不带 `--use-msa-server` 的命令生成 YAML，再对
`structures/boltz2/inputs/` 执行同样的 PyYAML 删除步骤，最后设置
`USE_MSA_SERVER=1` 运行全量脚本。

输入数量以数据库查询结果为准。当前公开数据快照是 4,139 条 train 加 513 条
test，共 4,652 个输入；旧脚本注释中的 4,653 来自较早的 4,140-train 数据库。
可以在服务器上确认实际数量：

```bash
pixi run db-psql -c "SELECT count(*) AS train_rows, count(DISTINCT compound_id) AS train_compounds FROM train_activity;"
pixi run db-psql -c "SELECT count(*) AS test_rows, count(DISTINCT compound_id) AS test_compounds FROM test_activity;"
pixi run db-psql -c "SELECT count(*) FROM compounds c WHERE c.std_smiles IS NOT NULL AND (EXISTS (SELECT 1 FROM train_activity t WHERE t.compound_id = c.id) OR EXISTS (SELECT 1 FROM test_activity t WHERE t.compound_id = c.id));"
```

如果 train/test 行数加起来是 4,653 但输入仍是 4,652，检查是否有化合物缺少
标准化 SMILES：

```bash
pixi run db-psql -c "SELECT c.id FROM compounds c WHERE c.std_smiles IS NULL AND (EXISTS (SELECT 1 FROM train_activity t WHERE t.compound_id = c.id) OR EXISTS (SELECT 1 FROM test_activity t WHERE t.compound_id = c.id));"
```

MSA server 依赖服务器访问 `https://api.colabfold.com`，每个 protein input 都会
产生远程查询和本地缓存，不能把它当成离线运行。若服务器无法访问该服务，需要
使用作者保存的原始 `AF-O75469-F1-msa_v6.a3m`；不要把旧下载 URL 当作可靠来源。

完整任务预计需要数天，脚本支持中断后恢复。完成后将结果写入数据库：

```bash
pixi run python track1_activity/boltz2/scripts/boltz2_postprocess.py --db
pixi run db-psql -c \
"SELECT count(*), count(*) FILTER (WHERE preprocessing_failed = false) FROM compound_boltz2;"
```

`2d_full_boltz` 还需要从完整 Boltz 输出生成两个派生特征：

```bash
pixi run python track1_activity/scripts/eda_cv_prep/11_compute_jazzy_pose.py
pixi run python track1_activity/scripts/extract_boltz2_confidence_features.py
```

确认下列产物存在后，才能训练第一个完整 tabular member：

```text
compound_boltz2                 # 数据库表
compound_boltz2_jazzy           # 数据库表
data/boltz2_confidence_features.parquet
data/chemprop_pretrain_log2fc_predictions.parquet
compound_chemeleon              # 数据库表
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
