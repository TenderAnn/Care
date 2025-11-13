# Care 项目运行手册

本手册汇总仓库内各子项目（向量评测、意图引擎、GraphRAG 融合、Freshrank 排序、评估回放、Dify 展示）的本地运行方法，方便在交付验收或复现时一步步搭建全流程。

## 1. 环境准备
- 建议使用 Python 3.10+（部分模块依赖 3.9 以上）。
- 创建虚拟环境并安装通用依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # 如果无统一 requirements，可按各模块说明安装
```

> 各子项目在自身目录下也提供 `requirements.txt` 或 `pyproject.toml`，若有冲突可在独立虚拟环境中分别安装。

## 2. 语料与评测（任务一）
目录：`insurance-emb-eval/`

1. 安装依赖：
   ```bash
   cd insurance-emb-eval
   pip install -r requirements.txt
   ```
2. 将原始 PDF 放入 `corpus/raw/`（如需），运行切片脚本：
   ```bash
   python scripts/chunking.py --config configs/chunking/base.yaml
   ```
3. 生成向量并评测：
   ```bash
   python scripts/encode_embeddings.py --config configs/encode/bge-large.yaml
   python scripts/eval_faiss.py --config configs/eval/faiss.yaml
   python scripts/bm25_and_fusion_eval.py --config configs/eval/bm25_rrf.yaml
   python scripts/rerank_cross_encoder.py --config configs/eval/cross_encoder.yaml
   python scripts/gen_embedding_report.py --config configs/report/default.yaml
   ```
4. 评测产物位于 `results/` 与 `reports/`，可使用 `scripts/pack_goal1_release.py` 打包交付材料。

## 3. 口语意图引擎（任务二）
目录：`intent_engine/`

1. 安装依赖并运行单元测试：
   ```bash
   cd intent_engine
   pip install -r requirements.txt
   pytest -q
   ```
2. 启动 FastAPI 服务：
   ```bash
   uvicorn api.main:app --reload --port 8100
   ```
3. 核心接口：
   - `POST /intent/parse`：返回意图、槽位、置信度与澄清建议。
   - `POST /intent/rewrite`：输出规范问句模板。
   - `POST /intent/clarify`：触发澄清提示。
   - `POST /intent/feedback`：写回反馈日志。
4. 相关文档在 `docs/STEP4_api_samples.md`、`STEP7_deploy_notes.md`、`STEP8_dify_flow.md`，其中包含 Dify 集成说明与性能测试结果。

## 4. GraphRAG 融合服务（任务三）
目录：`kg_fusion/`

1. 参考 `kg_fusion/docs/pipeline_setup.md`，按顺序执行：
   - 解析 PDF：`python -m kg_fusion.scripts.parse_layout ...`
   - 构建图谱：`python -m kg_fusion.scripts.build_graph ...`
   - 启动服务：`uvicorn kg_fusion.app.main:app --reload --port 8200`
2. `/kg/query` 接口会自动串联意图解析、GraphRAG 检索、语义召回、时效与监管打分，并返回 `plan`、`results`、`score_breakdown`、`debug` 字段供 Dify 使用。
3. 评估脚本：`python kg_fusion/eval/replay_pipeline.py --quiet`，输出结果位于 `kg_fusion/eval/reports/`。

## 5. Freshrank 动态排序（任务四）
目录：`freshrank/`

1. 安装依赖并运行 API：
   ```bash
   cd freshrank
   pip install -r requirements.txt
   uvicorn freshrank.service.api:app --reload --port 8300
   ```
2. 主要脚本：
   - `python scripts/rerank_offline.py --input data/parsed/sample.jsonl`
   - `python -m pytest tests -q`
3. 配置文件说明：
   - `regulatory/weights.yaml`：监管与时效权重，可结合 Phase4/Phase5 文档调整。
   - `data/metadata/ingestion_events.jsonl`：生命周期日志，由解析/建图脚本自动更新。
4. `/rerank` 接口接收 `/kg/query` 的候选结果，返回排序后的证据及时效、监管拆解，Dify 工具模板位于 `dify_tool_config.json`。

## 6. Dify 工作流（全链路展示）
1. 导入 `kg_fusion/docs/dify_flow_phase5.json` 到企业版 Dify。
2. 按 `docs/pipeline_setup.md` 中的提示配置工具 URL、鉴权与参数映射。
3. Flow 节点顺序：`UserQuery → IntentParse → GraphRAGQuery → LifecycleGuard → FreshrankRerank → AnswerSynthesizer → Diagnostics`。
4. LifecycleGuard 节点会读取 `score_breakdown.recency.sla_flags`，在超 SLA 时触发看板告警。

## 7. 评估与回归
- 执行 `kg_fusion/eval/replay_pipeline.py` 可在本地验证 `/kg/query → /rerank` 的准确率、覆盖率和分数拆解一致性。
- 如需扩展真实样本，可在 `kg_fusion/eval/cases/phase5_pipeline.jsonl` 中新增 case，并在 `doc_metadata.json` 补充时间与监管标签。

## 8. 常见问题
- **缺少 OCR/版面依赖**：`parse_layout.py` 支持 `--disable-ocr` / `--disable-layout` 回退。
- **意图服务不可用**：`kg_fusion.app.services.intent_client.IntentClient` 支持 `OFFLINE_MODE=1`，使用样例解析结果；详见 `kg_fusion/docs/pipeline_setup.md` 第 10 节。
- **向量检索超时**：可将 `VECTOR_BACKEND=offline`，改用本地 JSON 索引进行冒烟测试。
- **监管权重调参**：修改 `freshrank/regulatory/weights.yaml` 后，重新运行回放脚本观察指标变化。

## 9. 交付物索引
- 评测报告：`insurance-emb-eval/reports/`
- 意图模型与文档：`intent_engine/docs/`
- GraphRAG & Pipeline 指南：`kg_fusion/docs/pipeline_setup.md`
- 排序权重与工具：`freshrank/regulatory/`, `freshrank/dify_tool_config.json`
- Dify Flow 模板：`kg_fusion/docs/dify_flow_phase5.json`
- 项目阶段进展：`project_overview.txt`

通过以上步骤可以在本地复现从语料解析、图谱构建、语义召回、排序解释、到 Dify 可视化的完整链路。
