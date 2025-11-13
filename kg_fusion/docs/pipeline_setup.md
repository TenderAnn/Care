# kg-fusion Structured Pipeline Guide

This guide documents the ingestion, graph-building, and query pipeline that was rebuilt in Phase 1 and Phase 2. Follow these steps to reproduce the artefacts locally and wire the services that downstream tasks consume.

## 1. Prerequisites

- **Python**: 3.10+
- **Core dependencies**: install the project requirements (see `requirements.txt` in the repo root if available).
- **Optional but recommended**: layout-aware and OCR tooling used by the hybrid parser. Install them if you want the best parsing quality.

```bash
pip install "layoutparser[layoutmodels]" paddleocr pymupdf numpy
```

If you cannot install the optional dependencies, the parser will automatically fall back to heuristic layout detection and disable OCR. This is suitable for simple PDFs but will reduce accuracy on complex layouts or scanned documents.

## 2. Directory Layout

The pipeline reads and writes artefacts under `kg_fusion/data` by default:

| Directory | Purpose |
|-----------|---------|
| `data/raw/` | Source PDFs to be parsed. Place your insurance documents here. |
| `data/structured/` | Structured JSON outputs from `parse_layout.py` (also stores graph artefacts). |
| `data/parsed/` | Legacy section JSONL files for backward compatibility. |
| `data/index/` | Vector index data consumed by the semantic retriever (existing assets). |

Lifecycle metadata is shared with the freshrank module via `freshrank/data/metadata/ingestion_events.jsonl`.

## 3. Parsing Documents

Use the updated `parse_layout.py` script to convert PDFs into structured chunks enriched with slot labels and lifecycle timestamps.

```bash
python -m kg_fusion.scripts.parse_layout \
  --in-dir kg_fusion/data/raw \
  --structured-out kg_fusion/data/structured \
  --sections-out kg_fusion/data/parsed \
  --intent-dataset intent_dataset_zh_insurance_v1.jsonl \
  --synonyms synonyms_insurance_zh.tsv \
  --lifecycle-log freshrank/data/metadata/ingestion_events.jsonl \
  --verbose
```

Key flags:

- `--disable-layout` / `--disable-ocr`: turn off the advanced cues if the dependencies are unavailable.
- `--no-slot-mapper`: skip slot inference (the parser will still emit structural chunks).
- `--overwrite`: regenerate artefacts even if structured JSON already exists.

Each PDF produces:

- `*.structured.json`: a document-level payload with chunk metadata, anchors, slot matches, and parser configuration.
- `*.sections.jsonl`: one line per chunk replicating the legacy format for components that still depend on it.

The script automatically updates the lifecycle log with `parsed_at` and `metadata_ready_at` timestamps for every `doc_id`.

## 4. Building the GraphRAG Store

After parsing, run the new `build_graph.py` CLI to assemble graph nodes, edges, and query indices. The builder consumes structured artefacts and emits graph files under the same `structured/` directory.

```bash
python -m kg_fusion.scripts.build_graph \
  --structured-in kg_fusion/data/structured \
  --graph-root kg_fusion/data/structured \
  --lifecycle-log freshrank/data/metadata/ingestion_events.jsonl \
  --overwrite \
  --verbose
```

Important files generated:

- `graph_nodes.jsonl`: node records (`Document`, `Heading`, `chunk:*`, slot-derived entities) with anchors that point back to parsed chunks.
- `graph_edges.jsonl`: relationships such as `HAS_CHUNK`, `HAS_SUBHEADING`, and slot-neighbour links.
- `graph_metadata.json`: document-level counts used by the GraphRAG planner.

Every successful build updates the lifecycle log with a `served_at` timestamp to align with freshrank’s SLA tracking.

## 5. Querying via the FastAPI Service

The `/kg/query` endpoint now orchestrates:

1. Upstream intent parsing and rewrite (`intent_engine`).
2. Graph filters derived from the parsed slots.
3. Execution of the GraphRAG planner which blends slot-filtered nodes, their neighbourhood evidence, and vector search hits.

To run the service locally:

```bash
uvicorn kg_fusion.app.main:app --reload --port 8200
```

Example request (via `httpie`):

```bash
http POST :8200/kg/query text="我想了解福瑞保2023的住院津贴保障" topk:=10
```

Response highlights:

- `plan.graph_plan`: structured steps (`slot-filter`, `graph-neighbourhood`, `vector-boost`) with candidate counts.
- `results[]`: unified hits including anchors, preview text, and `score_breakdown` (graph, semantic, recency, regulatory components).
- `debug.fusion`: raw GraphRAG diagnostics (filtered node IDs, semantic chunk IDs, and vector backend payloads).

These fields are already compatible with downstream freshrank tooling and can be surfaced directly inside Dify flows for transparency.

## 6. Regenerating Artefacts Incrementally

- To add new documents, drop PDFs into `data/raw/` and rerun `parse_layout.py` with `--overwrite` (or without to keep untouched artefacts). The lifecycle tracker will append entries or refresh timestamps accordingly.
- Run `build_graph.py` afterwards to ingest the new structured JSON into the graph store.
- If you prefer append-only behaviour, omit `--overwrite`; the graph store will grow incrementally.

## 7. Troubleshooting

- **Missing dependencies**: the parser logs whether layout/ocr hooks are active. Use `--verbose` to inspect fallback decisions.
- **Lifecycle log conflicts**: delete the relevant lines in `ingestion_events.jsonl` or run `build_graph.py --overwrite` to recompute a clean state.
- **Vector backend**: `GraphRAGPlanner` delegates semantic search to `kg_fusion.app.vector_backend.search`. Ensure the index files under `data/index/` are up-to-date or implement the backend to call your retriever of choice.

With these steps you can reproduce the structured parses, graph artefacts, and query responses added in the latest development phases.

## 8. Phase 3 – Semantic Fusion & Regulatory Signals

Phase 3 introduces a weighted semantic retriever and regulatory-aware scoring so the `/kg/query` endpoint can surface higher quality evidence for freshrank and Dify flows.

### 8.1 Semantic Retriever

- The new `SemanticRetriever` (see `kg_fusion/app/semantic.py`) expands user queries with slot-derived phrases (product name,版本、权益字段等) and intent hints (如“免责条款”“理赔流程”)。
- Each expansion carries a configurable weight；the retriever executes all candidates against the vector index and merges them into a single list of semantic hits with provenance (`semantic_sources`).
- GraphRAG plans now include a `semantic-retrieve` step and debug payload listing every candidate query、命中数量以及被提升的 chunk，方便在本地或 Dify 中排查召回质量。

### 8.2 Regulatory-aware Fusion

- `kg_fusion/app/regulatory.py` loads监管标签（`freshrank/data/metadata/esg_tags.jsonl`）与 `regulatory/weights.yaml`，按 rulebook 的 multiplier/bonus 配置折算附加分。
- `fusion.run` 会在计算图谱与语义分数后套用监管加分，再叠加原有时效得分；`results[].extra.regulatory` 保存加分明细（命中的标签、confidence、贡献值），`debug.fusion.regulatory` 提供逐文档追踪信息。
- 时效打分现已返回 `debug.fusion.recency`，记录 bonus、flag 以及对应的生命周期元数据，方便检查数据质量或 SLA 异常。

### 8.3 运行提示

- 运行 `parse_layout.py` 和 `build_graph.py` 后即可直接启动 `uvicorn`，无需额外配置；若替换向量模型，可通过 `EMB_MODEL` 环境变量指向新的 SentenceTransformer。
- 可通过 `REG_TAGS`、`REG_RULEBOOK` 环境变量覆盖默认的监管标签与权重文件，以便在本地或测试环境中快速试验不同策略。

## 9. Phase 4 – Recency & Regulatory Deepening

Phase 4 将 freshrank 的生命周期指标与 GraphRAG 的融合结果对齐，提供面向 Dify 展示的时效/监管解释字段。

### 9.1 Recency 评分 & SLA 追踪

- `kg_fusion/app/recency.py` 现在会同时加载 `recency_meta.csv` 与 `freshrank/data/metadata/ingestion_events.jsonl`，将 `parsed_at / metadata_ready_at / served_at` 写入 `score_breakdown.recency.timeline`，并基于 `weights.yaml` 中的 `recency.curve` 配置计算衰减曲线、窗口惩罚与 `sla_flags`。
- `/kg/query` 返回的 `results[].score_breakdown.recency` 包含 `bonus`、`decay_curve`（age、decay_weight 等）以及 SLA 触发标记，可直接在 Dify 中展示“是否按 20 分钟上线”、“是否触发过期降权”等诊断信息。
- `extra.recency` 字段与 `debug.fusion.recency` 节点同步输出原始生命周期元数据，便于排查解析/建图流程是否存在延迟。

### 9.2 监管标签扩展

- `kg_fusion/app/regulatory.py` 和 freshrank 的 `regulatory_weight.py` 均改为解析多来源标签（ESG、consumer_protection、sales_compliance 等），并使用 `regulatory.tag_weights` 的数据驱动权重替代固定常量；贡献明细会写入 `extra.regulatory.contributions`。
- 更新后的 `regulatory_score` 会同时返回乘法因子与附加分，并在 `score_breakdown.regulatory` 中显示总得分及 confidence，帮助在 Dify 中解释文档为何加权。
- 配置层新增 `default_confidence`、`min_bonus` 等项，可通过修改 `freshrank/regulatory/weights.yaml` 快速调参，无需改动代码。

### 9.3 Dify 集成提示

- 在 Dify rerank 节点中读取 `score_breakdown.recency.decay_curve` 与 `score_breakdown.recency.sla_flags`，即可用表格/提醒卡片展示生命周期健康状况。
- `freshrank/dify_tool_config.json` 已补充验证步骤，提示在前端呈现 recency/regulatory 贡献字段，确保工作流演示时能够向业务方解释排序依据。

## 10. Phase 5 – Evaluation & Workflow Hardening

Phase 5 增加了可回放的端到端评测脚本、权重调参样例以及 Dify Flow 模板，确保在交付前能够验证 `/kg/query → /rerank` 的一致性并把 recency/监管诊断显式呈现给运营方。

### 10.1 `/kg/query → /rerank` 回放脚本

- `kg_fusion/eval/replay_pipeline.py` 会读取 `eval/cases/phase5_pipeline.jsonl` 中的代表性意图，将其顺序调用 `/kg/query` 和 freshrank `/rerank`，并输出精确率/覆盖率与 `score_breakdown` 一致性校验。
- 默认元数据覆盖文件为 `eval/cases/doc_metadata.json`，可在本地加入真实文档的发布时间、生效时间及监管标签命中，以模拟实际权重表现。
- 执行 `python kg_fusion/eval/replay_pipeline.py --quiet` 会在 `eval/reports/phase5_eval.json` 生成汇总报告，可作为发布前的回归基线；若想放大候选集合，可通过 `--topk` 覆盖逐 case 的 `topk` 设置。

### 10.2 rerank 权重校准

- `freshrank/regulatory/weights.yaml` 现已对 `consumer_protection`、`sales_compliance` 等标签配置乘法权重 + 附加分，使得投诉处理、销售合规类文档能够被明显提升；参数调整无需改动代码即可通过回放脚本验证效果。
- 回放脚本会在报告中写出每个 case 的 `rerank_precision` / `rerank_recall`，并标记 `breakdown_ok`，便于在表格中对比调参前后的差异；若你希望扩充真实语料，只需新增 JSONL case 与对应的 `doc_metadata` 条目即可。

### 10.3 Dify Flow 模板 & 运行看板

- `docs/dify_flow_phase5.json` 给出了可直接导入 Dify 的工作流模板：节点顺序为 `UserQuery → IntentParse → GraphRAGQuery → LifecycleGuard → FreshrankRerank → AnswerSynthesizer`，并附带 `Diagnostics` 面板展示 recency/监管权重与 SLA 告警。
- 模板中的 `LifecycleGuard` code 节点会遍历 `score_breakdown.recency.sla_flags`，一旦发现摄取/解析超 SLA 即在仪表板抛出告警；运营人员可据此追踪数据链路健康度。
- 将 `freshrank/dify_tool_config.json` 中的新提示与模板结合，即可在 Dify 前端呈现“权重拆解 + 健康监控”的完整演示路径。
