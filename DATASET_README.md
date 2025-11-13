# 口语↔规范 对齐数据集（保险/寿险）

本数据集由自动抽取与可控生成构建，用于“攻关任务二：口语化查询意图转换引擎”的训练与评测。
依据项目命题要求（如“快返型→生存金5年内返还”“养老社区保险=CCRC保险”等）构建术语本体、规范问句模板、
以及口语变体，并保留 Dify 在线反馈闭环的接口。

- 总条数：2100（≥1500）
- 文件：
  - `intent_dataset_zh_insurance_v1.csv` / `.jsonl`
  - `ontology_insurance_zh.yaml`
  - `synonyms_insurance_zh.tsv`
  - `templates_canonical_zh.json`
  - `feedback_seed.csv`（在线闭环种子）

## 字段说明（CSV/JSONL）
- `id`: 唯一ID
- `intent`: 意图分类（ELIGIBILITY/COVERAGE_EXCLUSION/BENEFIT_RETURN/...）
- `colloquial_query`: 口语化查询（含错别字/拼音等）
- `canonical_query`: 规范术语问句（检索/改写目标）
- `rewrite_suggestion`: 建议改写（默认=canonical_query）
- `slots`: 槽位（JSON）：product_name/product_line/version_year/field/benefit_type
- `ontology_terms`: 触发的本体术语集合
- `source_hint`: 来源提示（产品名/文件名线索）
- `quality`: 质量标记（from_filename+limited_pdf / synthetic_boost）

## 数据来源与方法
1) 从 `shuju.zip` 中解析文件名与少量页文本，抽取产品名/关键词，用以生成规范问句模板与槽位候选；
2) 基于术语本体与同义映射，使用可控生成产生多样口语变体（包含方言化、错别字、拼音变体）；
3) 弱监督方式为每条口语提供规范改写与意图/槽位标签，用作训练与评测；
4) 预留 `feedback_seed.csv` 格式，便于 Dify 在线回收点击/成功日志，闭环增量扩充。

## 许可与注意
- 本数据仅作内部模型训练/评测使用；
- 需结合实际文档校正极少数可能噪声样本；
- 与项目“目标二/目标三/目标四”在 Dify 中工具化集成。

