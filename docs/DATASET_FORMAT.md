# Dataset Format / 数据集格式

NightCruise exports dataset records in JSONL, JSON, or TXT.

NightCruise 支持导出 JSONL、JSON 或 TXT。

## JSONL

Each line is a JSON object:

```json
{"text":"...generated sample...","validation":{"score":10.0,"issues":[],"confidence":"high"},"discipline":{"major":"Computer Science","sub":"Artificial Intelligence"}}
```

Fields:

- `text`: generated bilingual dataset sample.
- `validation`: optional validator score and issues.
- `discipline`: heuristic discipline classification.

字段：

- `text`：生成出的中英双语训练样本。
- `validation`：可选验证器评分与问题列表。
- `discipline`：启发式学科分类结果。

## Training conversion / 训练格式转换

Different training frameworks expect different formats. The raw JSONL can be converted to instruction-style or chat-style formats later.

不同训练框架需要不同格式。NightCruise 的原始 JSONL 可以后续转换成 instruction 格式或 messages/chat 格式。
