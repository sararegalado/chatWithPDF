
# Load ground truth dataset
with open("src/E3/t1_intent_classification_v2/ground_truth/intents_dataset.json", "r", encoding="utf-8") as f:
    import json
    ground_truth_data = json.load(f)

# 10% of the data for examples for few-shot learning
num_examples = int(0.1 * len(ground_truth_data))
example_data = ground_truth_data[:num_examples]
# save examples to a separate file
with open("src/E3/t1_intent_classification_v2/ground_truth/few_shot_examples.json", "w", encoding="utf-8") as f:
    json.dump(example_data, f, ensure_ascii=False, indent=4)

# The rest for evaluation
eval_data = ground_truth_data[num_examples:]
with open("src/E3/t1_intent_classification_v2/ground_truth/evaluation_dataset.json", "w", encoding="utf-8") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=4)

print(f"Prepared data: {len(example_data)} examples for few-shot learning, {len(eval_data)} samples for evaluation.")