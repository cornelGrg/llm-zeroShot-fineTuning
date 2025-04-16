import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# switch between zero-shot and few-shot
USE_FEW_SHOT = True

def build_few_shot_prompt(phrase, categories, examples):
    prompt = (
        f"Classify the following automotive failure into one of these categories:\n"
        f"{', '.join(categories)}.\n\n"

        f"Examples:\n"
    )
    if USE_FEW_SHOT:
        for example in examples:
            ex_phrase = example['phrase']
            ex_category = example['category']
            prompt += f"Vulnerability: {ex_phrase}\nCategory: {ex_category}\n\n"

    prompt += f"Vulnerability: {phrase}\nCategory:"
    return prompt

def classify_phrase(phrase, categories, tokenizer, model, examples):
    prompt_text = build_few_shot_prompt(phrase, categories, examples)

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Respond only with the category name. No explanation."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            },
        ],
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to("mps") for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=1000)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    result_text = decoded[0].strip()

    lines = result_text.splitlines()
    lines = [line for line in lines if line.strip() and "model" not in line.lower()]

    category = lines[-1].strip("* ").strip() if lines else "Unknown"
    return category

def evaluate_accuracy(df, predictions):
    print("\n EVALUATING ACCURACY:")
    correct = 0
    total = len(df)

    for i in range(total):
        expected = df.iloc[i]['category'].strip().lower()
        predicted = predictions[i].strip().lower()
        print(f"\n\t expected: {expected} \t predicted: {predicted}")
        if predicted == expected:
            correct += 1

    accuracy = correct / total * 100
    print(f"\nModel Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

if __name__ == "__main__":
    categories = [
        "Fuel System Problems",
        "Ignition System Malfunctions",
        "Cooling System Anomalies",
        "Brake System Defects",
        "Transmission Problems",
        "Electrical/Electronic Failures"
    ]

    #csv contains (phrases and categories columns)
    df = pd.read_csv("dataset.csv")

    #select examples for training
    few_shot_examples = df.head(10).to_dict(orient='records') if USE_FEW_SHOT else None
    #use the reamining examples (the ones used for training are not case of study)
    test_df = df.iloc[10:].reset_index(drop=True)

    model_id = "google/gemma-3-1b-it"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("mps").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Classification Results:\n")
    predictions = []
    for i, row in test_df.iterrows():
        phrase = row['phrase']
        category = classify_phrase(phrase, categories, tokenizer, model, few_shot_examples)
        predictions.append(category)
        print(f"{i+1}. \"{phrase}\" →  {category}")

    evaluate_accuracy(test_df, predictions)
