import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def classify_phrase(phrase, categories, tokenizer, model):
    prompt_text = (
        f"Classify the following software vulnerability into one of these categories:\n"
        f"{', '.join(categories)}.\n\n"
        f"Vulnerability: {phrase}\nCategory:"
    )

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
        outputs = model.generate(**inputs, max_new_tokens=200)

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
    df = pd.read_csv("dataset.csv").head(10)

    model_id = "google/gemma-3-1b-it"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("mps").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Classification Results:\n")
    predictions = []
    for i, row in df.iterrows():
        phrase = row['phrase']
        category = classify_phrase(phrase, categories, tokenizer, model)
        predictions.append(category)
        print(f"{i+1}. \"{phrase}\" →  {category}")

    evaluate_accuracy(df, predictions)
