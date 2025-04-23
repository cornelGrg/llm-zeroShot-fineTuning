import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from huggingface_hub import snapshot_download
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import time
import torch


class FineTuningClassifier:
    def __init__(self, model_id, dataset_path, examples_path, csv_result_file, example_pool_size=0, test_mode="zero"):
        self.model_id = model_id
        self.dataset_path = dataset_path
        self.csv_result_file = csv_result_file

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print("Using {} device".format(self.device))

        self.test_mode = test_mode

        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # load dataset and set categories
        self.test_df = pd.read_csv(dataset_path)
        self.df_examples = pd.read_csv(examples_path)
        self.categories = [
            "Fuel System Problems",
            "Ignition System Malfunctions",
            "Cooling System Anomalies",
            "Brake System Defects",
            "Transmission Problems",
            "Electrical/Electronic Failures"
        ]

        if test_mode == "few": #     print(self.test_df.size)
            #Few-shot test with examples given INSIDE the prompt
            self.adjusted_example_pool_size = example_pool_size

            min_count_per_category = min(
                self.df_examples['category'].value_counts().get(cat, 0) for cat in self.categories
            )

            self.adjusted_example_pool_size = min(example_pool_size, min_count_per_category)
            if self.adjusted_example_pool_size < example_pool_size:
                print(f"Warning: Not enough examples in some categories. "
                      f"Reducing example_pool_size to {self.adjusted_example_pool_size}.")

            # get x examples per category
            few_shot_rows = []
            for category in self.categories:
                examples = self.df_examples[self.df_examples['category'] == category].head(self.adjusted_example_pool_size)
                few_shot_rows.append(examples)

            few_shot_df = pd.concat(few_shot_rows).reset_index(drop=True)
            self.few_shot_examples = few_shot_df.to_dict(orient='records')

        else:
            #Zero-shot test with no examples given or Definition-test with definitions
            self.adjusted_example_pool_size = 0
            self.few_shot_examples = 0


    def build_few_shot_prompt(self, phrase, examples):
        """
         Prompt construction function for few-shot test.
        :param phrase:
        :param examples:
        :return:
        """
        prompt = (
            f"Classify the following automotive failure into one of these categories:\n"
            f"{', '.join(self.categories)}.\n\n"
        )
        if self.adjusted_example_pool_size > 0:
            prompt += f"Examples:\n"
            for example in examples:
                prompt += f"Failure: {example['phrase']}\nCategory: {example['category']}\n\n"

        prompt += f"Failure: {phrase}\nCategory:"
        print(f"prompt FEW: {prompt}")
        return prompt


    def build_zero_shot_prompt(self, phrase):
        """
         Prompt construction function for zero-shot test.
        :param phrase:
        :return:
        """
        prompt = (
            f"Classify the following automotive failure into one of these categories:\n"
            f"{', '.join(self.categories)}.\n\n"
            f"Failure: {phrase}\nCategory:"
        )

        print(f"prompt ZERO: {prompt}")

        return prompt


    def build_definitions_prompt(self, phrase):
        """
         Prompt construction function for test with category definitions.
        :param phrase:
        :return:
        """
        prompt = (
            f"Classify the following automotive failure into one of these categories:\n"
            f"{', '.join(self.categories)}.\n\n"
            f"Definitions of the categories:\n"
            "Fuel System Problems: Issues related to the delivery, regulation, or combustion of fuel in the engine. These problems can arise from components such as the fuel pump, fuel injectors, fuel filter, or fuel lines. Symptoms often include poor fuel efficiency, engine stalling, or difficulty starting. \n",
            "Ignition System Malfunctions: Faults within the system responsible for igniting the air-fuel mixture in the engine's cylinders. This includes components like spark plugs, ignition coils, distributor caps, and crankshaft position sensors. Common symptoms include misfires, difficulty starting, or rough idling. \n",
            "Cooling System Anomalies: Problems affecting the system that regulates engine temperature to prevent overheating. Components include the radiator, water pump, thermostat, and coolant hoses. Indicators of issues are engine overheating, coolant leaks, or insufficient heating in the cabin.\n",
            "Brake System Defects: Malfunctions in the braking system that affect the vehicle's ability to slow down or stop safely. This includes defects in brake pads, rotors, calipers, or hydraulic components like the master cylinder. Symptoms include squealing noises, vibrations, or reduced braking effectiveness.\n",
            "Transmission Problems: Issues within the system responsible for transmitting power from the engine to the wheels, including automatic or manual transmissions. Common problems involve slipping gears, delayed shifting, or leaks in the transmission fluid. Symptoms include unusual noises and difficulty in shifting gears.\n",
            "Electrical/Electronic Failures: Faults in the vehicle's electrical or electronic systems, including the battery, alternator, wiring, or onboard computers. These can manifest as flickering lights, dead batteries, or malfunctioning electronic components like power windows or dashboard instruments.\n\n"
            f"Failure: {phrase}\nCategory:"
        )

        return prompt


    def classify_phrase(self, phrase):
        """
        Use LLM model to classify the phrase given .
        :param phrase:
        :return category:
        """

        match self.test_mode:
            case "zero":
                prompt_text = self.build_zero_shot_prompt(phrase)
            case "few":
                prompt_text = self.build_few_shot_prompt(phrase, self.few_shot_examples)
            case "def":
                prompt_text = self.build_definitions_prompt(phrase)

        #Zero-shot or Few_shot test with examples
        # prompt_text = self.build_few_shot_prompt(phrase, self.few_shot_examples)
        # prompt_text = self.build_zero_shot_prompt(phrase)

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

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=300)

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        result_text = decoded[0].strip()

        lines = result_text.splitlines()
        lines = [line for line in lines if line.strip() and "model" not in line.lower()]

        category = lines[-1].strip("* ").strip() if lines else "Unknown"
        return category


    def evaluate_accuracy(self, predictions):
        """
        Given a list of predictions, evaluate accuracy comparing with the dataset.
        :param predictions:
        :return: accuracy
        """
        correct = 0
        total = len(self.test_df)

        for i in range(total):
            expected = self.test_df.iloc[i]['category'].strip().lower()
            predicted = predictions[i].strip().lower()
            if predicted == expected:
                correct += 1

        accuracy = correct / total * 100
        return accuracy, correct, total

    def sklearn_metrics(self, topics, expected_labels, predicted_labels):
        """
        Calculate performance metrics using sklearn (confusion matrix, precision score, recall score, accuracy score).
        :return:
        """
        accuracy = accuracy_score(expected_labels, predicted_labels)

        print(f"Accuratezza del modello: {accuracy:.4f}")

        precision_per_label = precision_score(expected_labels, predicted_labels, labels=topics, average=None, zero_division=0)

        recall_per_label = recall_score(expected_labels, predicted_labels, labels=topics, average=None, zero_division=0)

        f1_per_label = f1_score(expected_labels, predicted_labels, labels=topics, average=None, zero_division=0)

        # Stampa i risultati per ogni label

        for label, precision, recall, f1 in zip(topics, precision_per_label, recall_per_label, f1_per_label):
            print(f"Risultati per '{label}':")

            print(f"  Precisione: {precision:.4f}")

            print(f"  Recall: {recall:.4f}")

            print(f"  F1-Score: {f1:.4f}")

            # confusion matrix

        confusion = confusion_matrix(expected_labels, predicted_labels, labels=topics)

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=topics)

        # Create a larger figure and pass it to disp.plot()
        fig, ax = plt.subplots(figsize=(12, 10))  # Adjust the figure size here

        disp.plot(ax=ax, xticks_rotation=45)  # Rotate labels for readability,  cmap="Blues"

        # Optional: Adjust layout
        plt.tight_layout()

        base_path="./graphs"

        # Save the figure
        match self.test_mode:
            case "few":
                plt.savefig(f"{base_path}/Gemma-3-1b-IT_vehicularFailures_few-shot_{self.adjusted_example_pool_size}examples.png")
            case "zero":
                plt.savefig(f"{base_path}/Gemma-3-1b-IT_vehicularFailures_zero-shot.png")
            case "def":
                plt.savefig(f"{base_path}/Gemma-3-1b-IT_vehicularFailures_definitions-test.png")

        # disp.plot().figure_.savefig(f"Gemma-3-1b-IT_vehicularFailures.png")

    def log_results_to_csv(self, accuracy, process_time):
        """
         Log accuracy results to a CSV file.
        :param accuracy:
        :param process_time:
        :return:
        """
        new_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d/%H:%M:%S"),
            "Example Pool Size": self.adjusted_example_pool_size,
            "Test mode": self.test_mode,
            "Accuracy (%)": round(accuracy, 2),
            "Process_time(s)": process_time
        }

        if os.path.exists(self.csv_result_file):
            df = pd.read_csv(self.csv_result_file)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(self.csv_result_file, index=False)


    def classify_and_evaluate(self):
        """
        Perform classification test on the dataset and print accuracy data
        """
        predictions = []

        t1_start = time.perf_counter()  # process_time()
        for i, row in self.test_df.iterrows():
            phrase = row['phrase']
            category = self.classify_phrase(phrase)
            predictions.append(category)
            print(f"{i + 1}. \"{phrase}\" →  {category}")
        t1_stop = time.perf_counter()  # process_time()
        process_time = t1_stop - t1_start

        accuracy, correct, total = self.evaluate_accuracy(predictions)
        print(f"\nModel Accuracy: {accuracy:.2f}% ({correct}/{total} correct) |Test Mode: {self.test_mode} |EXAMPLE POOL SIZE: {self.adjusted_example_pool_size}\n |Process time: {process_time:.2f} seconds")
        self.log_results_to_csv(accuracy, process_time)

        self.sklearn_metrics(self.categories, self.test_df['category'].to_list(), predictions)


if __name__ == "__main__":
    # snapshot_download(repo_id="google/gemma-3-1b-it") #Use only the first time to install the model locally
    classifier = FineTuningClassifier(
        model_id="google/gemma-3-1b-it",
        dataset_path="dataset.csv",
        examples_path="examples.csv",
        csv_result_file="./accuracy_example_pool_sizes.csv",
        example_pool_size=0, #number of examples used per category (if higher than available the maximum will be used), used for "zero" mode
        test_mode="def" #choose testing mode: "zero"=zero-shot, "few"=few-shot, "def"=definitions-test
    )

    classifier.classify_and_evaluate()
