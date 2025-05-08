import pandas as pd
import argparse
import bitsandbytes as bnb
import accelerate as aclrt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime
from huggingface_hub import snapshot_download
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import time
import torch


class FineTuningClassifier:
    def __init__(self, model, dataset_path, examples_path, csv_result_file, test_mode="zero"):
        match model:
            case "gemma2":
                self.model_id = "google/gemma-2-2b-it"
                self.model_name = "gemma2"
            case "gemma3_1b":
                self.model_id = "google/gemma-3-1b-it"
                self.model_name = "gemma3_1b"
            case "gemma3_4b":
                self.model_id = "google/gemma-3-4b-it"
                self.model_name = "gemma3_4b"
            case _:      #default case
                self.model_id = "google/gemma-3-1b-it"
                self.model_name = "gemma3_1b"

        self.dataset_path = dataset_path
        self.csv_result_file = csv_result_file

        #select best available device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print("Using {} device".format(self.device))

        self.test_mode = test_mode

        if (self.model_name=="gemma3_4b"):
            # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False
            )

            print("Using quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
            ).to(self.device).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float32).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # load dataset, examples and set categories
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

        if test_mode == "few" or test_mode == "def-few":
            #Few-shot test with examples given INSIDE the prompt
            self.adjusted_example_pool_size = 1

            # get 1 example per category
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
        # prompt = (  #circa 73% con gemma3 **
        #     f"Classify the following automotive failure into one of these categories: \n"
        #     f"{', '.join(self.categories)}.\n\n"
        # )

        prompt = (  #circa 88% con gemma3
            f"Classify the following automotive failure: "
            f"{phrase}\n"
            "into one of these categories: "
            f"{', '.join(self.categories)}.\n"
            "using the following examples:\n"
        )

        if self.adjusted_example_pool_size > 0:
            prompt += f"Examples:\n"
            for example in examples:
                # prompt += f"Failure: {example['phrase']}\nCategory: {example['category']}\n\n"
                prompt += f"{example['category']}: {example['phrase']}\n"

        # prompt += f"Failure: {phrase}\nCategory:" #circa 73% con gemma3 **
        return prompt


    def build_zero_shot_prompt(self, phrase):
        """
         Prompt construction function for zero-shot test.
        :param phrase:
        :return:
        """
        # prompt = (  #circa 76% con gemma 3
        #     f"Classify the following automotive failure into one of these categories:\n"
        #     f"{', '.join(self.categories)}.\n\n"
        #     f"Failure: {phrase}\nCategory:"
        # )

        prompt = (  #circa 90.91% con gemma 3
            f"Classify the following automotive failure: "
            f"{phrase}\n"
            "into one of these categories: "
            f"{', '.join(self.categories)}.\n\n"
        )

        return prompt


    def build_definitions_prompt(self, phrase):
        """
         Prompt construction function for test with category definitions.
        :param phrase:
        :return:
        """
        # prompt = ( #circa 51% con gemma3
        #     f"Classify the automotive failure into one of these categories:\n"
        #     f"{', '.join(self.categories)}.\n"
        #     f"Definitions of the categories:\n"
        #     "Fuel System Problems: Issues related to the delivery, regulation, or combustion of fuel in the engine. These problems can arise from components such as the fuel pump, fuel injectors, fuel filter, or fuel lines. Symptoms often include poor fuel efficiency, engine stalling, or difficulty starting. \n"
        #     "Ignition System Malfunctions: Faults within the system responsible for igniting the air-fuel mixture in the engine's cylinders. This includes components like spark plugs, ignition coils, distributor caps, and crankshaft position sensors. Common symptoms include misfires, difficulty starting, or rough idling. \n"
        #     "Cooling System Anomalies: Problems affecting the system that regulates engine temperature to prevent overheating. Components include the radiator, water pump, thermostat, and coolant hoses. Indicators of issues are engine overheating, coolant leaks, or insufficient heating in the cabin.\n"
        #     "Brake System Defects: Malfunctions in the braking system that affect the vehicle's ability to slow down or stop safely. This includes defects in brake pads, rotors, calipers, or hydraulic components like the master cylinder. Symptoms include squealing noises, vibrations, or reduced braking effectiveness.\n"
        #     "Transmission Problems: Issues within the system responsible for transmitting power from the engine to the wheels, including automatic or manual transmissions. Common problems involve slipping gears, delayed shifting, or leaks in the transmission fluid. Symptoms include unusual noises and difficulty in shifting gears.\n"
        #     "Electrical/Electronic Failures: Faults in the vehicle's electrical or electronic systems, including the battery, alternator, wiring, or onboard computers. These can manifest as flickering lights, dead batteries, or malfunctioning electronic components like power windows or dashboard instruments.\n"
        #     f"Failure: {phrase}\nCategory:"
        # )

        prompt = ( #circa 92% gemma3
            f"Classify the following automotive failure: "
            f"{phrase}\n"
            "into one of these categories: "
            f"{', '.join(self.categories)}.\n\n"
            "using the following definitions:\n"
            "Fuel System Problems: Issues related to the delivery, regulation, or combustion of fuel in the engine. These problems can arise from components such as the fuel pump, fuel injectors, fuel filter, or fuel lines. Symptoms often include poor fuel efficiency, engine stalling, or difficulty starting. \n"
            "Ignition System Malfunctions: Faults within the system responsible for igniting the air-fuel mixture in the engine's cylinders. This includes components like spark plugs, ignition coils, distributor caps, and crankshaft position sensors. Common symptoms include misfires, difficulty starting, or rough idling. \n"
            "Cooling System Anomalies: Problems affecting the system that regulates engine temperature to prevent overheating. Components include the radiator, water pump, thermostat, and coolant hoses. Indicators of issues are engine overheating, coolant leaks, or insufficient heating in the cabin.\n"
            "Brake System Defects: Malfunctions in the braking system that affect the vehicle's ability to slow down or stop safely. This includes defects in brake pads, rotors, calipers, or hydraulic components like the master cylinder. Symptoms include squealing noises, vibrations, or reduced braking effectiveness.\n"
            "Transmission Problems: Issues within the system responsible for transmitting power from the engine to the wheels, including automatic or manual transmissions. Common problems involve slipping gears, delayed shifting, or leaks in the transmission fluid. Symptoms include unusual noises and difficulty in shifting gears.\n"
            "Electrical/Electronic Failures: Faults in the vehicle's electrical or electronic systems, including the battery, alternator, wiring, or onboard computers. These can manifest as flickering lights, dead batteries, or malfunctioning electronic components like power windows or dashboard instruments. \n"
        )

        return prompt

    def build_definitions_and_example_prompt(self, phrase, examples):
        """
         Prompt construction function for test with category definitions and examples.
        :param phrase:
        :param examples:
        :return:
        """
        prompt = (  #circa 51% acc
            f"Classify the following automotive failure: "
            f"{phrase}\n"
            "into one of these categories: "
            f"{', '.join(self.categories)}.\n"
            "Using the following definitions:\n"
            "Fuel System Problems: Issues related to the delivery, regulation, or combustion of fuel in the engine. These problems can arise from components such as the fuel pump, fuel injectors, fuel filter, or fuel lines. Symptoms often include poor fuel efficiency, engine stalling, or difficulty starting. \n"
            "Ignition System Malfunctions: Faults within the system responsible for igniting the air-fuel mixture in the engine's cylinders. This includes components like spark plugs, ignition coils, distributor caps, and crankshaft position sensors. Common symptoms include misfires, difficulty starting, or rough idling. \n"
            "Cooling System Anomalies: Problems affecting the system that regulates engine temperature to prevent overheating. Components include the radiator, water pump, thermostat, and coolant hoses. Indicators of issues are engine overheating, coolant leaks, or insufficient heating in the cabin.\n"
            "Brake System Defects: Malfunctions in the braking system that affect the vehicle's ability to slow down or stop safely. This includes defects in brake pads, rotors, calipers, or hydraulic components like the master cylinder. Symptoms include squealing noises, vibrations, or reduced braking effectiveness.\n"
            "Transmission Problems: Issues within the system responsible for transmitting power from the engine to the wheels, including automatic or manual transmissions. Common problems involve slipping gears, delayed shifting, or leaks in the transmission fluid. Symptoms include unusual noises and difficulty in shifting gears.\n"
            "Electrical/Electronic Failures: Faults in the vehicle's electrical or electronic systems, including the battery, alternator, wiring, or onboard computers. These can manifest as flickering lights, dead batteries, or malfunctioning electronic components like power windows or dashboard instruments. \n"
            "And using the following examples:\n"
        )

        if self.adjusted_example_pool_size > 0:
            for example in examples: #circa 78% con gemma3,
                # prompt += f"Failure: {example['phrase']}\nCategory: {example['category']}\n\n"
                prompt += f"{example['category']}: {example['phrase']}\n"

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
            case "def-few":
                prompt_text = self.build_definitions_and_example_prompt(phrase, self.few_shot_examples)
            case _:     #default case
                prompt_text = self.build_zero_shot_prompt(phrase)

        # if self.model_name == "gemma3_1b":
        #     messages = [
        #         [
        #             {
        #                 "role": "system",
        #                 "content": [{"type": "text", "text": "Respond only with the category name. No explanation."}],
        #             },
        #             {
        #                 "role": "user",
        #                 "content": [{"type": "text", "text": prompt_text}],
        #             },
        #         ],
        #     ]

        # if self.model_name == "gemma2":
        messages = [
            {"role": "user", "content": f"Respond only with the category name. No explanation.\n {prompt_text}"},
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

        fig, ax = plt.subplots(figsize=(12, 10))

        disp.plot(ax=ax, xticks_rotation=45)

        plt.tight_layout()

        base_path="./graphs"

        match self.test_mode:
            case "few":
                plt.savefig(f"{base_path}/{self.model_name}_vehicularFailures_few-shot.png")
            case "zero":
                plt.savefig(f"{base_path}/{self.model_name}_vehicularFailures_zero-shot.png")
            case "def":
                plt.savefig(f"{base_path}/{self.model_name}_vehicularFailures_definitions-test.png")
            case "def-few":
                plt.savefig(f"{base_path}/{self.model_name}_vehicularFailures_definitions-and_examples-test.png")

    def log_results_to_csv(self, accuracy, process_time):
        """
         Log accuracy results to a CSV file.
        :param accuracy:
        :param process_time:
        :return:
        """
        new_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d/%H:%M:%S"),
            "Test mode": self.test_mode,
            "Model": self.model_name,
            "Process_time(s)": process_time,
            "Accuracy (%)": round(accuracy, 2),
        }

        if os.path.exists(self.csv_result_file):
            df = pd.read_csv(self.csv_result_file, sep=";")
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(self.csv_result_file, index=False, sep=";")


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
            # print(f"{i + 1}. \"{phrase}\" →  {category}")
        t1_stop = time.perf_counter()  # process_time()
        process_time = t1_stop - t1_start

        accuracy, correct, total = self.evaluate_accuracy(predictions)
        print(f"\nModel Accuracy: {accuracy:.2f}% ({correct}/{total} correct) |Test Mode: {self.test_mode} |Process time: {process_time:.2f} seconds")
        self.log_results_to_csv(accuracy, process_time)

        self.sklearn_metrics(self.categories, self.test_df['category'].to_list(), predictions)


if __name__ == "__main__":
    #snapshot_download(repo_id="google/gemma-3-1b-it") #Use only the first time to install the model locally

    parser = argparse.ArgumentParser(description="Fine-tuning classifier for automotive failure detection.")

    #change model used
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3_1b",  # Default value from the classifier initialization
        choices=["gemma2", "gemma3_1b", "gemma3_4b"],
        help="Choose model between gemma2, gemma3_1b. Default is 'gemma3_1b'."
    )

    #change examples used
    parser.add_argument(
        "--examples_path",
        type=str,
        default="context_examples.csv",
        choices=["examples.csv", "context_examples.csv"],
        help="Path to the examples CSV file. Default is 'examples.csv'."
    )

    #change test mode
    parser.add_argument(
        "--test_mode",
        type=str,
        default="zero",
        choices=["zero", "few", "def", "def-few"],
        help="Testing mode: 'zero', 'few', 'def', or 'def-few'. Default is 'def'."
    )

    # Parse arguments
    args = parser.parse_args()

    classifier = FineTuningClassifier(
        model = args.model,
        dataset_path = "dataset.csv",
        examples_path = args.examples_path,
        csv_result_file = "./accuracy_example_pool_sizes.csv",
        test_mode = args.test_mode
    )

    classifier.classify_and_evaluate()
