import pandas as pd
import argparse
# import bitsandbytes as bnb
import accelerate as aclrt
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Gemma3nForCausalLM
from datetime import datetime
from datasets import load_dataset, Dataset
import peft
import trl
from huggingface_hub import snapshot_download
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import time
import torch
from trl import SFTTrainer


def create_model_and_tokenizer(model_name, device, trained, perform_new_training):
    """
    Create and initialize model and tokenizer based on model name and device.
    
    Args:
        model_name: Name of the model ("gemma2", "gemma3_1b", "gemma3n_e2b_it")
        device: Device to load the model on
        trained (bool): Whether to load a fine-tuned model.
        perform_new_training (bool): Whether to use already existing trained model or overwrite it.
    
    Returns:
        tuple: (model, tokenizer, model_id, model_name)
    """
    match model_name:
        case "gemma2":
            model_id = "google/gemma-2-2b-it"
            model_name_normalized = "gemma2"
        case "gemma3_1b":
            model_id = "google/gemma-3-1b-it"
            model_name_normalized = "gemma3_1b"
        case "gemma3_4b":
            model_id = "google/gemma-3-4b-it"
            model_name_normalized = "gemma3_4b"
        case "gemma3n_e2b_it":
            model_id = "google/gemma-3n-E2B-it"
            model_name_normalized = "gemma3n_e2b_it"
        case _:      #default case
            model_id = "google/gemma-3-1b-it"
            model_name_normalized = "gemma3_1b"
    
    # # Special handling for Gemma-3n-E2B model due to ALT-UP quantization incompatibility
    # if model_name_normalized == "gemma3n_e2b_it":
    #     print("Note: Loading gemma-3n-E2B-it without quantization")
    #     # Try loading entirely on GPU without quantization
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_id,
    #         torch_dtype=torch.float16,
    #         low_cpu_mem_usage=True,
    #     ).to(device).eval()
    #     print("Model loaded without quantization on GPU")
    # else:
    #     # Quantization configuration for other models
    #     quant_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_use_double_quant=False
    #     )

    #     # Load the model with quantization and low CPU memory usage
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_id,
    #         quantization_config=quant_config,
    #         low_cpu_mem_usage=True,  # Enable low CPU memory usage
    #     ).to(device).eval()

    if model_name_normalized == "gemma3n_e2b_it":
        print("Loading base model Gemma-3n-E2B (without quantization due compatibility issues)")
        # Load without quantization for Gemma-3n-E2B
        model = Gemma3nForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # Use float16 instead of quantization
            low_cpu_mem_usage=True,
            attn_implementation='eager',
            device_map="auto"  # Let transformers handle device placement
        ).eval()
    else:
        print(f"Loading base model (quantized): {model_id}")
    
        # Quantization configuration for other models
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False
        )

        # Load the model with quantization and low CPU memory usage
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            attn_implementation='eager',
            low_cpu_mem_usage=True,  # Enable low CPU memory usage
            device_map="auto" # Let transformers handle device placement
        ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if trained:
        model_save_path = f"trainedModels/GEMMA/{model_name_normalized}"
        if os.path.exists(model_save_path) and not perform_new_training:
            print(f"Loading existing fine-tuned model from {model_save_path}")
            model = peft.PeftModel.from_pretrained(model, model_save_path)
            print("Fine-tuned model loaded.")
    
    return model, tokenizer, model_id, model_name_normalized


class FineTuningClassifier:
    def __init__(self, model, tokenizer, model_id, model_name, device, dataset_path, trained, examples_path, csv_result_file, test_mode="zero", perform_new_training=False):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_name = model_name
        self.device = device
        self.dataset_path = dataset_path
        self.csv_result_file = csv_result_file
        self.test_mode = test_mode
        self.trained = trained

        print(f"Running in: {test_mode} mode [{'trained]' if self.trained else 'base (untrained)]'}")
        print(f"Running with {self.model_id} model")
        
        # load dataset, examples and set categories
        self.test_df = pd.read_csv(dataset_path, sep="\t")
        
        self.categories = [
            "Fuel System Problems",
            "Ignition System Malfunctions",
            "Cooling System Anomalies",
            "Brake System Defects",
            "Transmission Problems",
            "Electrical/Electronic Failures"
        ]
        
        if self.trained:
            self.train_df = pd.read_csv(examples_path, sep="\t")
            print(f"Loaded {len(self.train_df)} examples for training from {examples_path}.")
            print(f"Using {len(self.test_df)} examples for testing from {dataset_path}.")

            model_save_path = f"trainedModels/GEMMA/{self.model_name}"
            # Training is needed if we force it or if the model doesn't exist
            if perform_new_training or not os.path.exists(model_save_path):
                if not os.path.exists(model_save_path):
                    print("No trained model found. Starting training...")
                else:
                    print("Overwriting existing model. Starting new training...")
                self.trainModel(model_save_path)
                print(f"Loading fine-tuned model from {model_save_path}")
                self.model = peft.PeftModel.from_pretrained(self.model, model_save_path)

        else:
            print ("Using base (untrained) model")
            print(f"Using {len(self.test_df)} examples for testing from {dataset_path}.")
            self.train_df = None # No training for base model

        print("Using {} device".format(self.device))

        if test_mode == "few" or test_mode == "def-few":
            #Few-shot test with examples given INSIDE the prompt
            self.adjusted_example_pool_size = 1
            
            # Load context examples specifically for few-shot modes
            context_examples_df = pd.read_csv("context_examples.tsv", sep="\t")

            # get 1 example per category
            few_shot_rows = []
            for category in self.categories:
                examples = context_examples_df[context_examples_df['category'] == category].head(self.adjusted_example_pool_size)
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
        prompt = (
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
            "With the following examples:\n"
        )

        if self.adjusted_example_pool_size > 0:
            for example in examples: #circa 75% con gemma3,
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
            outputs = self.model.generate(**inputs, max_new_tokens=30)

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        result_text = decoded[0].strip()

        lines = result_text.splitlines()
        lines = [line for line in lines if line.strip() and "model" not in line.lower()]

        category = lines[-1].strip("* ").strip() if lines else "Unknown"
        return category

    def __get_peft_params_default(self):
        # define default LORA parameters
        """return {
            'r':16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            'target_modules':["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj" ],
            'lora_alpha':16,
            'lora_dropout':0,     # Supports any, but = 0 is optimized
            'bias':"none",        # Supports any, but = "none" is optimized
             # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            'use_gradient_checkpointing':"unsloth",  # True or "unsloth" for very long context
            'random_state':3407, #x riproducibilità esperimento
            'use_rslora':False,  # We support rank stabilized LoRA
            'loftq_config':None  # And LoftQ
        }
        """

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return config

    def __get_train_params_default(self, modelSaveFileName):
        # define training arguments
        return TrainingArguments(
            output_dir=modelSaveFileName,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=60,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",  # Use this for WandB etc
    )

    def is_bfloat16_supported(self):
        """
        Checks if the current environment supports bfloat16 precision.
        This is dependent on the hardware (e.g., newer GPUs or TPUs).

        Returns:
            bool: True if bfloat16 is supported, False otherwise.
        """
        try:
            # Use PyTorch to check GPU capabilities
            if torch.cuda.is_available():
                cuda_capability = torch.cuda.get_device_capability()
                # Check if CUDA compute capability is 8.0 or higher (Ampere GPUs and newer)
                # Ampere GPUs (like A100) support bfloat16.
                return cuda_capability[0] >= 8
            else:
                # If no GPU is available, return False
                return False
        except ImportError:
            # PyTorch is not installed, assume no support
            return False


    def formatting_prompts_func(self, examples):
        """
        Formats the dataset into a training-friendly format for the model.
        Each row of the dataset is converted into a dictionary with a 'text' field
        that combines the phrase and its associated category into a prompt-response format.

        Returns:
            list[dict]: A list of dictionaries where each dictionary has a 'text' field
                        containing formatted strings.
        """
        texts = []
        for i in range(len(examples['phrase'])):
            phrase = examples["phrase"][i]
            category = examples["category"][i]
            # Format each row into a prompt-response string
            text = f"Classify the following automotive failure: {phrase}\ninto one of these categories: {', '.join(self.categories)}.\nCategory: {category}"
            texts.append(text)
        return {"text": texts}


    def trainModel(self, modelSaveFileName, peft_params=None, train_params=None):
        """
        effettua il fine tuning dal modello base con i dati del dataset
        :param modelSaveFileName: path where to save the model adapters
        :param peft_params: OBJ LoraConfig per parametri per le matrici da aggiungere ai pesi
        :param train_params: OBJ SFTConfig per parametri per il fine tuning (numero di epoche, lernong rate, ...)
        :return:
        """
        """IMPORT IL DATASET PER FARE IL TRAINING"""
        dataset = Dataset.from_pandas(self.train_df)
        dataset = dataset.map(self.formatting_prompts_func, batched=True)
        
        #  # --- test print ---
        # print("--- Sample from formatted training dataset ---")
        # for i in range(2): # Print the first 2 examples
        #     print(dataset[i]['text'])
        # print("------------------------------------------")
        # # -----------------------------------------

        """PARAMETRI PER IL TRAINING + LEARNING + SALVO IL MODELLO"""
        # LoRA Config
        if peft_params is None:
            peft_params = self.__get_peft_params_default()
        # Training Params
        if train_params is None:
            train_params = self.__get_train_params_default(modelSaveFileName)

        # Model: We now add LoRA adapters so we only need to update 1 to 10% of all parameters!
        self.model.config.use_cache = False # Must be false for gradient checkpointing
        base_model = get_peft_model(
            self.model,
            peft_params
        )

        max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!

        fine_tuning = SFTTrainer(
            model=base_model,
            # tokenizer=self.tokenizer,
            train_dataset=dataset,
            # dataset_text_field="phrase",  SFTConfig arg
            # max_seq_length=max_seq_length, SFTConfig arg
            # dataset_num_proc=2, SFTConfig arg
            # packing=False,  # Can make training 5x faster for short sequences.  SFTConfig arg
            args=train_params,
            peft_config=peft_params,
        )

        # Training modifica modello self.model
        fine_tuning.train()
        # Save Model
        base_model.save_pretrained(modelSaveFileName)  # salva i parametri del modello nella cartella modello
        self.tokenizer.save_pretrained(modelSaveFileName)

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
            if predicted in expected or expected in predicted:
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
        model_suffix = f"{self.model_name}{'_trained' if self.trained else ''}"

        match self.test_mode:
            case "few":
                plt.savefig(f"{base_path}/{model_suffix}_vehicularFailures_few-shot.png")
            case "zero":
                plt.savefig(f"{base_path}/{model_suffix}_vehicularFailures_zero-shot.png")
            case "def":
                plt.savefig(f"{base_path}/{model_suffix}_vehicularFailures_definitions-test.png")
            case "def-few":
                plt.savefig(f"{base_path}/{model_suffix}_vehicularFailures_definitions-and_examples-test.png")

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
            "Model": f"{self.model_name}{'_trained' if self.trained else ''}",
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

    # --- FLAG TO FORCE RETRAINING ---
    PERFORM_NEW_TRAINING = True # Set to True to force retraining even if a trained model exists

    parser = argparse.ArgumentParser(description="Fine-tuning classifier for automotive failure detection.")

    #change model used
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3_1b",  # Default value from the classifier initialization
        choices=["gemma2", "gemma3_1b", "gemma3n_e2b_it"],
        help="Choose model between gemma2, gemma3_1b. Default is 'gemma3_1b'."
    )

    # change model type used
    parser.add_argument(
        "--training",
        action='store_true', # Use as a flag: --training
        help="Choose between trained or untrained model"
    )

    #change examples used
    parser.add_argument(
        "--examples_path",
        type=str,
        default="examples.tsv",
        help="Path to the examples TSV file for training. Default is 'examples.tsv'."
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

    # Select best available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create model and tokenizer outside the class
    model, tokenizer, model_id, model_name = create_model_and_tokenizer(
        args.model, 
        device, 
        args.training, 
        PERFORM_NEW_TRAINING
    )

    classifier = FineTuningClassifier(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        model_name=model_name,
        device=device,
        dataset_path="dataset.tsv",
        trained=args.training,
        examples_path=args.examples_path,
        csv_result_file="./accuracy_example_pool_sizes.csv",
        test_mode=args.test_mode,
        perform_new_training=PERFORM_NEW_TRAINING
    )

    classifier.classify_and_evaluate()
