import pandas as pd
import numpy as np
import random
import argparse
# import bitsandbytes as bnb
import accelerate as aclrt
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Gemma3nForCausalLM, EarlyStoppingCallback
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
import shutil
import gc
import json

RANDOM_SEED = 3407

TESTING_DATASET_PATH = "dataset_medium.tsv"  # dataset.tsv is the default
TRAINING_DATASET_PATH = "examples.tsv"  # training_dataset.tsv is the default
VALIDATION_DATASET_PATH = "eval_dataset.tsv"  # eval_dataset.tsv is the default

# TESTING_DATASET_PATH = "./dataset/alternative_dataset/news_test_dataset.tsv" #alternative dataset for dishomogeneous dataset
# TRAINING_DATASET_PATH = "./dataset/alternative_dataset/news_training_dataset.tsv"  # training_dataset.tsv is the default
# VALIDATION_DATASET_PATH = "./dataset/alternative_dataset/news_eval_dataset.tsv"  # eval_dataset.tsv is the default


def create_model_and_tokenizer(model_name, device, trained, perform_new_training, examples_path=None):
    """
    Create and initialize model and tokenizer based on model name and device.
    
    Args:
        model_name: Name of the model ("gemma2", "gemma3_1b", "gemma3n_e2b_it")
        device: Device to load the model on
        trained (bool): Whether to load a fine-tuned model.
        perform_new_training (bool): Whether to use already existing trained model or overwrite it.
        examples_path (str, optional): Path to the training examples file. Required if trained=True.
    
    Returns:
        tuple: (model, tokenizer, model_id, model_name, training_dataset_length, num_epochs)
    """
    match model_name:
        case "gemma2":
            model_id = "google/gemma-2-2b-it"
            model_name_normalized = "gemma2"
        case "gemma3_1b":
            model_id = "google/gemma-3-1b-it"
            model_name_normalized = "gemma3_1b"
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

    training_dataset_length = 0
    num_epochs = 0

    if trained and not perform_new_training:
        model_base_dir = "trainedModels/GEMMA/"
        try:
            # Determine the target dataset length from the provided examples file
            train_df = pd.read_csv(examples_path, sep="\t")
            target_dataset_length = len(train_df)
            
            model_base_path = os.path.join(model_base_dir, f"{model_name_normalized}_{target_dataset_length}")

            if os.path.exists(model_base_path):
                # Find the latest epoch subfolder within the selected model folder
                epoch_folders = [d for d in os.listdir(model_base_path) if d.startswith('epoch_') and os.path.isdir(os.path.join(model_base_path, d))]
                if epoch_folders:
                    latest_epoch_folder = sorted(epoch_folders, key=lambda x: int(x.split('_')[1]), reverse=True)[0]
                    model_save_path = os.path.join(model_base_path, latest_epoch_folder)
                    num_epochs = int(latest_epoch_folder.split('_')[1])
                    training_dataset_length = target_dataset_length
                    
                    print(f"Loading trained model with training dataset length {training_dataset_length} and epoch number {num_epochs}")
                    print(f"Model path: {model_save_path}")
                    
                    model = peft.PeftModel.from_pretrained(model, model_save_path)
                    print("Fine-tuned model loaded.")
                else:
                    print(f"No epoch subfolders found in {model_base_path}. Will train a new model.")
            else:
                print(f"No trained model found for dataset length {target_dataset_length}. Will train a new model.")
        except FileNotFoundError:
            print(f"Warning: examples_path '{examples_path}' not found. Cannot determine which trained model to load.")
        except Exception as e:
            print(f"An error occurred while trying to load the fine-tuned model: {e}")

    return model, tokenizer, model_id, model_name_normalized, training_dataset_length, num_epochs

class Paraphraser:
    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
        self.device = device
        
    def paraphrase(self, question, num_beams=9, num_beam_groups=9, num_return_sequences=9, repetition_penalty=10.0, diversity_penalty=3.0, no_repeat_ngram_size=2, temperature=0.7, max_length=128, do_sample=True):
        """
        a partire da un sentence genera una lista di sentence semanticamente simili
        sentence = "I am 21 years old."
        :return: list of string
            ['I am 21 years of age.',
             'At 21 years old, I am currently living.',
             "It's my 21st birthday."]
        """

        input_ids = self.tokenizer(
            f'paraphrase: {question}',
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(self.device)

        outputs = self.model.generate(
            input_ids, repetition_penalty=repetition_penalty,  #The following generation flags are not valid and may be ignored: ['temperature']
            num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams, num_beam_groups=num_beam_groups,
            max_length=max_length, diversity_penalty=diversity_penalty
        )

        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res

    def growPhrases(self, phrases_df, numPhrase):
        """
        Extends a dataset by generating paraphrases for each phrase.
        :param phrases_df: DataFrame with 'phrase' and 'category' columns.
        :param numPhrase: The number of paraphrases to generate for each original phrase.
        """
        new_rows = []
        total_phrases = len(phrases_df)

        for _, row in phrases_df.iterrows():
            original_phrase = row['phrase']
            original_category = row['category']
            
            # Add the original phrase to the list of rows
            new_rows.append({'phrase': original_phrase, 'category': original_category})
            
            # Generate numPhrase new sentences for the original phrase
            new_sentences = self.paraphrase(
                original_phrase,
                num_beams=13,
                num_beam_groups=13,
                num_return_sequences=numPhrase,
                repetition_penalty=7.0,
                diversity_penalty=1.0,
                no_repeat_ngram_size=2,
                max_length=200,
            )

            for sentence in new_sentences:
                new_rows.append({'phrase': sentence, 'category': original_category})

        # create a new DataFrame from the list of new rows and shuffle it
        extended_df = pd.DataFrame(new_rows)
        extended_df = extended_df.sample(frac=1).reset_index(drop=True)

        # output_dir = "./dataset/training/"
        output_dir = "./dataset/alternative_dataset/extended_training_datasets/"
        os.makedirs(output_dir, exist_ok=True)
        
        # save the extended dataset to a new file
        output_filename = f"training_dataset_{len(extended_df)}.tsv"
        output_path = os.path.join(output_dir, output_filename)
        
        extended_df.to_csv(output_path, sep='\t', index=False)
        print(f"\nSuccessfully created extended dataset with {len(extended_df)} phrases.")
        print(f"Saved to: {output_path}")

class FineTuningClassifier:
    def __init__(self, model, tokenizer, model_id, model_name, device, dataset_path, eval_dataset_path, trained, examples_path, csv_result_file, test_mode="zero", perform_new_training=False, examples_per_category=None, use_early_stopping=False, training_dataset_length=0, num_epochs=0, fixed_num_epochs=None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_name = model_name
        self.device = device
        self.dataset_path = dataset_path
        self.csv_result_file = csv_result_file
        self.test_mode = test_mode
        self.trained = trained
        self.examples_per_category = examples_per_category
        self.use_early_stopping = use_early_stopping
        self.num_epochs = num_epochs  # Initialize with value from create_model_and_tokenizer
        self.max_steps = None   # Initialize training param trackers
        self.training_dataset_length = training_dataset_length # Initialize with value from create_model_and_tokenizer
        self.fixed_num_epochs = fixed_num_epochs
        self.eval_loss = None
        self.train_loss = None
        self.eval_accuracy = None
        
        torch.manual_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"Running in: {test_mode} mode [{'trained]' if self.trained else 'base (untrained)]'}")
        print(f"Running with {self.model_id} model")
        
        # load dataset, examples and set categories
        self.test_df = pd.read_csv(dataset_path, sep="\t")
        
        if self.dataset_path == "./dataset/alternative_dataset/news_test_dataset.tsv": # categories for not homogeneous news dataset
            self.categories = [
                "crime",
                "advertisement",
                "weather",
                "gossip",
                "politics",
                "finance",
                "culture"
            ]
        else:            # Default categories for automotive failure classification  
            self.categories = [
                "Fuel System Problems",
                "Ignition System Malfunctions",
                "Cooling System Anomalies",
                "Brake System Defects",
                "Transmission Problems",
                "Electrical/Electronic Failures"
            ]
        
        if self.trained:
            full_train_df = pd.read_csv(examples_path, sep="\t")
            full_eval_dataset_df = pd.read_csv(eval_dataset_path, sep="\t")
            
            self.eval_df = full_eval_dataset_df
            
            if self.examples_per_category is not None:
                # Select a subset of examples for training
                grouped = full_train_df.groupby('category')
                self.train_df = grouped.head(self.examples_per_category).reset_index(drop=True)
                print(f"Sub-sampling training data: {self.examples_per_category} examples per category.")
            else:
                self.train_df = full_train_df

            print(f"Loaded {len(self.train_df)} examples for training from {examples_path}.")
            print(f"Using {len(self.test_df)} examples for testing from {dataset_path}.")
            
            # If training was not performed outside, set dataset length
            if self.training_dataset_length == 0:
                self.training_dataset_length = len(self.train_df)

            # Determine model save path. It will be fully defined after training with epoch number.
            model_base_path = f"trainedModels/GEMMA/{self.model_name}_{self.training_dataset_length}"
            
            # Training is needed if we force it, or if a pre-trained model was not loaded.
            # A pre-trained model is loaded if training_dataset_length > 0 and num_epochs > 0 from the create function.
            should_train = perform_new_training or not (self.training_dataset_length > 0 and self.num_epochs > 0)

            if should_train:
                if perform_new_training:
                    print("Forcing new training...")
                else:
                    print("No suitable pre-trained model found. Starting training...")

                # For the experiment, train by epoch.
                use_epochs_for_training = self.examples_per_category is not None or self.fixed_num_epochs is not None
                model_save_path, self.eval_loss, self.train_loss = self.trainModel(model_base_path, use_epochs=use_epochs_for_training, num_epochs=self.fixed_num_epochs)
                print(f"Loading fine-tuned model from {model_save_path}")
                # The base model is already a PeftModel if we are retraining, so we need to load into the base model of the peft model
                if isinstance(self.model, peft.PeftModel):
                    base_model = self.model.get_base_model()
                    self.model = peft.PeftModel.from_pretrained(base_model, model_save_path)
                else:
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
            
            if self.dataset_path == "./dataset/alternative_dataset/news_test_dataset.tsv":
                context_examples_df = pd.read_csv("./dataset/alternative_dataset/news_context_examples_dataset.tsv", sep="\t")

            # get 1 example per category
            few_shot_rows = []
            for category in self.categories:
                examples = context_examples_df[context_examples_df['category'] == category].head(self.adjusted_example_pool_size)
                few_shot_rows.append(examples)

            few_shot_df = pd.concat(few_shot_rows).reset_index(drop=True)
            self.few_shot_examples = few_shot_df.to_dict(orient='records')
        
        elif test_mode == "paraph":          
            self.paraphraser = Paraphraser(
                device
            )

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
        
        if self.dataset_path == "./dataset/alternative_dataset/news_test_dataset.tsv":
            prompt = (  #circa 88% con gemma3
                f"Classify the following news article: "
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
        
        if self.dataset_path == "./dataset/alternative_dataset/news_test_dataset.tsv": # categories for not homogeneous news dataset
            prompt = (  
                f"Classify the following news article: " #for some reason keeping the automotive failure phrase increases accuracy by about 5% IN GEMMA 3 BUT GEMMA TWO LIKES THIS PROMPT
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
        
        if self.dataset_path == "./dataset/alternative_dataset/news_test_dataset.tsv":
            prompt = ( 
                f"Classify the following news article: "
                f"{phrase}\n"
                "into one of these categories: "
                f"{', '.join(self.categories)}.\n"
                "using the following definitions:\n"
                "crime: the focus on the stories related to the commitment of crimes (e.g., bank robbery, murder) or the potential criminal activity (e.g., corporate ethics, fraud)\n"
                "advertisement: the activity or profession of producing advertisements for commercial products or services\n"
                "weather: The study, prediction and reporting of meteorological phenomena\n"
                "gossip: the part of a newspaper in which you find stories about the social and private lives of famous people\n"
                "politics: A news produced by the mainstream media, such as news TV channel, that contains information related with politics, public issues and policies and political affairs.\n"
                "finance: News about the arts and other manifestations of human intellectual achievement regarded collectively\n"
                "culture: any news that pertains to money and investments, including news on markets\n"
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
        
        if self.dataset_path == "./dataset/alternative_dataset/news_test_dataset.tsv":
            prompt = (
                f"Classify the following news article: "
                f"{phrase}\n"
                "into one of these categories: "
                f"{', '.join(self.categories)}.\n"
                "using the following definitions:\n"
                "crime: the focus on the stories related to the commitment of crimes (e.g., bank robbery, murder) or the potential criminal activity (e.g., corporate ethics, fraud)\n"
                "advertisement: the activity or profession of producing advertisements for commercial products or services\n"
                "weather: The study, prediction and reporting of meteorological phenomena\n"
                "gossip: the part of a newspaper in which you find stories about the social and private lives of famous people\n"
                "politics: A news produced by the mainstream media, such as news TV channel, that contains information related with politics, public issues and policies and political affairs.\n"
                "finance: News about the arts and other manifestations of human intellectual achievement regarded collectively\n"
                "culture: any news that pertains to money and investments, including news on markets\n"
                "With the following examples:\n"
            )

        if self.adjusted_example_pool_size > 0:
            for example in examples:
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
            case "paraph":
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
            torch.manual_seed(RANDOM_SEED)
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
            'random_state':RANDOM_SEED, #x riproducibilità esperimento
            'use_rslora':False,  # We support rank stabilized LoRA
            'loftq_config':None  # And LoftQ
        }
        """

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"], #target all linear layers for lora (full fine-tuning like performance)
            lora_dropout=0.1,  # changed from 0.05 24/07/25
            bias="none",
            # random_state=RANDOM_SEED,  # For reproducibility DOESN'T WORK
            task_type="CAUSAL_LM",
        )
        return config

    def __get_train_params_default(self, modelSaveFileName, use_epochs=False, num_epochs=None):
        # define training arguments
        training_args = {
            "output_dir": modelSaveFileName,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
            "learning_rate": 2e-4,
            "fp16": not torch.cuda.is_bf16_supported(),
            "bf16": torch.cuda.is_bf16_supported(),
            "logging_steps": 16,
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "seed": RANDOM_SEED,
            "data_seed": RANDOM_SEED,
            "report_to": "none",  # Use this for WandB etc
        }

        if self.use_early_stopping:
            training_args["load_best_model_at_end"] = True
            training_args["eval_strategy"] = "epoch"
            training_args["save_strategy"] = "epoch"
            training_args["metric_for_best_model"] = "eval_loss"
            training_args["greater_is_better"] = False
        else:
            # training_args["eval_strategy"] = "no"
            # training_args["save_strategy"] = "epoch"
            # training_args["load_best_model_at_end"] = True
            training_args["eval_strategy"] = "epoch" if num_epochs is not None else "no"
            training_args["save_strategy"] = "epoch"
            training_args["metric_for_best_model"] = "eval_loss"
            training_args["greater_is_better"] = False

        if use_epochs or self.use_early_stopping:
            training_args["num_train_epochs"] = num_epochs if num_epochs is not None else 17 #1 -> 4 -> 2 -> max_steps(60) testing order
            # training_args["max_steps"] = 30
        else:
            training_args["num_train_epochs"] = 7 #average best epoch setting
    
        
        return TrainingArguments(**training_args)

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
            
            if self.dataset_path == "./dataset/alternative_dataset/news_test_dataset.tsv":
                text = f"Classify the following news article: {phrase}\ninto one of these categories: {', '.join(self.categories)}.\nCategory: {category}"
            texts.append(text)
        return {"text": texts}


    def trainModel(self, modelSaveFileName, peft_params=None, train_params=None, use_epochs=False, num_epochs=None):
        """
        effettua il fine tuning dal modello base con i dati del dataset
        :param modelSaveFileName: path where to save the model adapters
        :param peft_params: OBJ LoraConfig per parametri per le matrici da aggiungere ai pesi
        :param train_params: OBJ SFTConfig per parametri per il fine tuning (numero di epoche, lernong rate, ...)
        :return: The final path where the model was saved, including the epoch number.
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
            # We pass a temporary path to TrainingArguments, as it requires an output_dir.
            # The final model will be saved in a path constructed after training.
            temp_output_dir = os.path.join(modelSaveFileName, "temp_training_output")
            train_params = self.__get_train_params_default(temp_output_dir, use_epochs=use_epochs, num_epochs=num_epochs)

        # Store the training parameters used for logging
        if use_epochs or self.use_early_stopping:
            self.num_epochs = train_params.num_train_epochs
            self.max_steps = train_params.max_steps

        # Model: We now add LoRA adapters so we only need to update 1 to 10% of all parameters!
        self.model.config.use_cache = False # Must be false for gradient checkpointing
        base_model = get_peft_model(
            self.model,
            peft_params
        )

        max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!

        callbacks_ = None
        eval_dataset_ = None
        if self.use_early_stopping:
            callbacks_ = [EarlyStoppingCallback(early_stopping_patience=1)] 
            eval_dataset_ = Dataset.from_pandas(self.eval_df) 
            eval_dataset_ = eval_dataset_.map(self.formatting_prompts_func, batched=True)
        elif num_epochs is not None: # Also enable evaluation for epoch experiment
            eval_dataset_ = Dataset.from_pandas(self.eval_df)
            eval_dataset_ = eval_dataset_.map(self.formatting_prompts_func, batched=True)

        fine_tuning = SFTTrainer(
            model=base_model,
            # tokenizer=self.tokenizer,
            train_dataset=dataset,
            # eval_dataset=eval_dataset_,  # TODO change back to eval_dataset (unknown problem for reduce accuracy)
            eval_dataset=eval_dataset_,
            # dataset_text_field="phrase",  SFTConfig arg
            # max_seq_length=max_seq_length, SFTConfig arg
            # dataset_num_proc=2, SFTConfig arg
            # packing=False,  # Can make training 5x faster for short sequences.  SFTConfig arg
            args=train_params,
            # peft_config=peft_params, # already passing peft wrapped model above
            callbacks=callbacks_
        )

        # Training modifica modello self.model
        fine_tuning.train()
        self.num_epochs = fine_tuning.state.epoch
        print(f"Training stopped at epoch: {self.num_epochs}")
        if fine_tuning.state.best_model_checkpoint:
            print(f"Best model was from epoch: {fine_tuning.state.best_model_checkpoint}")
        
        eval_loss = None
        train_loss = None
        
        if num_epochs is not None:
            # Extract final evaluation loss from log history
            for log in reversed(fine_tuning.state.log_history):
                if 'eval_loss' in log:
                    eval_loss = log['eval_loss']
                    print(f"Final eval loss: {eval_loss}")

                if 'train_loss' in log:
                    train_loss = log['train_loss']
                    print(f"Final training loss: {train_loss}")
                
                if eval_loss is not None and train_loss is not None:
                    break

        # Construct final save path with epoch number
        final_model_save_path = os.path.join(modelSaveFileName, f"epoch_{int(self.num_epochs)}")
        os.makedirs(final_model_save_path, exist_ok=True)

        # Save Model
        base_model.save_pretrained(final_model_save_path)  # salva i parametri del modello nella cartella modello
        self.tokenizer.save_pretrained(final_model_save_path)
        
        return final_model_save_path, eval_loss, train_loss

    def evaluate_accuracy(self, predictions, dataset):
        """
        Given a list of predictions, evaluate accuracy comparing with the dataset.
        :param predictions:
        :return: accuracy
        """
        correct = 0
        total = len(dataset)

        for i in range(total):
            expected = dataset.iloc[i]['category'].strip().lower()
            predicted = predictions[i].strip().lower()
            if predicted in expected or expected in predicted:
                correct += 1

        accuracy = correct / total * 100
        return accuracy, correct, total

    def sklearn_metrics(self, topics, expected_labels, predicted_labels):
        """
        Calculate performance metrics using sklearn (confusion matrix, precision score, recall score, accuracy score).
        :return: Dictionary containing all metrics data
        """
        expected_labels_normalized = [label.lower() for label in expected_labels]
        predicted_labels_normalized = [label.lower() for label in predicted_labels]
        topics_normalized = [topic.lower() for topic in topics]
        
        # associate predicted labels that contain expected labels as substrings
        for i, predicted_label in enumerate(predicted_labels_normalized):
            for expected_label in expected_labels_normalized:
                if expected_label in predicted_label and expected_label != predicted_label:
                    predicted_labels_normalized[i] = expected_label
                    break

        accuracy = accuracy_score(expected_labels_normalized, predicted_labels_normalized)

        print(f"Accuratezza del modello: {accuracy:.4f}")

        precision_per_label = precision_score(expected_labels_normalized, predicted_labels_normalized, labels=topics_normalized, average=None, zero_division=0)

        recall_per_label = recall_score(expected_labels_normalized, predicted_labels_normalized, labels=topics_normalized, average=None, zero_division=0)

        f1_per_label = f1_score(expected_labels_normalized, predicted_labels_normalized, labels=topics_normalized, average=None, zero_division=0)

        # Stampa i risultati per ogni label
        metrics_data = {}
        for label, precision, recall, f1 in zip(topics_normalized, precision_per_label, recall_per_label, f1_per_label):
            print(f"Risultati per '{label}':")
            print(f"  Precisione: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            # Store metrics for JSON export
            metrics_data[label] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }

        # confusion matrix
        confusion = confusion_matrix(expected_labels_normalized, predicted_labels_normalized, labels=topics_normalized)

        # Add line breaks to category names for better readability
        display_labels = []
        for label in topics_normalized:
            # Split long category names and add line breaks
            if len(label) > 15:  # Adjust threshold as needed
                words = label.split()
                if len(words) > 1:
                    mid_point = len(words) // 2
                    line1 = ' '.join(words[:mid_point])
                    line2 = ' '.join(words[mid_point:])
                    display_labels.append(f"{line1}\n{line2}")
                else:
                    display_labels.append(label)
            else:
                display_labels.append(label)

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=display_labels)

        # Create figure with higher DPI and larger size
        fig, ax = plt.subplots(figsize=(16, 14), dpi=200)

        disp.plot(ax=ax, xticks_rotation=45)

        # Improve text readability with even larger fonts
        ax.set_xlabel('Predicted Label', fontsize=24, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=24, fontweight='bold')

        # Make tick labels even larger and bold
        ax.tick_params(axis='both', which='major', labelsize=16)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # Make the numbers in the matrix even larger and bold
        for text in disp.text_.ravel():
            text.set_fontsize(20)
            text.set_fontweight('bold')

        plt.tight_layout()

        base_path="./graphs/sklearn_metrics"
        trained_df_len = len(self.train_df) if self.trained and self.train_df is not None else 0
        model_suffix = f"{self.model_name}{'_trained_' + str(trained_df_len) if self.trained and trained_df_len > 0 else ''}"

        fileName = "vehicularFailures" if self.dataset_path == "./dataset/alternative_dataset/vehicular_failures_test_dataset.tsv" else "news"
        match self.test_mode:
            case "few":
                plt.savefig(f"{base_path}/{model_suffix}_{fileName}_few-shot.png", dpi=200, bbox_inches='tight')
            case "paraph":
                plt.savefig(f"{base_path}/{model_suffix}_{fileName}_paraph.png", dpi=200, bbox_inches='tight')
            case "zero":
                plt.savefig(f"{base_path}/{model_suffix}_{fileName}_zero-shot.png", dpi=200, bbox_inches='tight')
            case "def":
                plt.savefig(f"{base_path}/{model_suffix}_{fileName}_definitions-test.png", dpi=200, bbox_inches='tight')
            case "def-few":
                plt.savefig(f"{base_path}/{model_suffix}_{fileName}_definitions-and_examples-test.png", dpi=200, bbox_inches='tight')

        return metrics_data

    def log_results_to_csv(self, accuracy, process_time, detailed_metrics=None):
        """
         Log accuracy results to a CSV file and detailed metrics to JSON file.
        :param accuracy:
        :param process_time:
        :param detailed_metrics: Dictionary containing precision, recall, f1-score per class
        :return:
        """
        model_log_name = self.model_name
        if self.trained:
            model_log_name = f"{self.model_name}_trained_{self.training_dataset_length}"

        timestamp = datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
        
        new_row = {
            "Timestamp": timestamp,
            "Test mode": self.test_mode,
            "Model": model_log_name,
            "Process_time(s)": process_time,
            "Accuracy (%)": round(accuracy, 2),
        }

        if self.examples_per_category is not None:
            new_row["Training Examples per Category"] = self.examples_per_category
        
        # Add training parameters to the log
        if self.trained and self.num_epochs is not None:
            new_row["Num Epochs"] = int(self.num_epochs)
        
        # Define base columns
        all_columns = [
            "Timestamp", "Test mode", "Model", "Process_time(s)", "Accuracy (%)",
            "Training Examples per Category", "Num Epochs"
        ]

        if self.eval_loss is not None:
            new_row["Eval Loss"] = self.eval_loss
            all_columns.append("Eval Loss")
        if self.train_loss is not None:
            new_row["Train Loss"] = self.train_loss
            all_columns.append("Train Loss")
        if self.eval_accuracy is not None:
            new_row["eval_accuracy"] = self.eval_accuracy
            all_columns.append("eval_accuracy")

        #change column names for accuracy for training epoch experiment
        if (
            self.eval_accuracy is not None and
            self.eval_loss is not None and
            self.train_loss is not None and
            self.train_accuracy is not None and
            self.num_epochs is not None
        ):
            
            # Move value from "Accuracy (%)" to "train_accuracy"
            new_row["train_accuracy"] = self.train_accuracy
            # Replace column name in all_columns
            all_columns = [
                "Timestamp", "Test mode", "Model", "Process_time(s)", "train_accuracy",
                "Training Examples per Category", "Num Epochs"
            ] + [col for col in all_columns if col not in [
                "Timestamp", "Test mode", "Model", "Process_time(s)", "Accuracy (%)",
                "Training Examples per Category", "Num Epochs"
            ]]

        if os.path.exists(self.csv_result_file):
            df = pd.read_csv(self.csv_result_file, sep=";")
        else:
            df = pd.DataFrame(columns=[col for col in all_columns if col in new_row])

        # Add new columns if they don't exist
        for col in all_columns:
            if col not in df.columns and col in new_row:
                df[col] = None

        df_new_row = pd.DataFrame([new_row])
        df = pd.concat([df, df_new_row], ignore_index=True)

        # Reorder columns to the defined standard
        df = df.reindex(columns=[col for col in all_columns if col in df.columns])

        df.to_csv(self.csv_result_file, index=False, sep=";")
        
        # Save detailed metrics to JSON if provided
        if detailed_metrics is not None:
            self.save_sklearn_metrics_to_json(timestamp, model_log_name, detailed_metrics)

    def save_sklearn_metrics_to_json(self, timestamp, model_log_name, detailed_metrics):
        """
        Save detailed sklearn metrics to a JSON file.
        :param timestamp: Test run timestamp
        :param model_log_name: Model identifier
        :param detailed_metrics: Dictionary containing precision, recall, f1-score per class
        """
        json_filename = self.csv_result_file.replace('.csv', '_sklearn_metrics.json')
        
        # Create test run record
        test_run_record = {
            "timestamp": timestamp,
            "test_mode": self.test_mode,
            "model": model_log_name,
            "metrics_per_class": detailed_metrics
        }
        
        # Add additional context if available
        if self.examples_per_category is not None:
            test_run_record["training_examples_per_category"] = self.examples_per_category
        if self.trained and self.num_epochs is not None:
            test_run_record["num_epochs"] = int(self.num_epochs)
        if self.eval_loss is not None:
            test_run_record["eval_loss"] = self.eval_loss
        if self.train_loss is not None:
            test_run_record["train_loss"] = self.train_loss
        
        # Load existing data or create new
        if os.path.exists(json_filename):
            try:
                with open(json_filename, 'r', encoding='utf-8') as f:
                    all_metrics = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                all_metrics = []
        else:
            all_metrics = []
        
        # Append new record
        all_metrics.append(test_run_record)
        
        # Save updated data
        os.makedirs(os.path.dirname(json_filename) if os.path.dirname(json_filename) else '.', exist_ok=True)
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Sklearn metrics saved to: {json_filename}")

    def evaluate_train_eval_accuracy(self, training_dataset, validation_dataset):
        """
        Evaluate the model's accuracy on both training and validation datasets. Used for the epoch experiment
        """
        predictions = []
        
        train_df = pd.read_csv(training_dataset, sep="\t")
        eval_df = pd.read_csv(validation_dataset, sep="\t")
        
        print("Evaluating training accuracy...")
        t1_start_T = time.perf_counter()  # process_time()
    
        for i, row in train_df.iterrows():
                phrase = row['phrase']
                category = self.classify_phrase(phrase)
                predictions.append(category)
                # print(f"{i + 1}. \"{phrase}\" →  {category}")
        
        t1_stop_T = time.perf_counter()  # process_time()


        self.train_accuracy, correct, total = self.evaluate_accuracy(predictions, train_df)
        print(f"Training accuracy: {self.train_accuracy} completed in {t1_stop_T - t1_start_T:.2f} seconds")
        
        predictions = []  # Reset predictions for validation evaluation
        
        print("Evaluating validation accuracy...")
        t1_start_E = time.perf_counter()  # process_time()
        for i, row in eval_df.iterrows():
                phrase = row['phrase']
                category = self.classify_phrase(phrase)
                predictions.append(category)
                # print(f"{i + 1}. \"{phrase}\" →  {category}")
                
        t1_stop_E = time.perf_counter()  # process_time()
        
        process_time = t1_stop_T - t1_start_T + t1_stop_E - t1_start_E
        
        accuracy = 0.0 #overwrite accuracy for epoch experiment, this value is not being logged in the CSV for this use case

        self.eval_accuracy, correct, total = self.evaluate_accuracy(predictions, eval_df)  #FIX OR CREATE NEW evaluate_accuracy method for this function (or do it inside)
        print(f"Evaluation accuracy: {self.eval_accuracy} completed in {t1_stop_E - t1_start_E:.2f} seconds")
        self.log_results_to_csv(accuracy, process_time)
        
    
    def classify_and_evaluate(self):
        """
        Perform classification test on the dataset and print accuracy data
        """
        predictions = []

        t1_start = time.perf_counter()  # process_time()

        if self.test_mode == "paraph":
            if self.model_name == "gemma3n_e2b_it":
                torch._dynamo.config.recompile_limit = 16

            for i, row in self.test_df.iterrows():
                phrase = row['phrase']
                
                # Create a list with the original phrase and its paraphrases
                phrases_to_test = [phrase]
                # The user requested 4 paraphrases
                extended_phrases = self.paraphraser.paraphrase(
                    question=phrase,
                    num_beams=13,
                    num_beam_groups=13,
                    num_return_sequences=4,
                    repetition_penalty=7.0,
                    diversity_penalty=1.0,
                    no_repeat_ngram_size=2,
                    max_length=200,
                )
                phrases_to_test.extend(extended_phrases)

                # Classify all phrases (original + paraphrases)
                paraph_predictions = []
                # print("\nNow testing a new phrase, predicted categories were:")
                # print(f"Original: '{phrase}' with paraphrases: {extended_phrases}")
                for p in phrases_to_test:
                    category = self.classify_phrase(p)
                    # print(f"{i+1} →  {category}")
                    paraph_predictions.append(category)
                
                # Vote for the most frequent category
                if paraph_predictions:
                    final_category = max(set(paraph_predictions), key=paraph_predictions.count)
                    # print(f"Final category for phrase {i + 1} : {final_category}")
                else:
                    final_category = "Unknown"  # Fallback
                
                predictions.append(final_category)
                # print(f"{i + 1}. \"{phrase}\" →  {final_category} (from {paraph_predictions})")

        else:
            for i, row in self.test_df.iterrows():
                phrase = row['phrase']
                category = self.classify_phrase(phrase)
                predictions.append(category)
                # print(f"{i + 1}. \"{phrase}\" →  {category}")

        t1_stop = time.perf_counter()  # process_time()
        process_time = t1_stop - t1_start

        accuracy, correct, total = self.evaluate_accuracy(predictions, self.test_df)
        print(f"\nModel Accuracy: {accuracy:.2f}% ({correct}/{total} correct) |Test Mode: {self.test_mode} |Process time: {process_time:.2f} seconds")
        
        # Calculate detailed metrics and save to JSON
        detailed_metrics = self.sklearn_metrics(self.categories, self.test_df['category'].to_list(), predictions)
        self.log_results_to_csv(accuracy, process_time, detailed_metrics)


def run_training_experiment(args, device):
    """Runs an experiment to test model accuracy with varying numbers of training examples."""
    print("--- Starting Training Examples Experiment ---")
    max_examples = 10  # As per your dataset structure (10 examples per category)
    
    for i in range(5, max_examples + 1):  #CHANGE TO SET STARTING POINT FOR EXAMPLES PER CATEGORY
        print(f"\n--- Running experiment with {i} examples per category ---")
        
        # Always load the base model for each training run
        model, tokenizer, model_id, model_name, _, _ = create_model_and_tokenizer(
            args.model, 
            device,
            trained=False, # Start with base model, training will be handled by classifier
            perform_new_training=True, # Will be trained inside classifier
            examples_path=args.examples_path
        )

        classifier = FineTuningClassifier(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            model_name=model_name,
            device=device,
            dataset_path=TESTING_DATASET_PATH,
            eval_dataset_path="./dataset/alternative_dataset/news_test_dataset.tsv",
            trained=True,
            examples_path=args.examples_path,
            csv_result_file="./training_set_size_accuracy.csv", # specific CSV for this experiment
            test_mode="zero",  # only use zero-shot for this experiment
            perform_new_training=True, # Force retraining
            examples_per_category=i,
            use_early_stopping=args.use_early_stopping
        )
        classifier.classify_and_evaluate()

    print("--- Training Examples Experiment Finished ---")

def run_epoch_experiment(args, device, epoch_number_max):
    """Runs an experiment to test model accuracy and loss with a varying number of training epochs."""
    print("--- Starting Epoch vs. Accuracy/Loss Experiment ---")
    MAX_EPOCHS = 13  # Maximum number of epochs to test
    
    for epoch_num in range(epoch_number_max,epoch_number_max+1 ): #MAX_EPOCHS + 1
        print(f"\n--- Running experiment with {epoch_num} epochs ---")
        
        # Always load the base model for each training run
        model, tokenizer, model_id, model_name, _, _ = create_model_and_tokenizer(
            args.model, 
            device,
            trained=False,
            perform_new_training=True,
            examples_path=args.examples_path
        )

        # Clean up previous model directory to ensure a fresh start
        # Determine training dataset length to construct the path
        try:
            train_df = pd.read_csv(args.examples_path, sep="\t")
            training_dataset_length = len(train_df)
            model_base_path = f"trainedModels/GEMMA/{model_name}_{training_dataset_length}"
            if os.path.exists(model_base_path):
                print(f"Removing old model directory: {model_base_path}")
                shutil.rmtree(model_base_path)
        except FileNotFoundError:
            print(f"Warning: examples_path '{args.examples_path}' not found. Cannot determine model path for cleanup.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

        classifier = FineTuningClassifier(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            model_name=model_name,
            device=device,
            dataset_path=TRAINING_DATASET_PATH,   #CALCULATE ACCURACY ON THE TRAINING DATASET
            eval_dataset_path="./dataset/alternative_dataset/news_test_dataset.tsv",
            trained=True,
            examples_path=args.examples_path,
            csv_result_file="./epoch_vs_accuracy_loss.csv", # specific CSV for this experiment
            test_mode="zero",  # only use zero-shot for this experiment
            perform_new_training=True, # Force retraining
            fixed_num_epochs=epoch_num, # Set the exact number of epochs for this run
            use_early_stopping=False # Disable early stopping for this experiment
        )
        classifier.evaluate_train_eval_accuracy(TRAINING_DATASET_PATH, VALIDATION_DATASET_PATH)  #CALCULATE ACCURACY ON THE TRAINING DATASET
        
        # Clean up memory to prevent OOM errors in the next iteration
        del model
        del tokenizer
        del classifier
        model = None,
        tokenizer = None
        classifier = None
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 
        
    print("--- Epoch vs. Accuracy/Loss Experiment Finished ---")


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
    
    parser.add_argument(
        "--epoch_number_max",
        type=int,
        nargs='?',
        const=3,  # Default value if flag is present without a number
        default=None, # Default value if flag is not present
        help="Paraphrase and extend the training dataset. Optionally specify the number of paraphrases per phrase (default: 3)."
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
        choices=["zero", "few", "def", "def-few", "paraph"],
        help="Testing mode: 'zero', 'few', 'def', 'def-few', or 'paraph'. Default is 'zero'."
    )

    parser.add_argument(
        "--use_early_stopping",
        action='store_true',
        help="Use early stopping during training to prevent overfitting."
    )

    parser.add_argument(
        "--training_examples_experiment",
        action='store_true',
        help="Run an experiment on the number of training examples used to see how it affects accuracy."
    )

    parser.add_argument(
        "--epoch_experiment",
        action='store_true',
        help="Run an experiment on the number of training epochs to see how it affects accuracy and loss."
    )

    parser.add_argument(
        "--paraphrase_and_extend",
        type=int,
        nargs='?',
        const=3,  # Default value if flag is present without a number
        default=None, # Default value if flag is not present
        help="Paraphrase and extend the training dataset. Optionally specify the number of paraphrases per phrase (default: 3)."
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

    if args.paraphrase_and_extend is not None:
        print("--- Starting Paraphrase and Extend utility ---")
        num_paraphrases = args.paraphrase_and_extend
        print(f"Generating {num_paraphrases} paraphrases for each phrase in '{args.examples_path}'.")
        
        # Load the base dataset to be extended
        try:
            source_df = pd.read_csv(args.examples_path, sep='\t')
        except FileNotFoundError:
            print(f"Error: The source file '{args.examples_path}' was not found.")
            exit()

        # Initialize paraphraser
        paraphraser = Paraphraser(device=device)
        
        # Run the extension process
        paraphraser.growPhrases(source_df, num_paraphrases)
        
        print("--- Paraphrase and Extend utility finished. ---")

    elif args.training_examples_experiment:
        run_training_experiment(args, device)
    elif args.epoch_experiment:
        run_epoch_experiment(args, device, args.epoch_number_max)
    else:
        # Create model and tokenizer outside the class
        model, tokenizer, model_id, model_name, training_dataset_length, num_epochs = create_model_and_tokenizer(
            args.model, 
            device, 
            args.training, 
            PERFORM_NEW_TRAINING,
            examples_path=args.examples_path
        )

        classifier = FineTuningClassifier(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            model_name=model_name,
            device=device,
            dataset_path=TESTING_DATASET_PATH,
            eval_dataset_path="./dataset/alternative_dataset/news_test_dataset.tsv",
            trained=args.training,
            examples_path=args.examples_path,
            csv_result_file="./accuracy_example_pool_sizes.csv",
            test_mode=args.test_mode,
            perform_new_training=PERFORM_NEW_TRAINING,
            use_early_stopping=args.use_early_stopping,
            training_dataset_length=training_dataset_length,
            num_epochs=num_epochs
        )

        classifier.classify_and_evaluate()