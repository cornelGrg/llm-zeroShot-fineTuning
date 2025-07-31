import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Paraphraser:
    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
        self.device = device
        
    def paraphrase(self, question, num_beams=9, num_beam_groups=9, num_return_sequences=9, repetition_penalty=10.0, diversity_penalty=3.0, no_repeat_ngram_size=2, temperature=0.7, max_length=128, do_sample=False):
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
            max_length=max_length, diversity_penalty=diversity_penalty, do_sample=do_sample
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
                num_beams=numPhrase,
                num_beam_groups=numPhrase,
                num_return_sequences=numPhrase
            )

            for sentence in new_sentences:
                new_rows.append({'phrase': sentence, 'category': original_category})

        # create a new DataFrame from the list of new rows and shuffle it
        extended_df = pd.DataFrame(new_rows)
        extended_df = extended_df.sample(frac=1).reset_index(drop=True)

        output_dir = "./dataset/training/"
        os.makedirs(output_dir, exist_ok=True)
        
        # save the extended dataset to a new file
        output_filename = f"training_dataset_{len(extended_df)}.tsv"
        output_path = os.path.join(output_dir, output_filename)
        
        extended_df.to_csv(output_path, sep='\t', index=False)
        print(f"\nSuccessfully created extended dataset with {len(extended_df)} phrases.")
        print(f"Saved to: {output_path}")

if __name__ == "__main__":
    paraph = Paraphraser(device='cuda' if torch.cuda.is_available() else 'cpu')
    test_phrase = "The engine temperature gauge never reaches the normal operating range. The thermostat is stuck open, causing the engine to run too cool."
    # test_phrase = "The ABS and traction control lights are on. A diagnostic scan shows a fault in the right front wheel speed sensor."
    # paraphrases = paraph.paraphrase(test_phrase, num_beams=4, num_beam_groups=4, num_return_sequences=4) #test 1 parametri default
    paraphrases = paraph.paraphrase(
        question=test_phrase,
        num_beams=13,
        num_beam_groups=13,
        num_return_sequences=5,
        repetition_penalty=7.0,
        diversity_penalty=1.0,
        no_repeat_ngram_size=2,
        max_length=200,
    ) #test con parametri modificati meno aggressivi


    print ("Original Phrase: '", test_phrase, "'")
    print("Paraphrases:")
    for p in paraphrases:
        print("-", p)