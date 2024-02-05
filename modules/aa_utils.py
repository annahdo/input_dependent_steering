import torch
import os
from datetime import datetime
import numpy as np
import random
from tqdm import tqdm

# helper functions
def config_to_textfile(textfile, prompt1, prompt2, coeff, layer_id, block_name, token_pos, max_new_tokens, coeff_dict=None, use_norms=False):
    formatted_string = f"Parameters:\n\
    prompt1: '{prompt1}'\n\
    prompt2: '{prompt2}'\n\
    coeff: {coeff}\n\
    layer_id: {layer_id}\n\
    block_name: '{block_name}'\n\
    token_pos: {token_pos}\n\
    max_new_tokens: {max_new_tokens}\n\
    use_norms: {use_norms}\n\
    coeff_dict: {coeff_dict}\n\n"
    
    with open(textfile, 'a', encoding='utf-8') as f:
        f.write(formatted_string)
        

   
def get_difference(wrapped_model, prompt1, prompt2, layer_ids, block_names):
    differences = {}
    for block_name in block_names:
        differences[block_name] = {}
    
        for layer_id in layer_ids:
            wrapped_model.wrap_block(layer_id, block_name=block_name)
            
            # get internal activations
            wrapped_model.reset()
            wrapped_model.run_prompt(prompt1)
            activations1 = wrapped_model.get_activations(layer_id, block_name=block_name)
            wrapped_model.reset()
            wrapped_model.run_prompt(prompt2)
            activations2 = wrapped_model.get_activations(layer_id, block_name=block_name)

            diff = activations1[0,-1,:]-activations2[0,-1,:]
            differences[block_name][layer_id] = diff


    wrapped_model.reset()
    return differences

def activation_editing_single(wrapped_model, sentences, directions=None, coeff=1, coeff_dict=None, token_pos=None, max_new_tokens=30, batch_size=32, use_cache=True):

    generations = []
    num_batches = len(sentences)//batch_size
    if directions:
        for block_name in directions.keys():
            for layer_id, activation in directions[block_name].items():
                coeff_entry = 1
                if coeff_dict:
                    coeff_entry = coeff_dict[block_name][layer_id]
                wrapped_model.set_to_add(layer_id, coeff*coeff_entry*activation.to(wrapped_model.device), token_pos=token_pos, block_name=block_name)

    for sentence_batch in tqdm(batchify(sentences, batch_size), total=num_batches):
        generated = wrapped_model.generate(sentence_batch, max_new_tokens=max_new_tokens, use_cache=use_cache)
        generations.extend(generated)

    wrapped_model.reset()
    return generations


def set_activations_to_add(wrapped_model, directions, coeff=1, coeff_dict=None, token_pos=None):
    for block_name in directions.keys():
        for layer_id, activation in directions[block_name].items():
            if len(activation.shape) > 1:
                # activations and coeff_dict are shaped as heads need to flatten
                if coeff_dict:
                    activation = activation * coeff_dict[block_name][layer_id].unsqueeze(1)
                activation = activation.flatten()

            else:
                if coeff_dict:
                    activation = activation*coeff_dict[block_name][layer_id]
            wrapped_model.set_to_add(layer_id, coeff*activation.to(wrapped_model.device), token_pos=token_pos, block_name=block_name)


def activation_editing(wrapped_model, sentences, directions, coeff=1, coeff_dict=None,
                                token_pos=None, max_new_tokens=30, textfile=None, printout=True, calc_neutral=True, batch_size=32, use_cache=True):

    def append_to_file(text):
        if textfile:
            with open(textfile, 'a', encoding='utf-8') as f:
                f.write(text)

    num_batches = len(sentences)//batch_size
            
    generations = {"neutral": [], "positive": [], "negative": []}
    wrapped_model.reset()
    if calc_neutral:
        print("No activation addition\n")
        append_to_file("No activation addition\n")
        for sentence_batch in tqdm(batchify(sentences, batch_size), total=num_batches, disable=printout):
            generated = wrapped_model.generate(sentence_batch, max_new_tokens=max_new_tokens, use_cache=use_cache)
            generations["neutral"].extend(generated)
            for g in generated:
                if printout:
                    print(f"{g}\n")
                append_to_file(f"{g}\n\n")
          
    set_activations_to_add(wrapped_model, directions, coeff=coeff, coeff_dict=coeff_dict, token_pos=token_pos)
    
    print("-" * 30 + "\nPositive activation addition\n")
    append_to_file("-" * 30 + "\nPositive activation addition\n")
    for sentence_batch in tqdm(batchify(sentences, batch_size), total=num_batches, disable=printout):
        generated = wrapped_model.generate(sentence_batch, max_new_tokens=max_new_tokens, use_cache=use_cache)
        generations["positive"].extend(generated)
        for g in generated:
            if printout:
                print(f"{g}\n")
            append_to_file(f"{g}\n\n")
    
    
    wrapped_model.reset()
    set_activations_to_add(wrapped_model, directions, coeff=-coeff, coeff_dict=coeff_dict, token_pos=token_pos)

    print("-" * 30 + "\nNegative activation addition\n")
    append_to_file("-" * 30 + "\nNegative activation addition\n")
    for sentence_batch in tqdm(batchify(sentences, batch_size), total=num_batches, disable=printout):
        generated = wrapped_model.generate(sentence_batch, max_new_tokens=max_new_tokens, use_cache=use_cache)
        generations["negative"].extend(generated)
        for g in generated:
            if printout:
                print(f"{g}\n")
            append_to_file(f"{g}\n\n")
    
    wrapped_model.reset()

    return generations 
        
def make_new_file(path="", filename=""):# to save results
     # Create directories recursively
    os.makedirs(path, exist_ok=True)
    # Get current date and time
    now = datetime.now()
    # Format as string in the desired format (here as YearMonthDay_HourMinuteSecond)
    now = now.strftime("%Y%m%d_%H%M%S.txt")
    full_path = os.path.join(path, filename + "_" + now)
    with open(full_path, 'w', encoding='utf-8') as f:
            pass
    return full_path

def judge_generations(wrapped_model, generations, instruction, batch_size=32, use_cache=True):
    wrapped_model.reset()    
    neutral_sentences = [g+instruction for g in generations["neutral"]]
    positive_sentences = [g+instruction for g in generations["positive"]]
    negative_sentences = [g+instruction for g in generations["negative"]]
    
    neutral = []
    positive = []
    negative = []
    num_s = len(positive_sentences)
    print("-" * 30 + "\nMaking judgements\n")
    for i in tqdm(range(0, num_s, batch_size)):
        # Extract batch
        positive_batch = positive_sentences[i:i + batch_size]
        negative_batch = negative_sentences[i:i + batch_size]

        if len(neutral_sentences) > 0:
            neutral_batch = neutral_sentences[i:i + batch_size]
            neutral_judgements = wrapped_model.generate(neutral_batch, max_new_tokens=1, use_cache=use_cache)
            neutral.extend(neutral_judgements)

        positive_judgements = wrapped_model.generate(positive_batch, max_new_tokens=1, use_cache=use_cache)
        positive.extend(positive_judgements)

        negative_judgements = wrapped_model.generate(negative_batch, max_new_tokens=1, use_cache=use_cache)
        negative.extend(negative_judgements)
        
    return {"neutral": neutral, "positive": positive, "negative":negative}
        
def save_judgements(judgements, textfile=None):

    print("-" * 30 + "\nTurning judgements into numbers\n")

    # convert to numbers
    neutral = np.array([int(s[-1]) if s[-1].isdigit() else np.nan for s in judgements["neutral"]])
    positive = np.array([int(s[-1]) if s[-1].isdigit() else np.nan for s in judgements["positive"]])
    negative = np.array([int(s[-1]) if s[-1].isdigit() else np.nan for s in judgements["negative"]])
    
    neutral_mean = np.nanmean(neutral) if judgements["neutral"] else np.nan
    positive_mean = np.nanmean(positive) if judgements["positive"] else np.nan
    negative_mean = np.nanmean(negative) if judgements["negative"] else np.nan
    
    print(f"neutral:  {neutral_mean:.2f}")
    print(f"positive: {positive_mean:.2f}")
    print(f"negative: {negative_mean:.2f}")
    print("-"*30)
    if textfile:
        with open(textfile, "a", encoding='utf-8') as f:
            f.write(f"neutral:  {neutral_mean:.2f}\n")
            f.write(f"positive: {positive_mean:.2f}\n")
            f.write(f"negative: {negative_mean:.2f}\n")
            f.write("-"*30 + "\n")
        
    return {"neutral": neutral_mean, "positive": positive_mean, "negative":negative_mean}


def batchify(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def compare(wrapped_model, generations, batch_size=128, textfile=None):
    l = len(generations["positive"])
    half_l = int(l/2)

    target = np.ones(l)
    target[half_l:] = 2
    g1 = generations["positive"][:half_l] + generations["negative"][half_l:]
    g2 = generations["negative"][:half_l] + generations["positive"][half_l:]
    
    prompts = [f"Judge the happiness in the following two scenarios:\n\
1: '{p1}'\n\
2: '{p2}'\n\
The number of the scenario that sounds nicer is: " for (p1,p2) in zip(g1, g2)]
    
    num_batches = len(prompts)//batch_size
    evals = []
    for sentence_batch in batchify(prompts, batch_size):
        generated = wrapped_model.generate(sentence_batch, max_new_tokens=1, use_cache=True)
        evals.extend(generated)
    
    numbers = np.array([int(s[-1]) if s[-1].isdigit() else np.nan for s in evals])
    
    acc = (target==numbers).sum()/l
    print(f"acc: {acc:.4f}")
    if textfile:
        with open(textfile, 'a', encoding='utf-8') as f:
            f.write(f"acc when comparing pos and neg sentence pairs: {acc:.4f}")