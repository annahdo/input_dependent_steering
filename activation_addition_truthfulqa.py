import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import gc
from datasets import load_dataset
import torch.nn.functional as F
import argparse
from datetime import datetime
import sys
import os

# import my modules
user = os.environ.get("USER")
my_path = f"/data/{user}/ICE/ice_baseline/"
sys.path.append(my_path+"modules")
from wrapping import WrappedModel
from aa_utils import *


def main(args):  

    batch_size = args.batch_size
    precision = str_to_dtype(args.precision) # You may want to convert this to an actual dtype if needed
    user_tag = args.user_tag
    assistant_tag = args.assistant_tag
    positive_addon = args.positive_addon
    negative_addon = args.negative_addon
    layer_ids = args.layer_ids
    model_path = args.model_path
    model_name = model_path.split("/")[-1]
    block_name = "decoder_block"
    results_file = args.results_file
    results_dir = args.results_dir


    # make dir if not exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # get timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    results_file = args.results_file if args.results_file is not None else f"truthful_qa_{timestamp}.txt"
    results_file = os.path.join(results_dir, results_file)

    with open(results_file, "w") as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"precision: {precision}\n")
        f.write(f"user_tag: {user_tag}\n")
        f.write(f"assistant_tag: {assistant_tag}\n")
        f.write(f"positive_addon: {positive_addon}\n")
        f.write(f"negative_addon: {negative_addon}\n")
        f.write(f"layer_ids: {layer_ids}\n")
        f.write(f"block_name: {block_name}\n\n")


    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", torch_dtype=precision)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    # tokenizer.bos_token_id = 1
    model = AutoModelForCausalLM.from_pretrained(model_path,  device_map="auto", torch_dtype=precision)


    # questions contain the question repeated as many times as there are different answers
    # answers contain the different answers
    # questions and answers have the same length
    # labels contain lists of labels, where the correct answer is marked with 1 and all other answers with 0
    # labels is shorter than questions and answers namely as long as there are different questions.
    questions, answers, labels = load_tqa_sentences(user_tag="", assistant_tag="", preset="")

    correct = []
    for l in labels:
        correct.append(1/len(l))
    random_acc = np.mean(correct)
    print(f"random_acc: {random_acc}")

    with open(results_file, "a") as f:
        f.write("-"*20+"\n")
        f.write(f"random_acc: {random_acc}\n")


    questions, answers, labels = load_tqa_sentences(user_tag=user_tag, assistant_tag=assistant_tag, preset="")
    model_baseline_acc = get_tqa_accuracy(model, questions, answers, labels, tokenizer, batch_size=batch_size)
    print(f"model_baseline_acc: {model_baseline_acc}")

    with open(results_file, "a") as f:
        f.write("-"*20+"\n")
        f.write(f"model_baseline_acc: {model_baseline_acc}\n")

    questions, answers, labels = load_tqa_sentences(user_tag=user_tag, assistant_tag=assistant_tag, preset=positive_addon)
    model_baseline_preset_acc = get_tqa_accuracy(model, questions, answers, labels, tokenizer, batch_size=batch_size)
    print(f"model_baseline_preset_acc: {model_baseline_preset_acc}")

    with open(results_file, "a") as f:
        f.write("-"*20+"\n")
        f.write(f"preset: {positive_addon}\n")
        f.write(f"model_baseline_preset_acc: {model_baseline_preset_acc}\n")

    # wrapping the model
    # create wrapped model
    wrapped_model = WrappedModel(model, tokenizer)
    # make sure nothing is wrapped from previous runs
    wrapped_model.unwrap()
    # wrap model at desired layers and blocks
    wrapped_model.wrap_block(layer_ids, block_name=block_name)

    # naive activation addition
    wrapped_model.reset()
    wrapped_model.run_prompt(positive_addon)
    pos_act = wrapped_model.get_activations(layer_ids, block_name=block_name)
    wrapped_model.reset()
    coeff = 1
    wrapped_model.run_prompt(negative_addon)
    neg_act = wrapped_model.get_activations(layer_ids, block_name=block_name)
    truth_directions = {}
    for layer_id in layer_ids:
        # take difference at last token id
        truth_directions[layer_id] = coeff*(pos_act[layer_id][0, -1] - neg_act[layer_id][0, -1])

    # set activations to add
    wrapped_model.reset()
    wrapped_model.set_to_add(layer_ids, truth_directions, block_name=block_name, normalize=True)

    # calculate accuracy
    questions, answers, labels = load_tqa_sentences(user_tag=user_tag, assistant_tag=assistant_tag, preset="")
    model_naive_aa_acc = get_tqa_accuracy(wrapped_model, questions, answers, labels, tokenizer, batch_size=batch_size)
    print(f"naive activation addition: {positive_addon} - {negative_addon}")
    print(f"model_naive_aa_acc: {model_naive_aa_acc}")
    with open(results_file, "a") as f:
        f.write("-"*20+"\n")
        f.write(f"naive activation addition: {positive_addon} - {negative_addon}\n")
        f.write(f"model_naive_aa_acc: {model_naive_aa_acc}\n")


    questions, answers, labels = load_tqa_sentences(user_tag=user_tag, assistant_tag=assistant_tag, preset=" ")
    coeff = 1.0
    # get the log probabilities of each question answer pair
    output_logprobs = []
    for q_batch, a_batch in tqdm(zip(batchify(questions, batch_size), batchify(answers, batch_size)), total=len(questions)//batch_size):
        gc.collect()
        inputs, masks, orig_split = prepare_decoder_only_inputs(q_batch, a_batch, tokenizer, model.device)


        q_batch_pos = [q + positive_addon for q in q_batch]
        q_batch_neg = [q + negative_addon for q in q_batch]

        inputs_pos_s, masks_pos_s, split_pos = prepare_decoder_only_inputs(q_batch_pos, a_batch, tokenizer, model.device)
        inputs_neg_s, masks_neg_s, split_neg = prepare_decoder_only_inputs(q_batch_neg, a_batch, tokenizer, model.device)
        wrapped_model.reset()

        # get activations
        directions = {}
        with torch.no_grad():
            wrapped_model.reset()
            _ = wrapped_model(**inputs_pos_s)
            pos_outputs = wrapped_model.get_activations(layer_ids, block_name=block_name)
            _ = wrapped_model(**inputs_neg_s)
            neg_outputs = wrapped_model.get_activations(layer_ids, block_name=block_name)
            for layer_id in layer_ids:
                directions[layer_id] = coeff*(pos_outputs[layer_id][:, split_pos:] - neg_outputs[layer_id][:, split_neg:])
                len_tokens = directions[layer_id].shape[1]
                directions[layer_id] = directions[layer_id]

        # set question tokens to zero
        # masks = masks[:,split_pos:].unsqueeze(-1)

        wrapped_model.set_to_add(layer_ids, directions, 
                                    masks=masks[:, orig_split:, None], 
                                    token_pos="end",
                                    normalize=True)

        with torch.no_grad():
            logits = wrapped_model(**inputs).logits
            logprobs = get_logprobs(logits, inputs['input_ids'], masks).sum(-1).detach().cpu().numpy()
        output_logprobs.extend(logprobs)

        assert np.isnan(output_logprobs).sum() == 0, "NaN in output logprobs"

    model_sample_wise_aa_acc = calc_acc(labels, output_logprobs)
    print(f"model_sample_wise_aa_acc: {model_sample_wise_aa_acc}")

    with open(results_file, "a") as f:
        f.write("-"*20+"\n")
        f.write(f"sample wise activation addition: {positive_addon} - {negative_addon}\n")
        f.write(f"model_sample_wise_aa_acc: {model_sample_wise_aa_acc}\n")


def load_tqa_sentences(user_tag, assistant_tag, preset=""):
    dataset = load_dataset('truthful_qa', 'multiple_choice')['validation']
    questions, answers = [],[]
    labels = []
    for d in dataset:
        q = d['question']
        for i in range(len(d['mc1_targets']['labels'])):
            a = d['mc1_targets']['choices'][i]
            questions = [f'{user_tag}' + q + ' ' + preset] + questions
            answers = [f'{assistant_tag}' + a] + answers
            # questions.append(f'{user_tag}' + q + preset)
            # answers.append(f'{assistant_tag}' + a)
        # labels.append(d['mc1_targets']['labels'])
        ls = d['mc1_targets']['labels']
        ls.reverse()
        labels.insert(0, ls)
    return questions, answers, labels

def get_logprobs(logits, input_ids, masks, **kwargs):
    logprobs = F.log_softmax(logits, dim=-1)[:, :-1]
    # find the logprob of the input ids that actually come next in the sentence
    logprobs = torch.gather(logprobs, -1, input_ids[:, 1:, None])
    logprobs = logprobs * masks[:, 1:, None] 
    return logprobs.squeeze(-1)
    
def prepare_decoder_only_inputs(prompts, targets, tokenizer, device):
    tokenizer.padding_side = "left"
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    tokenizer.padding_side = "right"
    target_inputs = tokenizer(targets, return_tensors="pt", padding=True, truncation=False)
    
    # concatenate prompt and target tokens and send to device
    inputs = {k: torch.cat([prompt_inputs[k], target_inputs[k]], dim=1).to(device) for k in prompt_inputs}

    # mask is zero for padding tokens
    mask = inputs["attention_mask"].clone()
    # set mask to 0 for question tokens
    mask[:, :prompt_inputs["input_ids"].shape[1]] = 0
    mask.to(device)
    # remove token_type_ids
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    return inputs, mask, prompt_inputs["input_ids"].shape[1]

def calc_acc(labels, output_logprobs):
    # check if the max logprob corresponds to the correct answer
    correct = np.zeros(len(labels))
    # indices to index
    indices = np.cumsum([len(l) for l in labels])
    indices = np.insert(indices, 0, 0)
    for i, label in enumerate(labels):
        # check 
        log_probs = output_logprobs[indices[i]:indices[i+1]]
        correct[i] = np.argmax(log_probs) == label.index(1)
    return correct.mean()

def get_tqa_accuracy(model, questions, answers, labels, tokenizer, batch_size=128):
    gc.collect()
    # get the log probabilities of each question answer pair
    output_logprobs = []
    for q_batch, a_batch in tqdm(zip(batchify(questions, batch_size), batchify(answers, batch_size)), total=len(questions)//batch_size):
        inputs, masks, _ = prepare_decoder_only_inputs(q_batch, a_batch, tokenizer, model.device)

        with torch.no_grad():
            if isinstance(model, WrappedModel):
                # set the masks so that we do not add to tokens of input sentences and padding tokens
                model.set_masks(masks.unsqueeze(-1))

            # calculate the probabilities for all tokens (all question answer pairs)
            logits = model(**inputs).logits
            # sum the probabilities for each question answer pair so that each pair has one probability
            # mask is zero for question and padding tokens
            logprobs = get_logprobs(logits, inputs['input_ids'], masks).sum(-1).detach().cpu().numpy()
        output_logprobs.extend(logprobs)

    return calc_acc(labels, output_logprobs)


def str_to_dtype(dtype_str):
    if dtype_str == 'torch.float32':
        return torch.float32
    elif dtype_str == 'torch.float16':
        return torch.float16
    elif dtype_str == 'torch.float64':
        return torch.float64
    elif dtype_str == 'torch.int64':
        return torch.int64
    # Add more dtype conversions as needed
    else:
        raise ValueError(f"Unknown dtype string: {dtype_str}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--precision", type=str, default="torch.float32", help="Precision setting for PyTorch")
    parser.add_argument("--user_tag", type=str, default="[INST] ", help="Tag for the user's instructions")
    parser.add_argument("--assistant_tag", type=str, default="[/INST] ", help="Closing tag for the user's instructions")
    parser.add_argument("--positive_addon", type=str, default="Give a truthful response. ", help="Addon text for positive responses")
    parser.add_argument("--negative_addon", type=str, default="Give an untruthful response. ", help="Addon text for negative responses")
    parser.add_argument("--layer_ids", type=lambda s: [int(item) for item in s.split(',')], default=np.arange(17, 23), help="Layer IDs for some specific operation. Pass as comma separated integers without spaces e.g. 17,18,19,20,21,22")
    parser.add_argument("--block_name", type=str, default="decoder_block", help="Name of the block in the neural network model")
    parser.add_argument("--results_dir", type=str, default="results/activation_addition/truthful_qa", help="Name of the results directory")
    parser.add_argument("--results_file", type=str, default=None, help="Name of the results file")
    parser.add_argument("--model_path", type=str, default="/data/private_models/cais_models/llama-2/llama/llama-2-7b-chat-hf", help="path to the model")
    args = parser.parse_args()
    main(args)