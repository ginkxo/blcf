from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import numpy as np
import string 

IN_PATH = './inputs1.txt'
OUT_PATH = './outputs1.txt'

input_sents = []
output_sents = []

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")

def logit2prob(logit):
    odds = np.exp(logit)
    prob = odds / (1 + odds)
    return prob 

# load path 
with open(IN_PATH, 'r') as f:
    input_sents = f.readlines()

for origsent in input_sents:

    # sent = origsent.replace("i ", "you ") # TODO: primitive test replacement of first person subject pronom with second person
    sent = origsent 
    inputs = tokenizer(sent, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    # retrieve index of <mask>
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    minimum_token_id = logits[0, mask_token_index].argmin(axis=-1)
    min_value = logits[0, mask_token_index, minimum_token_id]

    mutable_logits = logits

    curr_tok_id = predicted_token_id 
    top_5_words = []

    for i in range(5):
        gen_word = tokenizer.decode(curr_tok_id)
        logitval = mutable_logits[0, mask_token_index, curr_tok_id] 
        gen_word = tokenizer.decode(curr_tok_id)
        prob = logit2prob(logitval) 
        top_5_words.append({'word': str(gen_word).strip(), 'p': prob, 'token_id': curr_tok_id})
        mutable_logits[0, mask_token_index, curr_tok_id] = min_value # replace the max with the min, to get the next max 
        curr_tok_id = mutable_logits[0, mask_token_index].argmax(axis=-1)
        
    top_word = top_5_words[0]['word']
    for k in range(5):
        if top_word not in string.punctuation or k+1 == 5:
            break 
        else:
            top_word = top_5_words[k+1]['word']

    fixed_sent = sent.replace("<mask>", top_word)
    output_sents.append({'original': origsent, 'top': top_word, 'words': top_5_words, 'fixed': fixed_sent})

with open(OUT_PATH, 'w') as g:
    for output_sent_dict in output_sents:
        g.write("original: ")
        g.write(output_sent_dict['original'])
        g.write("pred. word: ")
        g.write(output_sent_dict['top'])
        g.write("\n")
        g.write("top 5 words: ")
        top5 = output_sent_dict['words']
        for j in range(5):
            w = top5[j]['word']
            p = top5[j]['p']
            g.write("({} {}) ".format(str(w), str(p)))
        g.write("\n")
        g.write("full sentence: ")
        g.write(output_sent_dict['fixed'])
        g.write("\n")
        g.write("\n")

