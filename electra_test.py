from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
import re
import pandas as pd
import glob

discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")
pattern = re.compile(r"\[\[[A-Za-z0-9]*\]\]", re.IGNORECASE)

def matches_replacement_pattern(input_word):
    return pattern.match(input_word)

def get_fake_words(input_sentence:str):
    tokens = tokenizer.tokenize(input_sentence, add_special_tokens = True)
    fake_words_indexes = []
    for i in range(len(tokens)):
        if matches_replacement_pattern(tokens[i]) is not None:
            fake_words_indexes.append(i)
    return fake_words_indexes

def get_gt(fake_words:list, length:int)->list:
    gt = [float(0)]*length
    
    for i in fake_words:
        gt[i] = float(1)
    
    return gt

def get_outputs(input_sentence:str):
    fake_words = get_fake_words(input_sentence)
    
    input_sentence_modified = input_sentence.replace('[[', '')
    input_sentence_modified = input_sentence_modified.replace(']]', '')
    input_sentence_modified = input_sentence_modified.replace('<br />', '\n')
    
    fake_tokens = tokenizer.tokenize(input_sentence_modified, add_special_tokens=True)
    fake_inputs = tokenizer.encode(input_sentence_modified, return_tensors="pt")

    discriminator_outputs = discriminator(fake_inputs)
    predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)
    predictions = predictions.squeeze().tolist() 

    return get_gt(fake_words, len(predictions)), predictions 

def determine_score(gt, predictions):
    fp = 0
    tp = 0
    tn = 0
    fn = 0

    # Check positive
    same = True
    for i in range(len(gt)):
        if gt[i] != predictions[i]:
            same = False
            break
    
    return same

def check_if_fake(predictions):
    for i in predictions:
        if int(i) == 1:
            return True
    return False
#def check_tp(gt:list, predictions:list):

def detect_sentence(input_sentence:str):
    gt, predictions = get_outputs(input_sentence)
    return determine_score(gt, predictions)

def check_perf(df:pd.DataFrame):
    postive = 0
    negative = 0
    inputs_not_used = []
    sentences = df['original_text']
    for index, value in sentences.items():
        #print('Index %i: %s\n\n\n' %(index, value))
        
        try:
            if check_if_fake(get_outputs(value)[1]):
                postive += 1
                print("%s is fake" %value)
            else:
                negative += 1
                print("%s is not fake" %value)
        except RuntimeError:
            print('Input too long')
            inputs_not_used.append(index)
    
    print("Check general performance")
    print('\tNot used - %s' %str(inputs_not_used))
    print("\tPos: %i, Neg: %i, Score %f" %(postive, negative, (postive/(postive+negative))))

def check_all_inputs(df:pd.DataFrame):
    postive = 0
    negative = 0
    false_negative = 0
    inputs_not_used = []
    sentences = df['perturbed_text']
    for index, value in sentences.items():
        #print('Index %i: %s\n\n\n' %(index, value))
        
        try:
            if detect_sentence(value):
                postive += 1
            else:
                negative += 1
                
        except RuntimeError:
            print('Input too long')
            inputs_not_used.append(index)
    
    print("Check relative performance")
    print("\tPos: %i, Neg: %i, Score %f" %(postive, negative, (postive/(postive+negative))))
    print('\tNot used - %s' %str(inputs_not_used))

        #print(input_sentence_modified)

def get_all_csvs():
    return [file for file in glob.glob("*.csv")]

def run_test(test_data:str):
    df = pd.read_csv(test_data)
    print("Testing on %s" %test_data)
    #check_perf(df)
    #print('')
    check_all_inputs(df)
    print('\n\n')

def run_all_tests():
    for file in get_all_csvs():
        run_test(file)

run_all_tests()
#if __name__ == "__main__":
    #run_all_tests()
#df = pd.read_csv('qnp_bert.csv')
#check_all_inputs(df)
#check_perf(df)

# discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
# tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")

# sentence = "The quick brown fox jumps over the lazy dog"
# fake_sentence = "The quick brown fox fake over the lazy dog"

# fake_tokens = tokenizer.tokenize(sentence, add_special_tokens=True)
# fake_inputs = tokenizer.encode(sentence, return_tensors="pt")
# discriminator_outputs = discriminator(fake_inputs)
# predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

# print(fake_tokens, len(fake_tokens))

# print(predictions.squeeze().tolist(), len(predictions.squeeze().tolist()))