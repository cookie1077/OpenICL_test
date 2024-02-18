# here we do the testing. haha
# added more codes in the main branch. 

from datasets import load_dataset
from datasets import Dataset, DatasetDict
from openicl import DatasetReader
import json
import vessl 

vessl.init()

def gen(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)
            
train_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/new/train_newagain.jsonl"})
val_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/new/dev.jsonl"})
test_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/new/test.jsonl"})

dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})


# Loading dataset from huggingface
# lets do sst5
#dataset = load_dataset('SetFit/sst5')
dataset = dataset_dict 

print(dataset['test'])
print(dataset['test']['label'])

# Define a DatasetReader, with specified column names where input and output are stored.
# TODO : Define a DatasetReader 
data = DatasetReader(dataset, input_columns=['text'], output_column= 'label')

print(dataset.keys())  # prints the names of the available splits
train_dataset = dataset['train']  # gets the training split
test_dataset = dataset['test']  # gets the testing split

# TODO : Alter PromptTemplate 
from openicl import PromptTemplate


def test_soft(ice_num, data):

    # need to make them show percentage
    ice_dict = "</E> Movie Review: </text> Very Positive </VP>% Positive </P>% Neutral </N>% Negative </NG>% Very Negative </VN>%"
    #ice_dict = "</E>Movie Review: </text> \n</Label1> </Label2>%"

    tp_dict = {
        '0' : "</E>Movie Review: </text> Very Negative",
        '1' : "</E>Movie Review: </text> Negative",
        '2' : "</E>Movie Review: </text> Neutral" ,
        '3' : "</E>Movie Review: </text> Positive" ,
        '4' : "</E>Movie Review: </text> Very Positive" 
    }

    label_dict = {
        '0': "Very Negative",
        '1': "Negative",
        '2': "Neutral",
        '3': "Positive",
        '4': "Very Positive"
    }


    column_token_map = {'text': '</text>', 4 : '</VP>', 3 : '</P>', 2 : '</N>', 1 : '</NG>', 0 : '</VN>' }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')


    from openicl import RandomRetriever
    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = RandomRetriever(data, ice_num=ice_num, labels= ['0', '1', '2', '3', '4'], order=True)

    from openicl import PPLInferencer
    inferencer = PPLInferencer(model_name='distilgpt2', labels= ['0', '1', '2', '3', '4'])

    from openicl import AccEvaluator
    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)
    
    return score

# Test for sequence
def test_sequence(ice_num, data):

    # need to make them show percentage
    ice_dict = "</E>Movie Review: </text> </Label1> </1>% </Label2> </2>% </Label3> </3>% </Label4> </4>% </Label5> </5>%"
    #ice_dict = "</E>Movie Review: </text> \n</Label1> </Label2>%"

    tp_dict = {
        '0' : "</E>Movie Review: </text> Very Negative",
        '1' : "</E>Movie Review: </text> Negative",
        '2' : "</E>Movie Review: </text> Neutral" ,
        '3' : "</E>Movie Review: </text> Positive" ,
        '4' : "</E>Movie Review: </text> Very Positive" 
    }

    label_dict = {
        '0': "Very Negative",
        '1': "Negative",
        '2': "Neutral",
        '3': "Positive",
        '4': "Very Positive"
    }


    column_token_map = {'text': '</text>', 0 : '</1>', 'Label1' : '</Label1>', 1 : '</2>', 'Label2' : '</Label2>',2 : '</3>', 'Label3' : '</Label3>',3 : '</4>', 'Label4' : '</Label4>',4 : '</5>', 'Label5' : '</Label5>' }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')


    from openicl import RandomRetriever
    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = RandomRetriever(data, ice_num=ice_num, labels= ['0', '1', '2', '3', '4'], order=True)

    from openicl import PPLInferencer
    inferencer = PPLInferencer(model_name='distilgpt2', labels= ['0', '1', '2', '3', '4'])

    from openicl import AccEvaluator
    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)
    
    return score

def test_origin(ice_num, data):
    # need to make them show percentage
    ice_dict = "</E>Movie Review: </text> Positive </P>% Negative </N>%"

    ice_dict2 = {
        0 : "</E>Movie Review: </text> Very Negative",
        1 : "</E>Movie Review: </text> Negative",
        2 : "</E>Movie Review: </text> Neutral" ,
        3 : "</E>Movie Review: </text> Positive" ,
        4 : "</E>Movie Review: </text> Very Positive" 
    }

    tp_dict = {
        0 : "</E>Movie Review: </text> Very Negative",
        1 : "</E>Movie Review: </text> Negative",
        2 : "</E>Movie Review: </text> Neutral" ,
        3 : "</E>Movie Review: </text> Positive" ,
        4 : "</E>Movie Review: </text> Very Positive" 
    }

    column_token_map = {'text': '</text>', '0' : '</P>', '1' : '</N>' }
    ice_template = PromptTemplate(ice_dict2, column_token_map, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')


    from openicl import RandomRetriever
    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = RandomRetriever(data, ice_num=ice_num, labels= [0,1,2,3,4] )

    from openicl import PPLInferencer
    inferencer = PPLInferencer(model_name='distilgpt2', labels= [0,1,2,3,4])

    from openicl import AccEvaluator
    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)
    
    return score


soft = []
sequence = []
origin = []
x = [n for n in range(5)]

for i in range(5):
    soft.append(test_soft(i, data)['accuracy'])
    sequence.append(test_sequence(i, data)['accuracy'])
    origin.append(test_origin(i, data)['accuracy'])

print(soft)
print(sequence)
print(origin)

import matplotlib.pyplot as plt

plt.plot(x, sequence)
plt.plot(x, origin)

plt.legend()