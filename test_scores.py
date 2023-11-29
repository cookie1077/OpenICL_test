# here we do the testing. haha
# added more codes in the main branch. 

from datasets import load_dataset
from datasets import Dataset, DatasetDict
from openicl import DatasetReader
import json

def gen(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)
            
train_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/train_mini.jsonl"})
val_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/dev_mini.jsonl"})
test_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/test_mini.jsonl"})

dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
# i am changed too. 


# Loading dataset from huggingface
# lets do sst5
#dataset = load_dataset('SetFit/sst5')
dataset = dataset_dict

# Define a DatasetReader, with specified column names where input and output are stored.
# TODO : Define a DatasetReader 
data = DatasetReader(dataset, input_columns=['text'], output_column= 'label')

print(dataset.keys())  # prints the names of the available splits
train_dataset = dataset['train']  # gets the training split
test_dataset = dataset['test']  # gets the testing split

# TODO : Alter PromptTemplate 
from openicl import PromptTemplate
ice_dict = "</E> </label1> </label2> Movie Review: </text>"

tp_dict = {
    0: "</E>Very Negative Movie Review: </text>",
    1: "</E>Negative Movie Review: </text>",
    2: "</E>Neutral Movie Review: </text>" ,
    3: "</E>Positive Movie Review: </text>" ,
    4: "</E>Very Positive Movie Review: </text>" 
}

column_token_map = {'text': '</text>', 'label1' : '</label1>', 'label2' : '</label2>' }
ice_template = PromptTemplate(ice_dict, label_dict, column_token_map, ice_token='</E>')
prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')


from openicl import RandomRetriever
# Define a retriever using the previous `DataLoader`.
# `ice_num` stands for the number of data in in-context examples.
retriever = RandomRetriever(data, ice_num=8, labels= ['0', '1', '2', '3', '4'])

from openicl import PPLInferencer
inferencer = PPLInferencer(model_name='distilgpt2', labels= ['0', '1', '2', '3', '4'])

from openicl import AccEvaluator
# the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
# compute accuracy for the prediction
score = AccEvaluator().score(predictions=predictions, references=data.references)
print(score)
