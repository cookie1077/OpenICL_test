# here we do the testing. haha
# added more codes in the main branch. 

from datasets import load_dataset
from datasets import Dataset, DatasetDict
from openicl import DatasetReader
import json

# Before : preprocess data !

def gen(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)
            
train_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/train_mini.jsonl"})
val_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/dev.jsonl"})
test_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/train_mini.jsonl"})

dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
dataset = dataset_dict

# Define a DatasetReader, with specified column names where input and output are stored.
data = DatasetReader(dataset, input_columns=['text'], output_column='label')

print(dataset.keys())  # prints the names of the available splits
train_dataset = dataset['train']  # gets the training split
test_dataset = dataset['test']  # gets the testing split

from openicl import PromptTemplate
tp_dict = {
    0: "</E>Very Negative Movie Review: </text>",
    1: "</E>Negative Movie Review: </text>",
    2: "</E>Neutral Movie Review: </text>" ,
    3: "</E>Positive Movie Review: </text>" ,
    4: "</E>Very Positive Movie Review: </text>" 
}

template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')


from openicl import RandomRetriever
# Define a retriever using the previous `DataLoader`.
# `ice_num` stands for the number of data in in-context examples.
retriever = RandomRetriever(data, ice_num=3)

import ParentInferencer
inferencer = ParentInferencer.ParentInferencer(model_name='EleutherAI/gpt-j-6b')

from openicl import AccEvaluator
# the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
predictions = inferencer.inference(retriever, ice_template=template)

for i, p in enumerate(predictions):
    p["text"] = dataset_dict["test"][i]["text"]

#print(predictions)

# Save predictions as file ! 

with open('data/train_mini_new.jsonl', 'w') as f:
    for entry in predictions:
        json.dump(entry, f)
        f.write('\n')


'''


# Prepare variables



# 1. Preparation for output logs
# the reason for using output handler?
output_handler = PPLInferencerOutputHandler(self.accelerator)

sub_predictions = []
ppl = []
ice = []
output_json_filename = ""
output_json_filepath = ""

retriever = RandomRetriever(data, ice_num=8)
# 2. Get results of retrieval process
# index of example labels. rather, giving the example itself would be more helpful. 
ice_idx_list = retriever.retrieve()

# 3. Get labels of all the classes
# what does this code return? the keys and prompts of labels. 

from openicl import PromptTemplate
tp_dict = {
    0: "</E>Very Negative Movie Review: </text>",
    1: "</E>Negative Movie Review: </text>",
    2: "</E>Neutral Movie Review: </text>" ,
    3: "</E>Positive Movie Review: </text>" ,
    4: "</E>Very Positive Movie Review: </text>" 
}

template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')
labels = list(template.template.keys())[:]

# instead, we do...
ice_eos_token  = '\n'
ice_separator = '\n'

ice_template = template
prompt_template = template

# 4. Generate in-context examples for testing inputs
for idx in range(len(ice_idx_list)):
    ice.append(retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template))

output_handler.save_ice(ice)


# 5. Calculating PPL for prompts in each label's class
for label in labels:
    index = 0
    prompt_list = []
    sub_ppl_list = []
    normalizing_prompt_list = []
    context_length_list = []

    # 5.1 Generate prompts of current label and truncate 
    for idx in range(len(ice_idx_list)):
        prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template,
                                                    prompt_template=prompt_template,
                                                    remain_sep=normalizing_str is not None)
        if self.max_model_token_num is not None and self.api_name != 'gpt3':
            prompt_token_num = self.get_input_token_num(prompt)
            while len(ice_idx_list[idx]) > 0 and prompt_token_num > self.max_model_token_num:
                ice_idx_list[idx] = ice_idx_list[idx][:-1]
                ice[idx] = retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template)
                prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template,
                                                            prompt_template=prompt_template)
                prompt_token_num = self.get_input_token_num(prompt)

        if normalizing_str is not None:
            prompt_sep = prompt
            if prompt_template is not None:
                sep_token = prompt_template.sep_token
            else:
                sep_token = ice_template.sep_token
            sep_pos = prompt_sep.find(sep_token)

            context = prompt_sep[0:sep_pos]
            answer = prompt_sep[sep_pos:].replace(sep_token, '')
            prompt = context + answer
            normalizing_prompt = normalizing_str + answer

            context_length_list.append(self.get_input_token_num(context))
            normalizing_prompt_list.append(normalizing_prompt)
        prompt_list.append(prompt)

    if normalizing_str is not None:
        normalizing_str_len = self.get_input_token_num(normalizing_str)

    # 5.2 Get PPL
    logger.info(f"Calculating PPL for prompts labeled '{label}'")
    for idx in trange(0, len(prompt_list), self.batch_size, disable=not self.is_main_process):
        sub_prompt_list = prompt_list[idx:idx + self.batch_size]
        if normalizing_str is not None:
            sub_context_length_list = context_length_list[idx:idx + self.batch_size]
            sub_normalizing_prompt_list = normalizing_prompt_list[idx:idx + self.batch_size]

        with torch.no_grad():
            if normalizing_str is not None:
                res1 = self.__get_ppl(input_texts=sub_prompt_list, mask_length=sub_context_length_list)
                res2 = self.__get_ppl(input_texts=sub_normalizing_prompt_list,
                                        mask_length=[normalizing_str_len for i in range(len(sub_prompt_list))]
                                        )
                sub_res = res1 - res2
            else:
                sub_res = self.__get_ppl(sub_prompt_list).tolist()
        for res, prompt in zip(sub_res, sub_prompt_list):
            sub_ppl_list.append(res)
            output_handler.save_prompt_and_ppl(label, prompt[len(ice[idx]):], prompt, res, index)
            index = index + 1
    ppl.append(sub_ppl_list)

# 6. Get lowest PPL class as predictions
ppl = list(zip(*ppl))
for single_ppl in ppl:
    sub_predictions.append(labels[single_ppl.index(min(single_ppl))])
output_handler.save_predictions(sub_predictions)

# 7. Output
output_handler.subprocess_write_to_json(output_json_filepath, output_json_filename)
if self.accelerator is not None:
    self.accelerator.wait_for_everyone()
output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
output_handler.write_to_json(output_json_filepath, output_json_filename)

#return [sample['prediction'] for sample in output_handler.results_dict.values()]

'''