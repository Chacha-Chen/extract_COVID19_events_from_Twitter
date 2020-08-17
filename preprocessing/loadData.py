
from transformers import BertTokenizer, BertTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader

import logging
import os
from preprocessing.preprocessData import splitDatasetIntoTrainDevTest, preprocessDataAndSave
from utils import loadFromPickleFile
from preprocessing import const

class COVID19TaskDataset(Dataset):

    def __init__(self, instance_list):
        super(COVID19TaskDataset, self).__init__()
        self.instance_list = instance_list

    def __getitem__(self, index):
        return self.instance_list[index]

    def __len__(self):
        return len(self.instance_list)

class TokenizeCollator():

    def __init__(self, tokenizer, subtask_list, entity_start_token_id):
        
        self.tokenizer = tokenizer
        self.subtask_list = subtask_list
        self.entity_start_token_id = entity_start_token_id

    def __call__(self, batch):

        # Prepare Result
        gold_label_dict_batch = {subtask: [] for subtask in self.subtask_list}
        input_text_list_batch = []

        for input_text, subtask_label_dict in batch:
            input_text_list_batch.append(input_text)

            for subtask in self.subtask_list:
                gold_label_dict_batch[subtask].append(subtask_label_dict[subtask][1]) # 0 is gold chunk

        # Send to BERT's tokenizer
        tokenized_input_text_list_batch = self.tokenizer.batch_encode_plus(
            input_text_list_batch, pad_to_max_length=True, return_tensors='pt')

        input_ids = tokenized_input_text_list_batch['input_ids']
        token_type_ids = tokenized_input_text_list_batch['token_type_ids']
        attention_mask = tokenized_input_text_list_batch['attention_mask']

        # Further processing
        # entity_start_positions = (input_ids == self.entity_start_token_id).nonzero()
        entity_start_positions = torch.nonzero(input_ids == self.entity_start_token_id, as_tuple=False)

        input_label_dict = {
            subtask: torch.LongTensor(gold_label_list)
            for subtask, gold_label_list in gold_label_dict_batch.items()
        }
        if entity_start_positions.size(0) == 0:
            # Send entity_start_positions to [CLS]'s position i.e. 0
            entity_start_positions = torch.zeros(input_ids.size(0), 2).long()
        
        # DEBUG
        for subtask in self.subtask_list:
            assert input_ids.size(0) == input_label_dict[subtask].size(0)
        
        return {
            'input_ids': input_ids,
            'entity_start_positions': entity_start_positions,
            'gold_labels': input_label_dict,
            'batch_data': batch
        }


def loadData (event,
              entity_start_token_id,
              tokenizer,
              batch_size = 8,
              train_ratio = 0.6, dev_ratio = 0.15,
              shuffle_train_data_flg = True, num_workers = 0,
              input_text_processing_func_list=[],
              gold_chunk_processing_func_list=[]):
    """Return DataLoader for train/dev/test and subtask_list

    Input:
    event -- name of event, one of 
             ['positive', 'negative', 'can_not_test', 'death', 'cure_and_prevention']
    tokenizer
    
    
    Keyword Arguments:
    batch_size  -- [default 8]
    train_ratio -- [default 0.6]
    dev_ratio   -- [default 0.15]
    shuffle_train_data_flg -- whether shuffle train DataLoader [default True]
    num_workers -- [default 0]
    
    """

# DEBUG
# event = 'cure_and_prevention'
# tokenizer_class = BertTokenizer
# pretrained_tokenizer = 'bert-base-cased'
# save_directory = '.'
# batch_size = 8
# train_ratio = 0.6
# dev_ratio = 0.15
# shuffle_train_data_flg = True
# num_workers = 0

    # Init Tokenizer
    # entity_start_token_id = tokenizer.convert_tokens_to_ids(["<E>"])[0]

    # Load Data
    preprocessed_data_file = os.path.join(const.DATA_FOLDER, f'{event}-preprocessed-data.pkl')
    # 
    if not os.path.isfile(preprocessed_data_file):
        # TODO use logging module
        print(f"File {preprocessed_data_file} doesn't exist, generating...")
        preprocessDataAndSave(event)
    # 
    subtask_list, raw_input_text_and_label_list = loadFromPickleFile(preprocessed_data_file)

    if input_text_processing_func_list:
        tmp_list = []
        for tweet_text, input_text, subtask_label_dict in raw_input_text_and_label_list:

            for processing_func in input_text_processing_func_list:
                input_text = processing_func(input_text)
            tmp_list.append((tweet_text, input_text, subtask_label_dict))
        # TODO do we need to expliticly del(tmp_list)?
        raw_input_text_and_label_list = tmp_list

    if gold_chunk_processing_func_list:
        tmp_list = []
        for tweet_text, input_text, subtask_label_dict in raw_input_text_and_label_list:

            for subtask, (gold_chunk_list, gold_label) in subtask_label_dict.items():
                for processing_func in gold_chunk_processing_func_list:
                    gold_chunk_list = [processing_func(gold_chunk) for gold_chunk in gold_chunk_list]
                    subtask_label_dict[subtask] = (gold_chunk_list, gold_label)

            tmp_list.append((tweet_text, input_text, subtask_label_dict))
        raw_input_text_and_label_list = tmp_list


    (train_instance_list,
     dev_instance_list,
     test_instance_list) = splitDatasetIntoTrainDevTest(
         raw_input_text_and_label_list, train_ratio=train_ratio, dev_ratio=dev_ratio)

    # TODO move to logging
    print(f"Dataset Size Report: {len(train_instance_list)} / "
          f"{len(dev_instance_list)} / {len(test_instance_list)} (train/dev/test)")

    train_dataset = COVID19TaskDataset(train_instance_list)
    dev_dataset   = COVID19TaskDataset(dev_instance_list)
    test_dataset  = COVID19TaskDataset(test_instance_list)

    collate_fn = TokenizeCollator(tokenizer, subtask_list, entity_start_token_id)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train_data_flg, num_workers=num_workers,
        collate_fn = collate_fn)
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn = collate_fn)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn = collate_fn)

    return train_dataloader, dev_dataloader, test_dataloader, subtask_list
