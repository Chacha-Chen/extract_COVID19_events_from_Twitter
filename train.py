from transformers import BertTokenizerFast, BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
from model import MultiTaskBertForCovidEntityClassification,MultiTaskBertForCovidEntityClassification_new
import numpy as np
from preprocessing.loadData import loadData
from multitask_bert_entitity_classifier import load_from_pickle, get_multitask_instances_for_valid_tasks, COVID19TaskDataset, TokenizeCollator, format_time, plot_train_loss,split_data_based_on_subtasks, make_predictions_on_dataset,make_dir_if_not_exists
from utils import log_list, split_multitask_instances_in_train_dev_test, log_data_statistics, save_in_json, get_raw_scores, get_TP_FP_FN
import logging
from collections import Counter
import pickle
from pprint import pprint
import copy
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import os
from tqdm import tqdm
import argparse
import time
import datetime
import string
import re
import collections
from pprint import pprint
# DEBUG_FLAG = True
YAQI_FLAG = True
################### util ####################
def parse_arg():
    # TODO: add the following arguments
    # (1) gpu, default 0
    # (2) learning rate, default 2e-5

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file", help="Path to the pickle file that contains the training instances", type=str, default='./data/test_positive.pkl')
    # parser.add_argument("-d", "--logfile", help="Path to the pickle file that contains the training instances", type=str, required=True)
    parser.add_argument("-t", "--event", help="Event for which we want to train the baseline", type=str, default='positive') # ['positive', 'negative', 'can_not_test', 'death', 'cure_and_prevention']
    parser.add_argument("-s", "--save_directory", help="Path to the directory where we will save model and the tokenizer", type=str, default='saved_models/multitask_bert/debug_1')
    parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all the model results", type=str, default='./results/debug_bert')
    parser.add_argument("-rt", "--retrain", help="Flag that will indicate if the model needs to be retrained or loaded from the existing save_directory", action="store_true", default=True) ##TODO implementation
    parser.add_argument("-bs", "--batch_size", help="Train batch size for BERT model", type=int, default=8)
    parser.add_argument("-e", "--n_epochs", help="Number of epochs", type=int, default=1)
    return parser.parse_args()

def train_valid_test_split(index, train, valid, test):
    # index: np.array of index
    # train: ratio of training data
    # valid: ratio of validation data
    # test: ratio of testing data

    assert train+valid+test == 1.0
    
    np.random.shuffle(index)
    length = index.shape[0]

    train_num = length * train
    valid_num = length * valid

    return (
        np.sort(index[:train_num]),
        np.sort(index[train_num:train_num+valid_num]),
        np.sort(index[train_num+valid_num:])
    )

################### training functions ####################
# dataset
# testing/val script
def evaluation(model, dataloader, device, threshold=0.5):
    model.eval()
    
    for step, batch in enumerate(dataloader):
    
        for subtask in model.subtasks:
            subtask_labels = batch["gold_labels"][subtask]
            subtask_labels = subtask_labels.to(device)
            batch["gold_labels"][subtask] = subtask_labels

        input_dict = {"input_ids": batch["input_ids"].to(device),
                      "entity_start_positions": batch["entity_start_positions"].to(device),
                      "labels": batch["gold_labels"]}
        
        logits, _ = model(**input_dict)
        
        # Post-model subtask information aggregation.
        dev_logits = torch.stack(logits, dim=1).type(torch.float)
        dev_labels = torch.stack([batch["gold_labels"][subtask] for subtask in model.subtasks], dim =1).type(torch.int)
    
    # Moving to cpu for evaluations.
    dev_logits = dev_logits.detach().cpu().numpy()
    dev_labels = dev_labels.detach().cpu().numpy()
    
    # Assessment on the results according to labels and logits.  
    prediction = (dev_logits[subtask] > threshold).astype(int)
    
    # Calculating metrics
    precision = np.array([metrics.precision_score(prediction[:,idx], dev_labels[:,idx]) for idx in range(dev_labels.shape[1])])
    recall = np.array([metrics.recall_score(prediction[:,idx], dev_labels[:,idx]) for idx in range(dev_labels.shape[1])])
    f1 = np.array([metrics.f1_score(prediction[:,idx], dev_labels[:,idx]) for idx in range(dev_labels.shape[1])])
    confusion_matrix = np.array([metrics.confusion_matrix(prediction[:,idx], dev_labels[:,idx]).ravel() for idx in range(dev_labels.shape[1])])
    
    return precision, recall, f1, prediction, confusion_matrix

# prediction script
def make_prediction(model, dataloader, device, threshold=0.5):
    # run model and predict without having "y" label
    # only return the prediction
    
    model.eval()
    
    for step, batch in enumerate(dataloader):

        input_dict = {"input_ids": batch["input_ids"].to(device),
                      "entity_start_positions": batch["entity_start_positions"].to(device)}
        
        logits, _ = model(**input_dict)
        
        # Post-model subtask information aggregation.
        test_logits = torch.stack(logits, dim=1).type(torch.float)
    
    # Moving to cpu for evaluations.
    test_logits = test_logits.detach().cpu().numpy()
    
    # Assessment on the results according to labels and logits.  
    prediction = (test_logits[subtask] > threshold).astype(int)
    
    
    return prediction

def post_processing(model, valid_dataloader, test_dataloader, device):
    # Save the model name in the model_config file
    model_config = dict()
    results = dict()
    model_config["model"] = "MultiTaskBertForCovidEntityClassification_new"
    model_config["epochs"] = args.n_epochs
    
    # Check different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Evaluate on different thresholds in val set
    dev_results = [evaluation(model, valid_dataloader, t, device) for t in thresholds]
    
    # Finding out the best threshold
    # Getting the f1 scores on different thresholds and in every subtasks
    dev_f1_scores = np.array([dev_results[2] for t in thresholds]) # (thresholds_idx, subtask_idx)
    # Calculating the best thresholds indices on different subtasks
    best_dev_thresholds_idx = np.argmax(dev_f1_scores, axis=0)
    
    assert(best_dev_thresholds_idx.size[0] == len(model.subtasks), "Expected subtask size: "+len(model.subtasks)+" with calculated "+best_dev_thresholds_idx.size+".")
    
    best_dev_F1s = {}
    best_dev_thresholds = {}
    dev_subtasks_t_F1_P_Rs = {}
    for subtask_idx in range(best_dev_thresholds_idx.size[0]):
        # Find the subtasks using index
        subtask = model.subtasks[subtask_idx]
        # Find the thresholds of that task using index
        best_thresholds_idx = best_dev_thresholds_idx[subtask_idx]
        
        # Find bests
        # Find the best F1 of that task using index
        best_dev_F1s[subtask] = dev_f1_scores[best_thresholds_idx, subtask_idx]
        # Find the best threshold of that task using index
        best_dev_thresholds[subtask] = thresholds[best_thresholds_idx]
        
        # Log all results in output formats
        for t in thresholds:
            dev_P = dev_results[t][0][subtask_idx]
            dev_R = dev_results[t][1][subtask_idx]
            dev_F1 = dev_results[t][2][subtask_idx]
            dev_TN, dev_FP, dev_FN, dev_TP = dev_results[t][4][subtask_idx]
            
            dev_subtasks_t_F1_P_Rs[subtask].append((t, dev_F1, dev_P, dev_R, dev_TP + dev_FN, dev_TP, dev_FP, dev_FN)) # Copy-pasted from original code


    # Save model_config and results
    model_config_file = os.path.join(args.output_dir, "model_config.json")
    results_file = os.path.join(args.output_dir, "results.json")
    logging.info(f"Saving model config at {model_config_file}")
    save_in_json(model_config, model_config_file)
    logging.info(f"Saving results at {results_file}")
    save_in_json(results, results_file)
    
    
    
'''
def post_processing(model,args,dev_dataloader,device,dev_subtasks_data,test_subtasks_data,test_dataloader):
    # Save the model name in the model_config file
    model_config = dict()
    results = dict()
    model_config["model"] = "MultiTaskBertForCovidEntityClassification"
    model_config["epochs"] = args.n_epochs

    # Find best threshold for each subtask based on dev set performance
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dev_predicted_labels, dev_prediction_scores, dev_gold_labels = make_predictions_on_dataset(dev_dataloader, model,
                                                                                               device,
                                                                                               args.task + "_dev", True)
    best_dev_thresholds = {subtask: 0.5 for subtask in model.subtasks}
    best_dev_F1s = {subtask: 0.0 for subtask in model.subtasks}
    dev_subtasks_t_F1_P_Rs = {subtask: list() for subtask in model.subtasks}

    for subtask in model.subtasks:
        dev_subtask_data = dev_subtasks_data[subtask]
        dev_subtask_prediction_scores = dev_prediction_scores[subtask]
        for t in thresholds:
            dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN = get_TP_FP_FN(dev_subtask_data, dev_subtask_prediction_scores,
                                                                        THRESHOLD=t)
            dev_subtasks_t_F1_P_Rs[subtask].append((t, dev_F1, dev_P, dev_R, dev_TP + dev_FN, dev_TP, dev_FP, dev_FN))
            if dev_F1 > best_dev_F1s[subtask]:
                best_dev_thresholds[subtask] = t
                best_dev_F1s[subtask] = dev_F1

        logging.info(f"Subtask:{subtask:>15}")
        log_list(dev_subtasks_t_F1_P_Rs[subtask])
        logging.info(
            f"Best Dev Threshold for subtask: {best_dev_thresholds[subtask]}\t Best dev F1: {best_dev_F1s[subtask]}")

    # Save the best dev threshold and dev_F1 in results dict
    results["best_dev_threshold"] = best_dev_thresholds
    results["best_dev_F1s"] = best_dev_F1s
    results["dev_t_F1_P_Rs"] = dev_subtasks_t_F1_P_Rs

    # Evaluate on Test
    logging.info("Testing on test dataset")

    predicted_labels, prediction_scores, gold_labels = make_predictions_on_dataset(test_dataloader, model, device,
                                                                                   args.task)

    # Test
    for subtask in model.subtasks:
        logging.info(f"Testing the trained classifier on subtask: {subtask}")
        results[subtask] = dict()
        cm = metrics.confusion_matrix(gold_labels[subtask], predicted_labels[subtask])
        classification_report = metrics.classification_report(gold_labels[subtask], predicted_labels[subtask],
                                                              output_dict=True)
        logging.info(cm)
        logging.info(metrics.classification_report(gold_labels[subtask], predicted_labels[subtask]))
        results[subtask]["CM"] = cm.tolist()  # Storing it as list of lists instead of numpy.ndarray
        results[subtask]["Classification Report"] = classification_report

        # SQuAD style EM and F1 evaluation for all test cases and for positive test cases (i.e. for cases where annotators had a gold annotation)
        EM_score, F1_score, total = get_raw_scores(test_subtasks_data[subtask], prediction_scores[subtask])
        logging.info("Word overlap based SQuAD evaluation style metrics:")
        logging.info(f"Total number of cases: {total}")
        logging.info(f"EM_score: {EM_score}")
        logging.info(f"F1_score: {F1_score}")
        results[subtask]["SQuAD_EM"] = EM_score
        results[subtask]["SQuAD_F1"] = F1_score
        results[subtask]["SQuAD_total"] = total
        pos_EM_score, pos_F1_score, pos_total = get_raw_scores(test_subtasks_data[subtask], prediction_scores[subtask],
                                                               positive_only=True)
        logging.info(f"Total number of Positive cases: {pos_total}")
        logging.info(f"Pos. EM_score: {pos_EM_score}")
        logging.info(f"Pos. F1_score: {pos_F1_score}")
        results[subtask]["SQuAD_Pos. EM"] = pos_EM_score
        results[subtask]["SQuAD_Pos. F1"] = pos_F1_score
        results[subtask]["SQuAD_Pos. EM_F1_total"] = pos_total

        # New evaluation suggested by Alan
        F1, P, R, TP, FP, FN = get_TP_FP_FN(test_subtasks_data[subtask], prediction_scores[subtask],
                                            THRESHOLD=best_dev_thresholds[subtask])
        logging.info("New evaluation scores:")
        logging.info(f"F1: {F1}")
        logging.info(f"Precision: {P}")
        logging.info(f"Recall: {R}")
        logging.info(f"True Positive: {TP}")
        logging.info(f"False Positive: {FP}")
        logging.info(f"False Negative: {FN}")
        results[subtask]["F1"] = F1
        results[subtask]["P"] = P
        results[subtask]["R"] = R
        results[subtask]["TP"] = TP
        results[subtask]["FP"] = FP
        results[subtask]["FN"] = FN
        N = TP + FN
        results[subtask]["N"] = N

    # Save model_config and results
    model_config_file = os.path.join(args.output_dir, "model_config.json")
    results_file = os.path.join(args.output_dir, "results.json")
    logging.info(f"Saving model config at {model_config_file}")
    save_in_json(model_config, model_config_file)
    logging.info(f"Saving results at {results_file}")
    save_in_json(results, results_file)
'''
# training function
def train():
    args = parse_arg()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    make_dir_if_not_exists(args.output_dir)
    make_dir_if_not_exists(args.save_directory)
    logfile =  os.path.join(args.output_dir, "train_output.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

    # parameter setting
    max_len = 100       # TODO: compute the statistic of the length
    # subtask_num = 5     # might need to re-assign value after loading the data


    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU{torch.cuda.get_device_name(0)} to train")
    else:
        device = torch.device("cpu")
        logging.info(f"Using CPU to train")
    # device = f"cuda:{args.gpu}"
    # ====== data loading
    if YAQI_FLAG:
        ## TODO: yaqi, plz implement this part
        ## TODO goal: get train_dataloader; valid_dataloader; test_dataloader
        # x, y, subtasks = load_data() # TODO: implement load_data function
                                    #       check the data preprocessing code
                                    # x.shpae = [# instances, sequence_length] (Bert tokenized index)
                                    # y.shape = [# instances] (0, 1)
                                    # subtasks.shape = [#subtask] list of subtasks
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        tokenizer.add_tokens(["<E>", "</E>", "<URL>", "@USER"])
        tokenizer.save_pretrained(args.save_directory)
        entity_start_token_id = tokenizer.convert_tokens_to_ids(["<E>"])[0]

        (train_dataloader, valid_dataloader, test_dataloader, subtask_list) = loadData(
            args.event, entity_start_token_id, tokenizer,
            batch_size=8, train_ratio=0.6, dev_ratio=0.15, shuffle_train_data_flg=True, num_workers=0)

        config = BertConfig.from_pretrained('bert-base-cased')
        config.subtasks = subtask_list
        # # model = MultiTaskBertForCovidEntityClassification.from_pretrained('bert-base-cased', config=config)
        model = MultiTaskBertForCovidEntityClassification_new.from_pretrained('bert-base-cased', config=config)
        model.resize_token_embeddings(len(tokenizer))
        #
        # # data split (TODO: a better way to ensure the balance of subtask)
        # train_index, valid_index, test_index = train_valid_test_split(index=np.arange(x.shape[0]), train=0.8, valid=0.1, test=0.1)
        # x_train, y_train = x[train_index], y[train_index]
        # x_valid, y_valid = x[valid_index], y[valid_index]
        # x_test, y_test = x[test_index], y[test_index]
        # train_dataset = COVID19TaskDataset(x_train, y_train)
        # valid_dataset = COVID19TaskDataset(x_valid, y_valid)
        # test_dataset = COVID19TaskDataset(x_test, y_test)
        # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        # valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    else:
        task_instances_dict, tag_statistics, _ = load_from_pickle(args.data_file) ## TODO adhoc
        logging.info(f"Task dataset for task: {args.data_file} loaded from {args.data_file}.")
        data, subtasks_list = get_multitask_instances_for_valid_tasks(task_instances_dict, tag_statistics)
        train_data, dev_data, test_data = split_multitask_instances_in_train_dev_test(data)
        config = BertConfig.from_pretrained('bert-base-cased')
        config.subtasks = subtasks_list
        # model = MultiTaskBertForCovidEntityClassification.from_pretrained('bert-base-cased', config=config)
        model = MultiTaskBertForCovidEntityClassification_new.from_pretrained('bert-base-cased', config=config)
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        tokenizer.add_tokens(["<E>", "</E>", "<URL>", "@USER"])
        tokenizer.save_pretrained(args.save_directory)
        model.resize_token_embeddings(len(tokenizer))
        entity_start_token_id = tokenizer.convert_tokens_to_ids(["<E>"])[0]
        dev_subtasks_data = split_data_based_on_subtasks(dev_data, model.subtasks)
        test_subtasks_data = split_data_based_on_subtasks(test_data, model.subtasks)
        train_dataset = COVID19TaskDataset(train_data)
        dev_dataset = COVID19TaskDataset(dev_data)
        test_dataset = COVID19TaskDataset(test_data)
        logging.info("Loaded the datasets into Pytorch datasets")
        tokenize_collator = TokenizeCollator(tokenizer, model.subtasks, entity_start_token_id)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                      collate_fn=tokenize_collator)
        valid_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                    collate_fn=tokenize_collator)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                     collate_fn=tokenize_collator)
        logging.info("Created train and test dataloaders with batch aggregation")

    # init model & modify the embedding
    # TODO: the tokenizer part might need to be loaded before loading data

    model.to(device)  ## TODO old model move classifier
    # if not YAQI_FLAG:
    #     for subtask, classifier in model.classifiers.items():
    #         classifier.to(device)

    # Load the instances into pytorch dataset
    # TODO: Do we need to implement bucketIterator version? do it later
    # Load the instances into pytorch dataset


    # init optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    best_model = None
    best_score = 0.0
    best_epoch = 0
    training_stats = []
    epoch_train_loss = list()
    # start training
    logging.info(f"Initiating training loop for {args.n_epochs} epochs...")
    total_start_time = time.time()

    for epoch in range(args.n_epochs):
        model.train()
        pbar = tqdm(train_dataloader)

        # Reset the total loss for each epoch.
        total_train_loss = 0
        avg_train_loss = 0
        train_loss_trajectory = list()

        dev_log_frequency = 5
        n_steps = len(train_dataloader)
        dev_steps = int(n_steps / dev_log_frequency)

        start_time = time.time()
        for step, batch in enumerate(pbar):
            model.zero_grad()

            for subtask in model.subtasks:
                subtask_labels = batch["gold_labels"][subtask]
                subtask_labels = subtask_labels.to(device)
                batch["gold_labels"][subtask] = subtask_labels

            input_dict = {"input_ids": batch["input_ids"].to(device),
                          "entity_start_positions": batch["entity_start_positions"].to(device),
                          "labels": batch["gold_labels"]}

            # Forward
            # x = x.to(device)
            # y = y.to(device)
            # subtask = subtask.to(device)
            # entity_position = entity_position.to(device)
            logits, loss = model(**input_dict)
            loss.backward()

            total_train_loss += loss.item()
            avg_train_loss = total_train_loss/(step+1)
            
            pbar.set_description(f"Epoch:{epoch+1}|Batch:{step}/{len(train_dataloader)}[{100.0*step/len(train_dataloader)}%]|Avg. Loss:{avg_train_loss:.4f}")


            elapsed = format_time(time.time() - start_time)
            avg_train_loss = total_train_loss / (step + 1)

            # keep track of changing avg_train_loss
            train_loss_trajectory.append(avg_train_loss)
            pbar.set_description(
                f"Epoch:{epoch + 1}|Batch:{step}/{len(train_dataloader)}|Time:{elapsed}|Avg. Loss:{avg_train_loss:.4f}|Loss:{loss.item():.4f}")

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            pbar.update()

            # run validation
            if (step + 1) % dev_steps == 0:
                # model.eval()
                ##   TODO evaluation
                precision, recall, f1, prediction, _ = evaluation(model, valid_dataloader, device=device)
                print(f"Validation result. Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
                #
                # # check if this is the best model or not
                # if f1 >= best_score:
                #     best_model = copy.deepcopy(model.state_dict())
                #     best_score = f1
                #     best_epoch = epoch
                model.train()
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - start_time)

        # Record all statistics from this epoch.
        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Training Time': training_time})

        # Save the loss trajectory
        epoch_train_loss.append(train_loss_trajectory)

    # finish training
    print("Finished Training!")
    logging.info(f"Training complete with total Train time:{format_time(time.time() - total_start_time)}")
    log_list(training_stats)

    print(f"Best Validation Score = {best_score} at {best_epoch}")
    # model.load_state_dict(best_model) ##TODO best model
    model.save_pretrained(args.save_directory)

    # Plot the train loss trajectory in a plot
    train_loss_trajectory_plot_file = os.path.join(args.output_dir, "train_loss_trajectory.png")
    logging.info(f"Saving the Train loss trajectory at {train_loss_trajectory_plot_file}")
    plot_train_loss(epoch_train_loss, train_loss_trajectory_plot_file)

    # running testing
    # TODO: post-processing
    # (1) probing threshold for each subtask
    # (2) save threshold 
    # precision, recall, f1, prediction = evaluation(model, test_dataloader, device=device)
    post_processing(args, model, valid_dataloader, test_dataloader, device=device)

def main():
    train()



if __name__ == "__main__":
    main()
