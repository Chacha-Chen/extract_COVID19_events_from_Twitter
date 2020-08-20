from transformers import BertTokenizerFast, BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
import csv
from model import MultiTaskBertForCovidEntityClassification,MultiTaskBertForCovidEntityClassification_new
import numpy as np
from preprocessing.loadData import loadData
from preprocessing.utils import make_dir_if_not_exists, format_time, log_list, plot_train_loss, saveToJSONFile, loadFromJSONFile
import logging
import copy
from sklearn import metrics
import os
from tqdm import tqdm
import argparse
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

YAQI_FLAG = True

EVENT_LIST = ['positive', 'negative', 'can_not_test', 'death', 'cure_and_prevention']
################### util ####################
def parse_arg():
    # TODO: add the following arguments
    # (1) gpu, default 0
    # (2) learning rate, default 2e-5

    parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--data_file", help="Path to the data file", type=str, default='./data/test_positive.pkl')
    parser.add_argument("-o", "--output_dir", help="Path to the output directory", type=str, default='./results/8_epochs_test_0819')
    parser.add_argument("-rt", "--retrain", help="True if the model needs to be retrained", action="store_false", default=False)
    parser.add_argument("-bs", "--batch_size", help="Train batch size for BERT model", type=int, default=32)
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
    
    total_logits = []
    total_labels = []
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
        
        logits = list(logits.detach().cpu().numpy())
        total_logits += logits
        total_labels += list(np.array([list(batch["gold_labels"][subtask].detach().cpu().numpy()) for subtask in model.subtasks]).T)
        
    
    total_logits = np.array(total_logits)
    total_labels = np.array(total_labels)
    # Assessment on the results according to labels and logits.
    
    if type(threshold) == float:
        prediction = (total_logits > threshold).astype(int)
    else:
        prediction = np.vstack([(total_logits[:,subtask_idx] > threshold[subtask_idx]).astype(int) for subtask_idx in range(len(model.subtasks))]).T

    # Calculating metrics
    precision = np.array([metrics.precision_score(prediction[:,idx], total_labels[:,idx]) for idx in range(total_labels.shape[1])])
    recall = np.array([metrics.recall_score(prediction[:,idx], total_labels[:,idx]) for idx in range(total_labels.shape[1])])
    f1 = np.array([metrics.f1_score(prediction[:,idx], total_labels[:,idx]) for idx in range(total_labels.shape[1])])
    confusion_matrix = np.array([metrics.confusion_matrix(prediction[:,idx], total_labels[:,idx],labels=[0,1]).ravel() for idx in range(total_labels.shape[1])])
    # if confusion_matrix.size!=36:
    #     print('not 36')
    classification_report = [metrics.classification_report(prediction[:,idx], total_labels[:,idx], output_dict=True) for idx in range(total_labels.shape[1])]
    
    return precision, recall, f1, prediction, confusion_matrix, classification_report

# prediction script
def make_prediction(model, dataloader, device, threshold=0.5):
    # run model and predict without having "y" label
    # only return the prediction
    
    model.eval()
    dev_logits = []
    for step, batch in enumerate(dataloader):

        input_dict = {"input_ids": batch["input_ids"].to(device),
                      "entity_start_positions": batch["entity_start_positions"].to(device)}
        
        logits, _ = model(**input_dict)
        
        # Post-model subtask information aggregation.
        logits = list(logits.detach().cpu().numpy())
        dev_logits += logits
    
    dev_logits = np.array(dev_logits)
    
    # Assessment on the results according to labels and logits.  
    if type(threshold) == float:
        prediction = (dev_logits > threshold).astype(int)
    else:
        prediction = np.vstack([(dev_logits[:,subtask_idx] > threshold[subtask_idx]).astype(int) for subtask_idx in range(len(model.subtasks))])
    
    
    return prediction


def result_to_tsv(results,model_config, taskname,output_dir):
    # results = loadFromJSONFile(results_file)
    # model_config = loadFromJSONFile(model_config_file)
    # We will save the classifier results and model config for each subtask in this dictionary
    all_subtasks_results_and_model_configs = dict()
    all_task_results_and_model_configs = dict()
    all_task_question_tags = dict()
    tested_tasks = list()
    for key in results:
        if key not in ["best_dev_threshold", "best_dev_F1s", "dev_t_F1_P_Rs"]:
            tested_tasks.append(key)
            results[key]["best_dev_threshold"] = results["best_dev_threshold"][key]
            results[key]["best_dev_F1"] = results["best_dev_F1s"][key]
            results[key]["dev_t_F1_P_Rs"] = results["dev_t_F1_P_Rs"][key]
            all_subtasks_results_and_model_configs[key] = results[key], model_config
    all_task_results_and_model_configs[taskname] = all_subtasks_results_and_model_configs
    all_task_question_tags[taskname] = tested_tasks

    # Read the results for each task and save them in csv file
    # results_tsv_save_file = os.path.join("results", "all_experiments_multitask_bert_entity_classifier_results.tsv")
    # NOTE: After fixing the USER and URL tags

    results_tsv_save_file = os.path.join(output_dir,"result.tsv")
    with open(results_tsv_save_file, "a") as tsv_out:
        writer = csv.writer(tsv_out, delimiter='\t')
        # header = ["Event", "Sub-task",  "model name", "accuracy", "CM", "pos. F1", "dev_threshold", "dev_N",
        #           "dev_F1", "dev_P", "dev_R", "dev_TP", "dev_FP", "dev_FN", "N", "F1", "P", "R", "TP", "FP", "FN"]
        # writer.writerow(hesader)
        for taskname, question_tags in all_task_question_tags.items():
            current_task_results_and_model_configs = all_task_results_and_model_configs[taskname]
            for question_tag in question_tags:
                results_sub, model_config = current_task_results_and_model_configs[question_tag]
                # Extract results_sub
                classification_report = results_sub["Classification Report"]
                positive_f1_classification_report = classification_report['1']['f1-score']
                accuracy = classification_report['accuracy']
                CM = results_sub["CM"]
                # Best threshold and dev F1
                best_dev_threshold = results_sub["best_dev_threshold"]
                dev_t_F1_P_Rs = results_sub["dev_t_F1_P_Rs"]
                best_dev_threshold_index = int(best_dev_threshold * 10) - 1
                # Each entry in dev_t_F1_P_Rs is of the format t, dev_F1, dev_P, dev_R, dev_TP + dev_FN, dev_TP, dev_FP, dev_FN
                t, dev_F1, dev_P, dev_R, dev_N, dev_TP, dev_FP, dev_FN = dev_t_F1_P_Rs[best_dev_threshold_index]
                # Alan's metrics
                F1 = results_sub["F1"]
                P = results_sub["P"]
                R = results_sub["R"]
                TP = results_sub["TP"]
                FP = results_sub["FP"]
                FN = results_sub["FN"]
                N = results_sub["N"]
                # Extract model config
                model_name = model_config["model"]

                row = [taskname, question_tag, model_name, accuracy, CM,
                        positive_f1_classification_report,
                       best_dev_threshold, dev_N, dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN, N, F1, P, R, TP, FP,
                              FN]
                writer.writerow(row)

def post_processing(args, model, valid_dataloader, test_dataloader, device):
    # Save the model name in the model_config file
    model_config = dict()
    results = dict()
    model_config["model"] = "MultiTaskBertForCovidEntityClassification_new"
    model_config["epochs"] = args.n_epochs
    
    # Check different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Evaluate on different thresholds in dev set
    dev_results = [evaluation(model, valid_dataloader, device, t) for t in thresholds]
    
    # Finding out the best threshold using dev set
    # Getting the f1 scores on different thresholds and in every subtasks
    dev_f1_scores = np.array([dev_results[t_idx][2] for t_idx in range(len(thresholds))]) # (thresholds_idx, subtask_idx)
    # Calculating the best thresholds indices on different subtasks
    best_dev_thresholds_idx = np.argmax(dev_f1_scores, axis=0)
    
    # assert best_dev_thresholds_idx.size == len(model.subtasks, "Expected subtask size: "+str(len(model.subtasks))+" with calculated "+str(best_dev_thresholds_idx.size)+".")
    
    best_dev_F1s = {}
    best_dev_thresholds = {}
    dev_subtasks_t_F1_P_Rs = {subtask: list() for subtask in model.subtasks}
    for subtask_idx in range(best_dev_thresholds_idx.size):
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
        for t_idx in range(len(thresholds)):
            dev_P = dev_results[t_idx][0][subtask_idx]
            dev_R = dev_results[t_idx][1][subtask_idx]
            dev_F1 = dev_results[t_idx][2][subtask_idx]
            dev_TN, dev_FP, dev_FN, dev_TP = dev_results[t_idx][4][subtask_idx]
            
            dev_subtasks_t_F1_P_Rs[subtask].append((thresholds[t_idx], dev_F1, dev_P, dev_R, dev_TP + dev_FN, dev_TP, dev_FP, dev_FN)) # Copy-pasted from original code

    results["best_dev_threshold"] = best_dev_thresholds
    results["best_dev_F1s"] = best_dev_F1s
    results["dev_t_F1_P_Rs"] = dev_subtasks_t_F1_P_Rs
    
    # Apply to testset
    # TODO: Squad style and raw style scores are not evaluated here yet.
    
    logging.info("Testing on test dataset")
    # Turn into list
    best_thresholds = [best_dev_thresholds[subtask] for subtask in model.subtasks]
    # Getting test results
    test_result = evaluation(model, test_dataloader, device, best_thresholds)
    
    for subtask_idx in range(len(best_thresholds)):
        subtask = model.subtasks[subtask_idx]
        results[subtask] = {}
        P = test_result[0][subtask_idx]
        R = test_result[1][subtask_idx]
        F1 = test_result[2][subtask_idx]
        TN, FP, FN, TP = test_result[4][subtask_idx]
        classification_report = test_result[5][subtask_idx]
        
        results[subtask]["CM"] = [TN, FP, FN, TP]  # Storing it as list of lists instead of numpy.ndarray
        results[subtask]["Classification Report"] = classification_report
    
    
        results[subtask]["F1"] = F1
        results[subtask]["P"] = P
        results[subtask]["R"] = R
        results[subtask]["TN"] = TN
        results[subtask]["TP"] = TP
        results[subtask]["FP"] = FP
        results[subtask]["FN"] = FN
        N = TP + FN
        results[subtask]["N"] = N
        
        print(results)
        logging.info("New evaluation scores:")
        logging.info(f"F1: {F1}")
        logging.info(f"Precision: {P}")
        logging.info(f"Recall: {R}")
        logging.info(f"True Positive: {TP}")
        logging.info(f"False Positive: {FP}")
        logging.info(f"False Negative: {FN}")
    

    # Save model_config and results
    model_config_file = os.path.join(args.output_dir, "model_config.json")
    results_file = os.path.join(args.output_dir, "results.json")
    logging.info(f"Saving model config at {model_config_file}")
    saveToJSONFile(model_config, model_config_file)
    logging.info(f"Saving results at {results_file}")
    saveToJSONFile(results, results_file)
    return results, model_config
    


def train(event, logging, args):
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
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        tokenizer.add_tokens(["<E>", "</E>", "<URL>", "@USER"])
        tokenizer.save_pretrained(args.output_dir)
        entity_start_token_id = tokenizer.convert_tokens_to_ids(["<E>"])[0]

        # # data split (TODO: a better way to ensure the balance of subtask)
        (train_dataloader, valid_dataloader, test_dataloader, subtask_list) = loadData(
            event, entity_start_token_id, tokenizer,
            batch_size=args.batch_size, train_ratio=0.6, dev_ratio=0.15, shuffle_train_data_flg=True, num_workers=0)

        config = BertConfig.from_pretrained('bert-base-cased')
        config.subtasks = subtask_list
        # # model = MultiTaskBertForCovidEntityClassification.from_pretrained('bert-base-cased', config=config)
        model = MultiTaskBertForCovidEntityClassification_new.from_pretrained('bert-base-cased', config=config)
        model.resize_token_embeddings(len(tokenizer))

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
        tokenizer.save_pretrained(args.output_dir)
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

    if args.retrain:
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
                    precision, recall, f1, prediction, _, _ = evaluation(model, valid_dataloader, device=device)
                    print('\n')
                    print(f"Validation result. Precision",precision, "Recall", recall, "F1", f1)
                    #
                    # # check if this is the best model or not
                    if f1 >= best_score:
                        best_model = copy.deepcopy(model.state_dict())
                        best_score = f1
                        best_epoch = epoch
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
    if args.retrain:
        model.load_state_dict(best_model) ##TODO best model
    model.save_pretrained(args.output_dir)

    # Plot the train loss trajectory in a plot
    train_loss_trajectory_plot_file = os.path.join(args.output_dir, "train_loss_trajectory.png")
    logging.info(f"Saving the Train loss trajectory at {train_loss_trajectory_plot_file}")
    plot_train_loss(epoch_train_loss, train_loss_trajectory_plot_file)

    # running testing
    # TODO: post-processing
    # (1) probing threshold for each subtask
    # (2) save threshold 
    # precision, recall, f1, prediction = evaluation(model, test_dataloader, device=device)
    print("post_processing!")
    results, model_config = post_processing(args, model, valid_dataloader, test_dataloader, device=device)
    print("generating tsv files!")
    result_to_tsv(results, model_config, event, args.output_dir)


def main():
    args = parse_arg()
    make_dir_if_not_exists(args.output_dir)


    results_tsv_save_file = os.path.join(args.output_dir,"result.tsv")
    with open(results_tsv_save_file, "a") as tsv_out:
        writer = csv.writer(tsv_out, delimiter='\t')
        header = ["Event", "Sub-task",  "model name", "accuracy", "CM", "pos. F1", "dev_threshold", "dev_N",
                  "dev_F1", "dev_P", "dev_R", "dev_TP", "dev_FP", "dev_FN", "N", "F1", "P", "R", "TP", "FP", "FN"]
        writer.writerow(header)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logfile =  os.path.join(args.output_dir, "train_output.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

    for event in EVENT_LIST[1:]:
        print("Working in Event:", event)
        train(event, logging, args)

if __name__ == "__main__":
    main()
