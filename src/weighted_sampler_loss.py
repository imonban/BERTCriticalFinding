def class_imbalance_sampler(labels):
    class_count = torch.bincount(labels.squeeze())
    class_weighting = 1. / class_count
    sample_weights = class_weighting[labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels))
    return sampler


def clinicalmodeltraining_w_classwights(df):
    df = data_split(df)
    dataset_train, dataset_val = tokenizationBERT(df)
    class_weights = compute_class_weight('balanced',
                                                  np.unique(list(df['Label'])),
                                                  list(df['Label']))
    class_weights = torch.from_numpy(class_weights).float()
    model = BertForSequenceClassification.from_pretrained(
        #"bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        "emilyalsentzer/Bio_ClinicalBERT",
        num_labels = 4, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False # Whether the model returns all hidden-states.
      )

    batch_size = 5
    sampler = class_imbalance_sampler(list(df['Label']))
    #dataloader_train = DataLoader(dataset_train,  ## weighted sampler
    #                          sampler=sampler, 
    #                          batch_size=batch_size)
    dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)
    dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)
    print('Class weights')
    print(class_weights)
    model.cuda()
    lr = 0.00005
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    
    epochs = 50

    #scheduler = get_linear_schedule_with_warmup(optimizer, 
    #                                        num_warmup_steps=0,
    #                                        num_training_steps=len(dataloader_train)*epochs)
    #scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, factor=0.01, patience=2)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       
            #outputs = model(**inputs)
            labels = inputs.get("labels").to(device)
            outputs = model(**inputs)
            logits = outputs.get('logits').to(device)
            #print(logits)
            #print(labels)
            loss = nn.CrossEntropyLoss(weight = class_weights.to(device))
            output = loss(logits.view(-1, 4), labels.view(-1))
            #output = loss(logits.view(-1, 2), labels.view(-1))
            #loss = outputs[0]
            loss_train_total += output
            output.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(output/len(batch))})
         
        torch.save(model.state_dict(), f'finetuned_ClinicalBERT_epoch_{epoch}.model')
        
        tqdm.write(f'\nEpoch {epoch}')
    
        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')
    
        val_loss, predictions, true_vals = evaluate(model, dataloader_validation, device)
        
        val_f1 = f1_score_func(predictions, true_vals)
        #scheduler.step(val_f1)
        scheduler.step()
        writer.add_scalar("Loss/train", loss_train_avg, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Loss/f1_val", val_f1, epoch)
        writer.add_scalar("learning rate", scheduler.get_last_lr()[-1], epoch)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')
    model.save_pretrained("JCAR_cliniclbert")
    writer.flush()
    #_, predictions, true_vals = evaluate(dataloader_validation)
    return model