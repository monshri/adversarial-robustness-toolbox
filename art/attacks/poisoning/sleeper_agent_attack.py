from art.attacks.poisoning.gradient_matching_model_retraining import GradientMatchingAttack


class SleeperAgentAttack(GradientMatchingAttack):
    def poison(
        self, x_trigger: np.ndarray, y_trigger: np.ndarray, x_train: np.ndarray, y_train: np.ndarray
    ,patch,patching_strategy) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes a portion of poisoned samples from x_train to make a model classify x_target
        as y_target by matching the gradients.

        :param x_trigger: A list of samples to use as triggers.
        :param y_trigger: A list of target classes to classify the triggers into.
        :param x_train: A list of training data to poison a portion of.
        :param y_train: A list of labels for x_train.
        :return: A list of poisoned samples, and y_train.
        """
        from art.estimators.classification.pytorch import PyTorchClassifier
        from art.estimators.classification.tensorflow import TensorFlowV2Classifier

        if isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            poisoner = self.__poison__tensorflow
            finish_poisoning = self.__finish_poison_tensorflow
        elif isinstance(self.substitute_classifier, PyTorchClassifier):
            poisoner = self.__poison__pytorch
            finish_poisoning = self.__finish_poison_pytorch
        else:
            raise NotImplementedError(
                "GradientMatchingAttack is currently implemented only for Tensorflow V2 and Pytorch."
            )

        # Choose samples to poison.
        x_train = np.copy(x_train)
        y_train = np.copy(y_train)
        if len(np.shape(y_trigger)) == 2:  # dense labels
            classes_target = set(np.argmax(y_trigger, axis=-1))
        else:  # sparse labels
            classes_target = set(y_trigger)  
        num_poison_samples = int(self.percent_poison * len(x_train))

        # Try poisoning num_trials times and choose the best one.
        best_B = np.finfo(np.float32).max  # pylint: disable=C0103
        best_x_poisoned = None
        best_indices_poison = None

        if len(np.shape(y_train)) == 2:
            y_train_classes = np.argmax(y_train, axis=-1)
        else:
            y_train_classes = y_train
        x_trigger = self.apply_patching(patch,x_trigger,patching_strategy=patching_strategy)    
        x_send = np.copy(x_train)
        indices_poison = self.select_poison_indices(self.substitute_classifier,x_send,y_train_classes,classes_target,self.percent_poison)
        
        for _ in trange(self.max_trials):
            x_poison = x_train[indices_poison]
            y_poison = y_train[indices_poison]
            self.__initialize_poison(x_trigger, y_trigger, x_poison, y_poison)
            x_poisoned, B_ = poisoner(x_poison,y_poison,x_train,y_train,indices_poison)  # pylint: disable=C0103
            finish_poisoning()
            B_ = np.mean(B_)  # Averaging B losses from multiple batches.  # pylint: disable=C0103
            print(B_)
        x_train[best_indices_poison] = best_x_poisoned
        return x_train, y_train  # y_train has not been modified.
        
    def model_retraining(self,x_train,y_train,epochs):
        x_train = x_train.astype(np.float32)
        check_train = self.substitute_classifier.model.training 
        self.substitute_classifier.fit(x_train,y_train,epochs=epochs)
        self.substitute_classifier.model.training = check_train
    
    def __poison__pytorch(self, x_poison: np.ndarray, y_poison: np.ndarray,x_train,y_train,indices_poison) -> Tuple[Any, Any]:
        """
        Optimize the poison by matching the gradient within the perturbation budget.

        :param x_poison: List of samples to poison.
        :param y_poison: List of the labels for x_poison.
        :return: A pair of poisoned samples, B-score (cosine similarity of the gradients).
        """

        import torch  # lgtm [py/import-and-import-from]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        class PoisonDataset(torch.utils.data.Dataset):
            """
            Iterator for a dataset to poison.
            """

            def __init__(self, x: np.ndarray, y: np.ndarray):
                self.len = x.shape[0]
                self.x = torch.as_tensor(x, dtype=torch.float)
                self.y = torch.as_tensor(y)

            def __getitem__(self, index):
                return self.x[index], torch.as_tensor([index]), self.y[index]

            def __len__(self):
                return self.len

        trainloader = torch.utils.data.DataLoader(
            PoisonDataset(x_poison, y_poison), batch_size=self.batch_size, shuffle=False, num_workers=1
        )

        epoch_iterator = trange(self.max_epochs) if self.verbose > 0 else range(self.max_epochs)
        for epoch in epoch_iterator:
            batch_iterator = tqdm(trainloader) if isinstance(self.verbose, int) and self.verbose >= 2 else trainloader
            sum_loss = 0
            count = 0
            for x, indices, y in batch_iterator:
                x = x.to(device)
                y = y.to(device)
                indices = indices.to(device)
                self.backdoor_model.zero_grad()
                loss, poisoned_samples = self.backdoor_model(x, indices, y, self.grad_ws_norm)
                loss.backward()
                self.backdoor_model.noise_embedding.embedding_layer.weight.grad.sign_()
                self.optimizer.step()
                sum_loss += loss.clone().cpu().detach().numpy()
                count += 1
            if self.verbose > 0:
                epoch_iterator.set_postfix(loss=sum_loss / count)
            self.lr_schedule.step()
            if epoch in [125,250,375,499]:
                x_train[indices_poison]=x_poison
                self.model_retraining(x_train,y_train,epochs=40)

        B_sum = 0  # pylint: disable=C0103
        count = 0
        all_poisoned_samples = []
        self.backdoor_model.eval()
        poisonloader = torch.utils.data.DataLoader(
            PoisonDataset(x_poison, y_poison), batch_size=self.batch_size, shuffle=False, num_workers=1
        )
        for x, indices, y in poisonloader:
            x = x.to(device)
            y = y.to(device)
            indices = indices.to(device)
            B, poisoned_samples = self.backdoor_model(x, indices, y, self.grad_ws_norm)  # pylint: disable=C0103
            all_poisoned_samples.append(poisoned_samples.detach().cpu().numpy())
            B_sum += B.detach().cpu().numpy()  # pylint: disable=C0103
            count += 1
        return np.concatenate(all_poisoned_samples, axis=0), B_sum / count


    def select_poison_indices(self,classifier, x,y_train_classes, classes_target, poison_pp):
        # CHECK IF THE MODEL IS TRAIN/EVAL?????
        import torch
        get_target_indices = []
        i=0
        for y in y_train_classes:
            if y in classes_target:
                get_target_indices.append(i)
            i+=1    
        poison_num = int(poison_pp*len(get_target_indices))        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        grad_norms = []
        criterion = torch.nn.CrossEntropyLoss()
        model = classifier.model
        model.eval()
        differentiable_params = [p for p in classifier.model.parameters() if p.requires_grad]
        for idx in get_target_indices:
            image = torch.tensor(x[idx]).to(device).type(torch.cuda.FloatTensor)  # this will get image and labels from target class only
            label = torch.tensor(y_train_classes[idx]).to(device)
            loss = criterion(model(image.unsqueeze(0)), label.unsqueeze(0))
            gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            grad_norms.append(grad_norm.sqrt())  

        indices = sorted(range(len(grad_norms)), key=lambda k: grad_norms[k])
        indices = indices[-poison_num:]
        result_indices = np.array(get_target_indices)[indices].tolist()
        return result_indices # this will get only indices for target class

    # Function to apply patching according to patching strategy
    def apply_patching(patch,x_trigger,patching_strategy):
        if patching_strategy=='random':
            for x in x_trigger:
                x_cord = random.randint(0, 24)
                y_cord = random.randint(0, 24)
                x[:,x_cord:x_cord+8,y_cord:y_cord+8]=patch
        else:
            x_trigger[:,:,-8:,-8:] = patch
        return x_trigger    
