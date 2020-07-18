from __future__ import absolute_import
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

class AUROC_Callback(Callback):
    def __init__(self, generator, workers=1):
        super().__init__()
        self.generator = generator
        self.workers = workers

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.generator, workers=self.workers)
        y_true= self.generator.labels
        meanAUROC = roc_auc_score(y_true,y_pred)
        print(' - mAUROC:', meanAUROC)

class MultipleClassAUROC(Callback):
    '''
    Sample Usage: 
    auroc = MultipleClassAUROC(
            sequence=validation_sequence,
            class_names=class_names,
            weights_path=output_weights_path,
            stats=training_stats,
            workers=generator_workers,
        ) 
    '''
    def __init__(self, sequence, class_names, weights_path, stats=None, workers=1):
        super(Callback, self).__init__()
        self.sequence = sequence
        self.workers = workers
        self.class_names = class_names
        self.weights_path = weights_path
        self.best_weights_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_{}".format(os.path.split(weights_path)[1]),
        )
        self.best_auroc_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_auroc.log",
        )
        self.stats_output_path = os.path.join(
            os.path.split(weights_path)[0],
            ".training_stats.json"
        )

        # for resuming previous training
        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_auroc": 0}

        # aurocs log
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the average AUROC and save the best model weights according
        to this metric.

        """
        print("\n*********************************")
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        print("current learning rate: {}".format(self.stats['lr']))

        """
        y_hat shape: (#samples, len(class_names))
        y: [(#samples, 1), (#samples, 1) ... (#samples, 1)]
        """
        y_hat = self.model.predict(self.sequence, workers=self.workers)
        y = self.sequence.get_y_true()

        print("*** epoch#{} dev auroc ***".format(epoch + 1))
        current_auroc = []
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
            except ValueError:
                score = 0
            self.aurocs[self.class_names[i]].append(score)
            current_auroc.append(score)
            print("{}. {}: {}".foramt(i+1,self.class_names[i],score))
        print("*********************************")

        # customize your multiple class metrics here
        mean_auroc = np.mean(current_auroc)
        print("mean auroc: {}".format(mean_auroc))
        if mean_auroc > self.stats["best_mean_auroc"]:
            print("update best auroc from {} to {}".format(self.stats['best_mean_auroc'],mean_auroc))

            # 1. copy best model
            shutil.copy(self.weights_path, self.best_weights_path)

            # 2. update log file
            print("update log file: {}".format(self.best_auroc_log_path))
            with open(self.best_auroc_log_path, "a") as f:
                f.write("(epoch#{}) auroc: {}, lr: {}\n".format(epoch + 1,mean_auroc,self.stats['lr']))

            # 3. write stats output, this is used for resuming the training
            with open(self.stats_output_path, 'w') as f:
                json.dump(self.stats, f)

            print("update model file: {} -> {}".format(self.weights_path, self.best_weights_path))
            self.stats["best_mean_auroc"] = mean_auroc
            print("*********************************")
        return

