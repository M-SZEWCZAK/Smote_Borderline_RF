from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from os_sklearn.ensemble._forest import OSRandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from umap import UMAP
import warnings
warnings.filterwarnings('ignore')


class Comparator:
    def __init__(
            self,
            datasets=[[fetch_datasets()['us_crime'].data, fetch_datasets()['us_crime'].target],
                      [fetch_datasets()['letter_img'].data, fetch_datasets()['letter_img'].target]],
            test_size=0.2,
            oversampling_strategies=['random', 'SMOTE', 'BorderlineSMOTE', 'ADASYN'], # 'random', 'SMOTE', 'BorderlineSMOTE', 'ADASYN'
            metrics=['precision', 'recall', 'f1-score', 'accuracy'], # 'precision', 'recall', 'f1-score', 'accuracy'
            n_trees=100,
            iterations=100,
            print_indices_list=[[0]] + [[]] * 99,
            dataset_name=['us_crime', 'letter_img'],
            mode='both' # 'both', 'bagging', 'augmentation'
    ):
        self.datasets = datasets
        self.test_size = test_size
        self.oversampling_strategies = oversampling_strategies
        self.metrics = metrics
        self.n_trees = n_trees
        self.iterations = iterations
        self.print_indices_list = print_indices_list
        self.dataset_name = dataset_name
        self.mode = mode
        self.visuals = sum(len(indices) for indices in self.print_indices_list if indices)

    def prepare_data(self, dataset):
        return train_test_split(dataset[0], dataset[1], stratify=dataset[1], test_size=self.test_size)

    def compute(self):

        print('=========================================================================')
        print('=========================     START COMPUTING     =======================')
        print('=========================================================================\n')
        print(f'Mode: {self.mode}')
        print(f'Iterations: {self.iterations}')
        print(f'Oversampling strategies: {self.oversampling_strategies}')
        print(f'Metrics: {self.metrics}')
        print(f'Number of trees: {self.n_trees}')
        print(f'Datasets: {self.dataset_name}')

        self.results_bgg = []
        self.results_aug = []
        self.results_rf = []
        self.visualization_data_bgg = []
        self.visualization_data_aug = []
        for i, dataset in enumerate(self.datasets):
            dataset_name = self.dataset_name[i]
            print(f'\n \n + DATASET: {dataset_name}')
            if self.mode == 'both':
                bgg_results, bgg_visualization_data = self.compute_bagging(dataset, dataset_name)
                aug_results, aug_visualization_data = self.compute_augmentation(dataset, dataset_name)
                rf_results = self.compute_baseline(dataset, dataset_name)
                self.results_bgg.append(bgg_results)
                self.results_aug.append(aug_results)
                self.results_rf.append(rf_results)
                self.visualization_data_bgg.append(bgg_visualization_data)
                self.visualization_data_aug.append(aug_visualization_data)
            elif self.mode == 'bagging':
                bgg_results, bgg_visualization_data = self.compute_bagging(dataset, dataset_name)
                rf_results = self.compute_baseline(dataset, dataset_name)
                self.results_rf.append(rf_results)
                self.results_bgg.append(bgg_results)
                self.visualization_data_bgg.append(bgg_visualization_data)
            elif self.mode == 'augmentation':
                aug_results, aug_visualization_data = self.compute_augmentation(dataset, dataset_name)
                rf_results = self.compute_baseline(dataset, dataset_name)
                self.results_rf.append(rf_results)
                self.results_aug.append(aug_results)
                self.visualization_data_aug.append(aug_visualization_data)
            else:
                raise ValueError(f"Mode {self.mode} is not supported. Choose from 'both', 'bagging', or 'augmentation'.")
            
        print('\n=========================================================================')
        print('==================     COMPUTING ENDED SUCCESSFULLY      ================')
        print('=========================================================================\n')

    def _print_progress_bar(self, iteration, total, prefix='', length=30):
        percent = f"{100 * (iteration / float(total)):.1f}"
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% Complete', end='\r')
        if iteration == total:
            print()

    def compute_bagging(self, dataset, dataset_name):
        print('\n-=-=-=-=-=-=   BAGGING   =-=-=-=-=-')
        visualization_data = []
        class_names = np.unique(dataset[1])
        if 'accuracy' not in self.metrics:
            n_metrics = len(self.metrics) * len(class_names)
        else:
            n_metrics = (len(self.metrics) - 1) * len(class_names) + 1
        results = []

        for strategy in self.oversampling_strategies:
            strategy_results = [[] for _ in range(n_metrics)]
            strategy_visualization_data = []
            for j in range(self.iterations):
                self._print_progress_bar(j + 1, self.iterations, prefix=f'{strategy} - bagging')
                if self.print_indices_list[j] is None:
                    indices = None
                else:
                    indices = self.print_indices_list[j]
                
                X_train, X_test, y_train, y_test = self.prepare_data(dataset)
                forest = OSRandomForestClassifier(
                    oversampling_strategy=strategy,
                    print_indices_list=indices,
                    n_estimators=self.n_trees,
                    data_name=dataset_name,
                    iteration=j)
                forest.fit(X_train, y_train)

                if indices is not None:
                    for i in indices:
                        strategy_visualization_data.append(forest.estimators_[i].visualization_pack)

                y_pred = forest.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True, target_names=[str(c) for c in class_names])
                
                idx = 0
                for cls in class_names:
                    for metric in self.metrics:
                        if metric == 'accuracy':
                            continue
                        if metric in report[str(cls)]:
                            strategy_results[idx].append(report[str(cls)][metric])
                        else:
                            raise ValueError(f"Metric {metric} not found in report for class {cls}.")
                        idx += 1
                if 'accuracy' in self.metrics:
                    strategy_results[-1].append(report['accuracy'])

            results.append(strategy_results)
            visualization_data.append(strategy_visualization_data)

        return results, visualization_data

    def compute_augmentation(self, dataset, dataset_name):
        print('\n-=-=-=-=-=-=   AUGMENTATION   =-=-=-=-=-')
        visualization_data = []
        class_names = np.unique(dataset[1])
        if 'accuracy' not in self.metrics:
            n_metrics = len(self.metrics) * len(class_names)
        else:
            n_metrics = (len(self.metrics) - 1) * len(class_names) + 1
        results = []
        
        for strategy in self.oversampling_strategies:
            strategy_results = [[] for _ in range(n_metrics)]
            strategy_visualization_data = []
            for j in range(self.iterations):
                self._print_progress_bar(j + 1, self.iterations, prefix=f'{strategy} - augmentation')

                X_train, X_test, y_train, y_test = self.prepare_data(dataset)

                if strategy == "random":
                    sampler = RandomOverSampler()
                elif strategy == "SMOTE":
                    sampler = SMOTE()
                elif strategy == "BorderlineSMOTE":
                    sampler = BorderlineSMOTE()
                elif strategy == "ADASYN":
                    sampler = ADASYN()
                else:
                    raise ValueError(f"Oversampling strategy {strategy} is not supported.")
                
                forest = RandomForestClassifier(
                    n_estimators=self.n_trees
                )

                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

                forest.fit(X_resampled, y_resampled)
                y_pred = forest.predict(X_test)

                if self.print_indices_list[j] is not None:
                    strategy_visualization_data.append([X_train, X_resampled, y_resampled, dataset_name, strategy, j])

                report = classification_report(y_test, y_pred, output_dict=True, target_names=[str(c) for c in class_names])
                idx = 0
                for cls in class_names:
                    for metric in self.metrics:
                        if metric == 'accuracy':
                            continue
                        if metric in report[str(cls)]:
                            strategy_results[idx].append(report[str(cls)][metric])
                        else:
                            raise ValueError(f"Metric {metric} not found in report for class {cls}.")
                        idx += 1
                if 'accuracy' in self.metrics:
                    strategy_results[-1].append(report['accuracy'])
            
            results.append(strategy_results)
            visualization_data.append(strategy_visualization_data)
                
        return results, visualization_data
    
    def compute_baseline(self, dataset, dataset_name):
        print('\n-=-=-=-=-=-=   BASELINE   =-=-=-=-=-')
        class_names = np.unique(dataset[1])
        if 'accuracy' not in self.metrics:
            n_metrics = len(self.metrics) * len(class_names)
        else:
            n_metrics = (len(self.metrics) - 1) * len(class_names) + 1
        results = [[] for _ in range(n_metrics)]
        for j in range(self.iterations):
            self._print_progress_bar(j + 1, self.iterations, prefix='baseline')

            X_train, X_test, y_train, y_test = self.prepare_data(dataset)
                
            forest = RandomForestClassifier(
                n_estimators=self.n_trees
            )

            forest.fit(X_train, y_train)
            y_pred = forest.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, target_names=[str(c) for c in class_names])
            idx = 0
            for cls in class_names:
                for metric in self.metrics:
                    if metric == 'accuracy':
                        continue
                    if metric in report[str(cls)]:
                        results[idx].append(report[str(cls)][metric])
                    else:
                        raise ValueError(f"Metric {metric} not found in report for class {cls}.")
                    idx += 1
            if 'accuracy' in self.metrics:
                results[-1].append(report['accuracy'])
                
        return results
    
    def print_table(self, results, class_names, metrics):
        idx = 0
        for metric in metrics:
            if metric == 'accuracy':
                vals = np.array(results[-1])
                print(f"\n{'Accuracy':>12} {'min':>12} {'avg':>12} {'max':>12} {'std':>12}")
                print(f"{'':>12} {np.min(vals):12.4f} {np.mean(vals):12.4f} {np.max(vals):12.4f} {np.std(vals):12.4f}")
            else:
                print(f"\n{'':>12} {'min_'+metric:>12} {'avg_'+metric:>12} {'max_'+metric:>12} {'std_'+metric:>12}")
                for cidx, cls in enumerate(class_names):
                    vals = np.array(results[idx])
                    print(f"{str(cls):>12} {np.min(vals):12.4f} {np.mean(vals):12.4f} {np.max(vals):12.4f} {np.std(vals):12.4f}")
                    idx += 1

    def plot_violin_metrics(self, dataset_index):
        dataset_name = self.dataset_name[dataset_index]
        if self.mode == 'both':
            names = [f"{name} bagging" for name in self.oversampling_strategies] + \
                    [f"{name} augmentation" for name in self.oversampling_strategies] + \
                    ['baseline']
            results = self.results_bgg[dataset_index] + self.results_aug[dataset_index] + [self.results_rf[dataset_index]]
        elif self.mode == 'bagging':
            names = [f"{name} bagging" for name in self.oversampling_strategies] + ['baseline']
            results = self.results_bgg[dataset_index] + [self.results_rf[dataset_index]]
        elif self.mode == 'augmentation':
            names = [f"{name} augmentation" for name in self.oversampling_strategies] + ['baseline']
            results = self.results_aug[dataset_index] + [self.results_rf[dataset_index]]

        labels = np.unique(self.datasets[dataset_index][1])
        metrics = self.metrics
        n_classes = len(labels)

        plot_data = []
        plot_labels = []
        plot_methods = []
        plot_metrics = []
        plot_classes = []

        idx_metric = 0
        for m, metric in enumerate(metrics):
            if metric == 'accuracy':
                for i, method in enumerate(names):
                    vals = np.array(results[i][-1])
                    plot_data.extend(vals)
                    plot_labels.extend([method] * len(vals))
                    plot_methods.extend([method] * len(vals))
                    plot_metrics.extend([metric] * len(vals))
                    plot_classes.extend(['accuracy'] * len(vals))
            else:
                for c, cls in enumerate(labels):
                    for i, method in enumerate(names):
                        vals = np.array(results[i][idx_metric])
                        plot_data.extend(vals)
                        plot_labels.extend([method] * len(vals))
                        plot_methods.extend([method] * len(vals))
                        plot_metrics.extend([metric] * len(vals))
                        plot_classes.extend([str(cls)] * len(vals))
                    idx_metric += 1

        df = pd.DataFrame({
            'Value': plot_data,
            'Method': plot_methods,
            'Metric': plot_metrics,
            'Class': plot_classes
        })
        palettes = [["#b39ddb"],
                   ["#ffcc80"],
                   ["#a5d6a7"],
                   ["#90caf9"]]
        plot_idx = 0
        for m, metric in enumerate(metrics):
            if metric == 'accuracy':
                plt.figure(figsize=(8, 6))
                sns.violinplot(
                    data=df[df['Metric'] == 'accuracy'],
                    x='Method', y='Value',
                    palette=palettes[plot_idx//n_classes]
                )
                plt.title(f'Accuracy for {dataset_name}')
                plt.xlabel(None)
                plt.ylabel(None)
                plt.xticks(rotation=70)
                plt.tight_layout()
                plt.show()
                plot_idx += 1
            else:
                for cls in labels:
                    plt.figure(figsize=(8, 6))
                    sns.violinplot(
                    data=df[(df['Metric'] == metric) & (df['Class'] == str(cls))],
                    x='Method', y='Value',
                    palette=palettes[plot_idx//n_classes]
                    )
                    plt.title(f'{metric} (class {cls}) for {dataset_name}')
                    plt.xticks(rotation=70)
                    plt.xlabel(None)
                    plt.ylabel(None)
                    plt.tight_layout()
                    plt.show()
                    plot_idx += 1

                
    def plot_data(self, X_drawn, X_resampled, y_resampled, data_name, oversampling_strategy, iteration, index, type):
        X_drawn_tmp = np.array(X_drawn)
        X_resampled_tmp = np.array(X_resampled)
        y_resampled_tmp = np.array(y_resampled)

        if X_resampled_tmp.shape[1] == 1:
            X_plot = np.hstack([X_resampled_tmp, np.zeros((X_resampled_tmp.shape[0], 1))])
        elif X_resampled_tmp.shape[1] == 2:
            X_plot = X_resampled_tmp
        else:
            umap_model = UMAP(n_components=2)
            X_plot = umap_model.fit_transform(X_resampled_tmp)

        marker = len(X_drawn_tmp)
        classes = np.unique(y_resampled_tmp)

        # Prepare colormaps
        orig_palette = sns.color_palette("magma", len(classes))
        synth_palette = sns.color_palette("viridis", len(classes))

        plt.figure(figsize=(7, 5))

        # Plot synthetic samples
        for idx, cls in enumerate(classes):
            synth_mask = (y_resampled_tmp[marker:] == cls)
            if np.any(synth_mask):
                sns.scatterplot(
                    x=X_plot[marker:, 0][synth_mask],
                    y=X_plot[marker:, 1][synth_mask],
                    color=synth_palette[idx],
                    marker="X",
                    s=100,
                    label=f"Synthetic class {int(cls)}",
                    linewidth=0.4
                )

        # Plot original samples
        for idx, cls in enumerate(classes):
            orig_mask = (y_resampled_tmp[:marker] == cls)
            sns.scatterplot(
                x=X_plot[:marker, 0][orig_mask],
                y=X_plot[:marker, 1][orig_mask],
                color=orig_palette[idx],
                marker="o",
                linewidth=0.4,
                s=40,
                label=f"Original class {int(cls)}"
            )

        plt.title(f"{data_name} data after {oversampling_strategy} {type} (and TSNE), passed to tree no. {index} in forest no. {iteration}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print('')
        for idx, cls in enumerate(classes):
            orig_mask = (y_resampled_tmp[:marker] == cls)
            print(f"Class {int(cls)} has {np.sum(orig_mask)} original samples and {np.sum(y_resampled_tmp[marker:] == cls)} synthetic samples after {oversampling_strategy} oversampling.")
        print('')

    def summary(self):
        print('\n', '=' * 73)
        print('=========================         SUMMARY         =======================')
        print('=' * 73, '\n')

        for i in range(len(self.datasets)):
            print('*' * 5, f' DATASET: {self.dataset_name[i]}\n')
            class_names = np.unique(self.datasets[i][1])

            if self.mode in ['both', 'bagging']:
                for j, strategy in enumerate(self.oversampling_strategies):
                    print('\n', '\n', f'\n+++ {strategy} - bagging')
                    for v in range(self.visuals):
                        self.plot_data(self.visualization_data_bgg[i][j][v][0],
                                       self.visualization_data_bgg[i][j][v][1],
                                       self.visualization_data_bgg[i][j][v][2],
                                       self.visualization_data_bgg[i][j][v][4],
                                       self.visualization_data_bgg[i][j][v][3],
                                       self.visualization_data_bgg[i][j][v][6],
                                       self.visualization_data_bgg[i][j][v][5],
                                       'bagging')
                    results = self.results_bgg[i][j]
                    self.print_table(results, class_names, self.metrics)

            if self.mode in ['both', 'augmentation']:
                for j, strategy in enumerate(self.oversampling_strategies):
                    print(f'\n \n+++ {strategy} - augmentation +++')
                    self.plot_data(self.visualization_data_aug[i][j][0][0],
                                   self.visualization_data_aug[i][j][0][1],
                                   self.visualization_data_aug[i][j][0][2],
                                   self.visualization_data_aug[i][j][0][3],
                                   self.visualization_data_aug[i][j][0][4],
                                   self.visualization_data_aug[i][j][0][5],
                                   '-', 
                                   'augmentation')
                    results = self.results_aug[i][j]
                    self.print_table(results, class_names, self.metrics)

            self.print_table(self.results_rf[i], class_names, self.metrics)

            self.plot_violin_metrics(i)