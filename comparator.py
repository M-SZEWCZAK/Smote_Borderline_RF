from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score
from scipy.stats import gmean
from sklearn.model_selection import train_test_split
from os_sklearn.ensemble._forest import OSRandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from umap import UMAP
import warnings
import os
warnings.filterwarnings('ignore')



class SRComparator():

    def __init__(
            self,
            dataset_name="ecoli",
            oversampling_strategy='BorderlineSMOTE',
            n_trees=100,
            iterations=100,
            n_rates=10,
            ):
        self.dataset_name=dataset_name
        self.oversampling_strategy=oversampling_strategy
        self.n_trees=n_trees
        self.iterations=iterations
        self.metrics=['precision', 'recall']
        self.MIN_RATE=0.5
        self.MAX_RATE=0.7
        self.n_rates=n_rates
        self.generate_comparators()

    def generate_comparators(self):
        self.comparators = []
        self.sampling_rates = np.linspace(self.MIN_RATE, self.MAX_RATE, self.n_rates)
        self.results_folder = f"../sampling_rate_comparison_results/{self.dataset_name}"
        os.makedirs(self.results_folder, exist_ok=True)
        for rate in self.sampling_rates:
            comparator = Comparator(
                dataset_names=[self.dataset_name],
                oversampling_strategies=[self.oversampling_strategy],
                metrics=self.metrics,
                n_trees=self.n_trees,
                iterations=self.iterations,
                sampling_rate=rate,
                results_path=f"../sampling_rate_comparison_results/{self.dataset_name}/sr-{round(rate, 2)}.csv",
                mode='bagging',
            )
            self.comparators.append(comparator)

    def compute(self):
        for comparator in self.comparators:
            comparator.compute(print_var=False, baseline=False)

    def plot_rates(self):
        
        plot = sns.boxplot
        df = self.load_results({'type': 'bagging'})
        labels = [str(c) for c in df[df['class'] != 'all']['class'].unique()]

        palettes = [
            ["#b39ddb"], 
            ["#a5d6a7"],
            ["#90caf9"],
            ["#fff59d"],
            ["#ffcc80"],
            ["#ffab91"],
            ["#b39ddb"], 
            ["#a5d6a7"],
            ["#90caf9"],
            ["#fff59d"],
            ["#ffcc80"],
            ["#ffab91"]
        ]
        
        palette_idx = 0

        for metric in self.metrics:

            plt.figure(figsize=(8,6))
            plot(
                data=df[(df['metric'] == metric) & 
                        (df['class'] == 'all')],
                x='sampling_rate',
                y='value',
                palette=palettes[palette_idx],
                showmeans=True,
                meanprops={"marker": "o",
                            "markerfacecolor": (1, 0, 0, 0),
                            "markeredgecolor": "red",
                            "markersize": 7}
            )
            plt.title(f'{metric.capitalize()} for different sampling rates')
            plt.xlabel(None)
            plt.ylabel(None)
            # plt.xticks(rotation=70)
            plt.tight_layout()
            plt.show()

            if metric in ['precision', 'recall', 'f1-score']:
                for cls in labels:
                    if str(cls) == "-1":
                        cls_name = "majority"
                    elif str(cls) == "1":
                        cls_name = "minority"
                    else:
                        cls_name = str(cls)
                    plt.figure(figsize=(8,6))
                    plot(
                        data=df[(df['metric'] == metric) & 
                                (df['class'] == str(cls))],
                        x='sampling_rate',
                        y='value',
                        palette=palettes[palette_idx],
                        showmeans=True,
                        meanprops={"marker": "o",
                                "markerfacecolor": (1, 0, 0, 0),
                                "markeredgecolor": "red",
                                "markersize": 7}
                    )
                    plt.title(f'{metric.capitalize()} for different sampling rates - {cls_name} Class ')
                    plt.xlabel(None)
                    plt.ylabel(None)
                    # plt.xticks(rotation=70)
                    plt.tight_layout()
                    plt.show()

            palette_idx += 1
            
    def load_results(self, filters=None):
        big_df = pd.DataFrame(columns=['sampling_rate', 'forest_index', 'dataset', 'type', 'strategy', 'iteration', 'class', 'metric', 'value'])
        for rate in self.sampling_rates:
            results_path = os.path.join(self.results_folder, f"sr-{round(rate, 2)}.csv")
            chunks = pd.read_csv(results_path, chunksize=1000)
            filtered_chunks = []
            for chunk in chunks:
                if filters:
                    for key, value in filters.items():
                        chunk = chunk[chunk[key] == value]
                filtered_chunks.append(chunk)
            if filtered_chunks:
                df = pd.concat(filtered_chunks, ignore_index=True)
            else:
                df = pd.DataFrame(columns=[
                    'forest_index', 'dataset', 'type', 'strategy', 'iteration',
                    'class', 'metric', 'value'
                ])
            df['sampling_rate'] = round(rate, 2)
            big_df = pd.concat([big_df, df], ignore_index=True)

        return big_df

class Comparator:
    def __init__(
            self,
            datasets=None,
            dataset_names=['us_crime', 'letter_img'],
            test_size=0.2,
            oversampling_strategies=['random', 'SMOTE', 'BorderlineSMOTE', 'ADASYN'], # 'random', 'SMOTE', 'BorderlineSMOTE', 'ADASYN'
            sampling_rate = 0.5,
            metrics=['precision', 'recall', 'f1-score', 'accuracy', 'auc', 'g-mean'], # 'precision', 'recall', 'f1-score', 'accuracy', 'auc', 'g-mean'
            n_trees=100,
            iterations=100,
            mode='both', # 'both', 'bagging', 'augmentation'
            save_forests=False, # saves only one forest per iteration
            save_all_results=False, # takes lots of resources (save only one forest data per iteration)
            results_path=None
    ):
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.fetch_datasets()
        self.test_size = test_size
        self.oversampling_strategies = oversampling_strategies
        self.sampling_rate = sampling_rate
        self.metrics = metrics
        self.n_trees = n_trees
        self.iterations = iterations
        self.mode = mode
        self.prepare_data()
        self.results_storage = []
        self.bagging_storage = []
        self.augmentation_storage = []
        self.baseline_storage = []
        self.forests_storage = []
        self.forest_counter = 0
        self.save_all_results = save_all_results
        self.save_forests = save_forests
        self.results_path = results_path

        if os.path.exists(self.results_path):
            pd.DataFrame(columns=[
                'forest_index', 'dataset', 'type', 'strategy', 'iteration',
                'class', 'metric', 'value'
            ]).to_csv(self.results_path, index=False)

    def fetch_datasets(self):
        if self.datasets is not None:
            return  # Datasets are already provided

        self.datasets = []
        for dataset_name in self.dataset_names:
            dataset = fetch_datasets()[dataset_name]
            self.datasets.append([dataset.data, dataset.target])


    def prepare_data(self):
        self.labels = []
        for i in range(len(self.datasets)):
            dataset = self.datasets[i]
            X, y = dataset[0], dataset[1]
            labels = np.unique(y)
            # self.datasets[i] = train_test_split(X, y, test_size=self.test_size, stratify=y)
            self.labels.append(labels)


    def get_forest_id(self, type, dataset_name, strategy, iteration):
        df = self.load_results({'type': type, 'dataset': dataset_name, 'strategy': strategy, 'iteration': iteration})
        if not df.empty:
            return df.iloc[0]['forest_index']
        else:
            return None
        

    def get_forest(self, forest_index):
        if forest_index >= self.forest_counter:
            raise ValueError(f"Forest index {forest_index} is out of bounds. Maximum index is {self.forest_counter - 1}.")
        
        data = next((item for item in self.forests_storage if item[0] == forest_index), None)
        if data is None:
            raise ValueError(f"Forest with {forest_index} wasn't saved.")
        return data[1]


    def load_results(self, filters=None):
        chunks = pd.read_csv(self.results_path, chunksize=1000)
        filtered_chunks = []
        for chunk in chunks:
            if filters:
                for key, value in filters.items():
                    chunk = chunk[chunk[key] == value]
            filtered_chunks.append(chunk)
        if filtered_chunks:
            df = pd.concat(filtered_chunks, ignore_index=True)
        else:
            df = pd.DataFrame(columns=[
                'forest_index', 'dataset', 'type', 'strategy', 'iteration',
                'class', 'metric', 'value'
            ])
        return df


    def compute(self, print_var = True, baseline=True):
        if print_var:
            print('=========================================================================')
            print('=========================     START COMPUTING     =======================')
            print('=========================================================================\n')

            print(f'Datasets: {self.dataset_names}')
            print(f'Mode: {self.mode}')
            print(f'Oversampling strategies: {self.oversampling_strategies}')
            print(f'Metrics: {self.metrics}')
            print(f'Iterations: {self.iterations}')
            print(f'Number of trees: {self.n_trees}')

        for i in range(len(self.datasets)):
            dataset_name = self.dataset_names[i]
            labels = self.labels[i]
            data = self.datasets[i]
            print(f'\n \n + DATASET: {dataset_name}')
            if self.mode == 'both':
                self.compute_bagging(data, dataset_name, labels)
                self.compute_augmentation(data, dataset_name, labels)
            elif self.mode == 'bagging':
                self.compute_bagging(data, dataset_name, labels)
            elif self.mode == 'augmentation':
                self.compute_augmentation(data, dataset_name, labels)
            else:
                raise ValueError(f"Mode {self.mode} is not supported. Choose from 'both', 'bagging', or 'augmentation'.")

            if baseline:
                self.compute_baseline(data, dataset_name, labels)
            
        self.save_results()
        if print_var:
            print('\n=========================================================================')
            print('==================     COMPUTING ENDED SUCCESSFULLY      ================')
            print('=========================================================================\n')


    def _print_progress_bar(self, iteration, prefix='', length=30):
        percent = f"{100 * (iteration / float(self.iterations)):.1f}"
        filled_length = int(length * iteration // self.iterations)
        bar = '█' * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% Complete', end='\r', flush=True)
        if iteration == self.iterations:
            print()


    def extract_metrics(self, type, report, dataset_name, strategy, iteration, labels, forest, X_test, y_test, y_pred):

        for cls in labels:
            for metric in ['precision', 'recall', 'f1-score']:
                if metric in self.metrics:
                    self.results_storage.append([
                        self.forest_counter, 
                        dataset_name, 
                        type, 
                        strategy, 
                        iteration, 
                        str(cls), 
                        metric, 
                        report[str(cls)][metric]])
                else:
                    continue
                        
        if 'accuracy' in self.metrics:
            self.results_storage.append([
                self.forest_counter, dataset_name, type, strategy, iteration, 'all', 'accuracy', report['accuracy']])
            
        if 'precision' in self.metrics:
            self.results_storage.append([
                self.forest_counter, dataset_name, type, strategy, iteration, 'all', 'precision', report['macro avg']['precision']])
            
        if 'recall' in self.metrics:
            self.results_storage.append([
                self.forest_counter, dataset_name, type, strategy, iteration, 'all', 'recall', report['macro avg']['recall']])
            
        if 'f1-score' in self.metrics:
            self.results_storage.append([
                self.forest_counter, dataset_name, type, strategy, iteration, 'all', 'f1-score', report['macro avg']['f1-score']])
        
        if 'auc' in self.metrics:
            y_proba = forest.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            self.results_storage.append([
                self.forest_counter, dataset_name, type, strategy, iteration, 'all', 'auc', auc])
                    
        if 'g-mean' in self.metrics:
            reacalls = recall_score(y_test, y_pred, average=None)
            gmean_val = gmean(reacalls)
            self.results_storage.append([
                self.forest_counter, dataset_name, type, strategy, iteration, 'all', 'g-mean', gmean_val])


    def save_results(self):
        pd.DataFrame(self.results_storage, columns=[
            'forest_index', 'dataset', 'type', 'strategy', 'iteration',
            'class', 'metric', 'value'
        ]).to_csv(self.results_path, index=False)


    def compute_bagging(self, data, dataset_name, labels):

        print('\n-=-=-=-=-=-=   BAGGING   =-=-=-=-=-')

        for strategy in self.oversampling_strategies:
            for j in range(self.iterations):

                self._print_progress_bar(j + 1, prefix=f'{strategy} - bagging')
                
                X_train, X_test, y_train, y_test = train_test_split(
                    data[0], data[1], stratify=data[1], test_size=self.test_size)

                forest = OSRandomForestClassifier(
                    oversampling_strategy=strategy,
                    sampling_rate=self.sampling_rate,
                    n_estimators=self.n_trees)
                
                forest.fit(X_train, y_train)

                y_pred = forest.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True, target_names=[str(c) for c in labels])
                self.extract_metrics('bagging', report, dataset_name, strategy, j, labels, forest, X_test, y_test, y_pred)

                if self.save_all_results:
                    if self.save_forests:
                        self.forests_storage.append([self.forest_counter,forest])
                    for tree_idx, tree in enumerate(forest.estimators_):
                        self.bagging_storage.append([self.forest_counter,
                                                    tree.visualization_pack[0],
                                                    tree.visualization_pack[1],
                                                    tree.visualization_pack[2],
                                                    dataset_name,
                                                    strategy, 
                                                    j,
                                                    tree_idx])
                        
                if  j == 0 and not self.save_all_results:
                    if self.save_forests:
                        self.forests_storage.append([self.forest_counter,forest])
                    tree = forest.estimators_[0]
                    self.bagging_storage.append([self.forest_counter,
                                                tree.visualization_pack[0],
                                                tree.visualization_pack[1],
                                                tree.visualization_pack[2],
                                                dataset_name,
                                                strategy, 
                                                j,
                                                0])                                                 

                self.forest_counter += 1
                    

    def compute_augmentation(self, data, dataset_name, labels):

        print('\n-=-=-=-=-=-=   AUGMENTATION   =-=-=-=-=-')
        
        for strategy in self.oversampling_strategies:
            for j in range(self.iterations):

                self._print_progress_bar(j + 1, prefix=f'{strategy} - augmentation')

                X_train, X_test, y_train, y_test = train_test_split(
                    data[0], data[1], stratify=data[1], test_size=self.test_size)

                if strategy == "random":
                    sampler = RandomOverSampler(sampling_strategy=self.sampling_rate)
                elif strategy == "SMOTE":
                    sampler = SMOTE(sampling_strategy=self.sampling_rate)
                elif strategy == "BorderlineSMOTE":
                    sampler = BorderlineSMOTE(sampling_strategy=self.sampling_rate)
                elif strategy == "ADASYN":
                    sampler = ADASYN(sampling_strategy=self.sampling_rate)
                else:
                    raise ValueError(f"Oversampling strategy {strategy} is not supported.")
                
                forest = RandomForestClassifier(
                    n_estimators=self.n_trees
                )

                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                forest.fit(X_resampled, y_resampled)

                y_pred = forest.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True, target_names=[str(c) for c in labels])
                self.extract_metrics('augmentation', report, dataset_name, strategy, j, labels, forest, X_test, y_test, y_pred)

                if self.save_all_results or j == 0:
                    if self.save_forests:
                        self.forests_storage.append(forest)

                    self.augmentation_storage.append([
                        self.forest_counter, X_train, X_resampled, y_resampled, dataset_name, strategy, j])
                    
                self.forest_counter += 1
    

    def compute_baseline(self, data, dataset_name, labels):

        print('\n-=-=-=-=-=-=   BASELINE   =-=-=-=-=-')

        for j in range(self.iterations):

            self._print_progress_bar(j + 1, prefix='baseline')

            X_train, X_test, y_train, y_test = train_test_split(
                data[0], data[1], stratify=data[1], test_size=self.test_size)
            forest = RandomForestClassifier(
                n_estimators=self.n_trees
            )
            forest.fit(X_train, y_train)

            y_pred = forest.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, target_names=[str(c) for c in labels])

            self.extract_metrics('baseline', report, dataset_name, '-', j, labels, forest, X_test, y_test, y_pred)

            if self.save_all_results or j == 0:
                if self.save_forests:
                    self.forests_storage.append(forest)

                self.baseline_storage.append([
                    self.forest_counter, X_train, X_train, y_train, dataset_name])
                
            self.forest_counter += 1
    

    def print_table(self, type, dataset, strategy, labels):
        df = self.load_results({'type': type, 'dataset': dataset, 'strategy': strategy})
        for metric in self.metrics:
            if metric in ['accuracy', 'auc', 'g-mean']:
                continue
            else:
                print(f"\n{'':>12} {'min_'+metric:>12} {'avg_'+metric:>12} {'max_'+metric:>12} {'std_'+metric:>12}")
                for cls in labels:
                    vals = df[
                        (df['class'] == str(cls)) & 
                        (df['type'] == type) &
                        (df['metric'] == metric) & 
                        (df['strategy'] == strategy) & 
                        (df['dataset'] == dataset)]['value'].astype(float)
                    if not vals.empty:
                        print(f"{str(cls):>12} {vals.min():12.4f} {vals.mean():12.4f} {vals.max():12.4f} {vals.std():12.4f}")
                    else:
                        print(f"{str(cls):>12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
        
        for metric in self.metrics:
            vals = df[
                (df['metric'] == metric) &
                (df['class'] == 'all') &
                (df['type'] == type) & 
                (df['strategy'] == strategy) & 
                (df['dataset'] == dataset)]['value'].astype(float)
            print(f"\n{'':>12} {'min':>12} {'avg':>12} {'max':>12} {'std':>12}")
            print(f"{metric+' :':>12} {vals.min():12.4f} {vals.mean():12.4f} {vals.max():12.4f} {vals.std():12.4f}")


    def plot_metrics(self, dataset, labels, plot_classes, plot_type):

        if plot_type == 'box':
            plot = sns.boxplot
        elif plot_type == 'violin':
            plot = sns.violinplot
        else:
            raise ValueError("plot_type must be either 'box' or 'violin'.")
        
        df = self.load_results({'dataset': dataset})

        palettes = [
            ["#b39ddb"], 
            ["#a5d6a7"],
            ["#90caf9"],
            ["#fff59d"],
            ["#ef9a9a"],
            ["#ffcc80"]
        ]
        
        palette_idx = 0

        for metric in self.metrics:

            plt.figure(figsize=(8,6))
            plot(
                data=df[(df['metric'] == metric) & 
                        (df['dataset'] == dataset) &
                        (df['class'] == 'all')],
                x=df['type'] + '  ' + df['strategy'],
                y='value',
                palette=palettes[palette_idx],
                showmeans=True,
                meanprops={"marker": "o",
                            "markerfacecolor": (1, 0, 0, 0),
                            "markeredgecolor": "red",
                            "markersize": 7}
            )
            plt.title(f'{metric.capitalize()} for {dataset}')
            plt.xlabel(None)
            plt.ylabel(None)
            plt.xticks(rotation=70)
            plt.tight_layout()
            plt.show()

            if metric in ['precision', 'recall', 'f1-score'] and plot_classes:
                for cls in labels:
                    if str(cls) == "-1":
                        cls_name = "majority"
                    elif str(cls) == "1":
                        cls_name = "minority"
                    else:
                        cls_name = str(cls)
                    plt.figure(figsize=(8,6))
                    plot(
                        data=df[(df['metric'] == metric) & 
                                (df['dataset'] == dataset) &
                                (df['class'] == str(cls))],
                        x=df['type'] + '  ' + df['strategy'],
                        y='value',
                        palette=palettes[palette_idx],
                        showmeans=True,
                        meanprops={"marker": "o",
                                   "markerfacecolor": (1, 0, 0, 0),
                                   "markeredgecolor": "red",
                                   "markersize": 7}
                    )
                    plt.title(f'{metric.capitalize()} for {dataset} - {cls_name} Class ')
                    plt.xlabel(None)
                    plt.ylabel(None)
                    plt.xticks(rotation=70)
                    plt.tight_layout()
                    plt.show()

            palette_idx += 1

                
    def plot_data(self, forest_idx, tree_idx=None):

        if forest_idx >= self.forest_counter:
            raise ValueError(f"Forest index {forest_idx} is out of bounds. Maximum index is {self.forest_counter - 1}.")
        
        bagging_ids = [item[0] for item in self.bagging_storage] if self.bagging_storage else []
        augmentation_ids = [item[0] for item in self.augmentation_storage] if self.augmentation_storage else []

        if forest_idx in bagging_ids:
            type = 'bagging'
        elif forest_idx in augmentation_ids:
            type = 'augmentation'
        else:
            type = 'baseline'

        if type == 'bagging':
            if tree_idx is None:
                raise ValueError("For bagging tree_idx must be provided.")
            
            visualization_data = next((item for item in self.bagging_storage if item[0] == forest_idx and item[7] == tree_idx), None)
            _, X_drawn, X_resampled, y_resampled, dataset, strategy, iteration, _ = visualization_data
            title = f"(TSNE) {dataset} after {strategy} bagging (iteration: {iteration}, tree: {tree_idx})"

        elif type == 'augmentation':
            visualization_data = next((item for item in self.augmentation_storage if item[0] == forest_idx), None)
            _, X_drawn, X_resampled, y_resampled, dataset, strategy, iteration = visualization_data
            title = f"(TSNE) {dataset} after {strategy} augmentation (iteration: {iteration})"
        
        else:
            visualization_data = next((item for item in self.baseline_storage if item[0] == forest_idx), None)
            if visualization_data is None:
                raise ValueError(f"No baseline data for forest index - {forest_idx}.")
            _, X_drawn, X_resampled, y_resampled, dataset = visualization_data
            title = f"(TSNE) {dataset}"

        dataset_idx = self.dataset_names.index(dataset)
        labels = self.labels[dataset_idx]

        # Prepare data for plotting
        if X_resampled.shape[1] == 1:
            X_plot = np.hstack([X_resampled, np.zeros((X_resampled.shape[0], 1))])
        elif X_resampled.shape[1] == 2:
            X_plot = X_resampled
        else:
            umap_model = UMAP(n_components=2, random_state=42)
            umap_model.fit(X_drawn)
            X_plot = umap_model.transform(X_resampled)
            # X_plot = X_resampled[:, :2]

        marker = len(X_drawn)

        orig_palette = sns.color_palette("magma", len(labels))
        synth_palette = sns.color_palette("viridis", len(labels))

        plt.figure(figsize=(8, 6))

        classes = np.unique(y_resampled)

        # Plot synthetic samples
        for idx, cls in enumerate(classes):
            synth_mask = (y_resampled[marker:] == cls)
            if np.any(synth_mask):
                sns.scatterplot(
                    x=X_plot[marker:, 0][synth_mask],
                    y=X_plot[marker:, 1][synth_mask],
                    color=synth_palette[idx],
                    marker="X",
                    s=100,
                    label=f"synthetic class:  {int(cls)}",
                    linewidth=0.4
                )

        # Plot original samples
        for idx, cls in enumerate(classes):
            orig_mask = np.array(y_resampled[:marker] == cls)
            sns.scatterplot(
                x=X_plot[:marker, 0][orig_mask],
                y=X_plot[:marker, 1][orig_mask],
                color=orig_palette[idx],
                marker="o",
                linewidth=0.4,
                s=40,
                label=f"class:  {int(cls)}"
            )

        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print('')
        for idx, cls in enumerate(classes):
            orig_mask = (y_resampled[:marker] == cls)
            if type == 'bagging':
                print(f"Class {int(cls)} has {np.sum(orig_mask)} original samples and {np.sum(y_resampled[marker:] == cls)} synthetic samples after {strategy} bagging.")
            elif type == 'augmentation':
                print(f"Class {int(cls)} has {np.sum(orig_mask)} original points and {np.sum(y_resampled[marker:] == cls)} synthetic points after {strategy} augmentation.")
            else:
                print(f"Class {int(cls)} has {np.sum(orig_mask)} points.")
        print('')


    def plot_set(self, dataset_name):
        df = self.load_results({'type': 'baseline', 'dataset': dataset_name, 'strategy': '-'})
        forest_idx = df['forest_index'].values[0]
        self.plot_data(forest_idx)


    def summary(self, plot_data=False, plot_metrix=True, separate_plots_for_classes=True, plot_type='box'):
        print('\n', '=' * 73)
        print('=========================         SUMMARY         =======================')
        print('=' * 73, '\n')

        for i in range(len(self.dataset_names)):
            print('*' * 5, f' DATASET: {self.dataset_names[i]}\n')
            print('*' * 5, f' oversampling rate: {self.sampling_rate}\n')
            labels = self.labels[i]
            dataset_name = self.dataset_names[i]

            if plot_data:
                self.plot_set(dataset_name)

            if self.mode == 'both':
                for strategy in self.oversampling_strategies:
                    print('\n', '\n', f'\n+++ {strategy} - bagging')
                    if plot_data:
                        idx = self.get_forest_id('bagging', dataset_name, strategy, 0)
                        self.plot_data(forest_idx=idx, tree_idx=0)
                    self.print_table('bagging', dataset_name, strategy, labels)

                for strategy in self.oversampling_strategies:
                    print(f'\n \n+++ {strategy} - augmentation +++')
                    if plot_data:
                        idx = self.get_forest_id('augmentation', dataset_name, strategy, 0)
                        print(f'Forest index: {idx}')
                        self.plot_data(forest_idx=idx)
                    self.print_table('augmentation', dataset_name, strategy, labels)

            elif self.mode == 'bagging':
                for strategy in self.oversampling_strategies:
                    print('\n', '\n', f'\n+++ {strategy} - bagging')
                    if plot_data:
                        idx = self.get_forest_id('bagging', dataset_name, strategy, 0)
                        self.plot_data(forest_idx=idx, tree_idx=0)
                    self.print_table('bagging', dataset_name, strategy, labels)

            elif self.mode == 'augmentation':
                for strategy in self.oversampling_strategies:
                    print(f'\n \n+++ {strategy} - augmentation +++')
                    if plot_data:
                        idx = self.get_forest_id('augmentation', dataset_name, strategy, 0)
                        print(f'Forest index: {idx}')
                        self.plot_data(forest_idx=idx)
                    self.print_table('augmentation', dataset_name, strategy, labels)

            print(f'\n \n+++ baseline +++')
            self.print_table('baseline', dataset_name, '-', labels)

            if plot_metrix:
                self.plot_metrics(dataset_name, labels, plot_classes=separate_plots_for_classes, plot_type=plot_type)