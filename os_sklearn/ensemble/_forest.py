import numpy as np
from sklearn.utils._param_validation import StrOptions
from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def _unwrap_data(X, y, sample_weight):
    if sample_weight is None:
        return X, y

    return np.repeat(X, sample_weight.astype(int), axis=0), np.repeat(y, sample_weight.astype(int), axis=0)

class OSRandomForestClassifier(ForestClassifier):

    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,
        **DecisionTreeClassifier._parameter_constraints,
        "class_weight": [
            StrOptions({"balanced_subsample", "balanced"}),
            dict,
            list,
            None,
        ],
    }
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        oversampling_strategy="random",
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super().__init__(
            estimator=OSDecisionTreeClassifier(
                oversampling_strategy=oversampling_strategy
            ),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.monotonic_cst = monotonic_cst
        self.ccp_alpha = ccp_alpha
        self.oversampling_strategy = oversampling_strategy

class OSDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(
        self,
        *,
        oversampling_strategy="random",
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
        )
        self.oversampling_strategy = oversampling_strategy

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
    ):
        if self.oversampling_strategy == "random":
            sampler = RandomOverSampler()
        elif self.oversampling_strategy == "SMOTE":
            sampler = SMOTE()
        elif  self.oversampling_strategy == "BorderlineSMOTE":
            sampler = BorderlineSMOTE()
        elif self.oversampling_strategy == "ADASYN":
            sampler = ADASYN()
        else:
            raise ValueError(
                f"Oversampling strategy {self.oversampling_strategy} is not supported."
            )

        X_drawn, y_drawn = _unwrap_data(X, y, sample_weight)
        X_resampled, y_resampled = sampler.fit_resample(X_drawn, y_drawn)

        sample_weight = [1] * len(X_drawn) + [0.5] * (len(X_resampled) - len(X_drawn))

        self.visualization_pack = [np.array(X_drawn), np.array(X_resampled), np.array(y_resampled)]

        return super()._fit(
            X_resampled,
            y_resampled,
            sample_weight=sample_weight,
            check_input=check_input,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
        )