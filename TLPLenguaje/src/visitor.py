import sys
import os
sys.path.append(os.path.dirname(__file__))

from parser.TrezParser import TrezParser
from parser.TrezParserVisitor import TrezParserVisitor as AntlrTrezVisitor
import math_utilsdoz
from errors import TrezRuntimeError, UndefinedSymbolError
from lib.iodoz.iodoz import read_file_doz, write_file_doz
from lib.structsdoz.structsdoz import TrezQueue, TrezStack, make_queue, make_stack
from lib.inspectdoz.inspectdoz import spy as inspectdoz_spy, shape as inspectdoz_shape
from lib.optimdoz.optimdoz import sgd as optimdoz_sgd, adam as optimdoz_adam, zeros_like as optimdoz_zeros
from lib.lossesdoz.lossdoz import (
    cross_entropy as lossdoz_ce,
    cross_entropy_grad as lossdoz_ce_grad,
    bce_loss as lossdoz_bce,
    bce_grad as lossdoz_bce_grad,
    contrastive_loss as lossdoz_contrastive,
    contrastive_grad as lossdoz_contrastive_grad,
    triplet_loss as lossdoz_triplet,
    triplet_grad as lossdoz_triplet_grad,
    kl_divergence as lossdoz_kl,
    kl_grad as lossdoz_kl_grad,
    huber_loss as lossdoz_huber,
    huber_grad as lossdoz_huber_grad,
)
from autograd import (
    Tensor as AutogradTensor,
    tensor as autodoz_tensor,
    tensor_zeros as autodoz_zeros,
    tensor_ones as autodoz_ones,
)
from lib.metricsdoz.metricsdoz import (
    accuracy as metricsdoz_accuracy,
    precision as metricsdoz_precision,
    recall as metricsdoz_recall,
    f1_score as metricsdoz_f1,
    confusion_matrix as metricsdoz_confusion,
    rmse as metricsdoz_rmse,
    mae as metricsdoz_mae,
    r2_score as metricsdoz_r2,
)
from lib.mldoz.linear_regression import (
    linreg_fit as mldoz_linreg_fit,
    linreg_predict as mldoz_linreg_predict,
)
from lib.mldoz.logistic_regression import (
    logreg_fit as mldoz_logreg_fit,
    logreg_predict as mldoz_logreg_predict,
    logreg_predict_proba as mldoz_logreg_proba,
)
from lib.mldoz.svm import (
    svm_fit as mldoz_svm_fit,
    svm_predict as mldoz_svm_predict,
)
from lib.mldoz.tree import (
    tree_clf_fit as mldoz_tree_clf_fit,
    tree_reg_fit as mldoz_tree_reg_fit,
    tree_predict as mldoz_tree_predict,
)
from lib.mldoz.knn import (
    knn_fit as mldoz_knn_fit,
    knn_predict as mldoz_knn_predict,
    knn_predict_clf as mldoz_knn_clf,
    knn_predict_reg as mldoz_knn_reg,
)
from lib.mldoz.kmeans import (
    kmeans_fit as mldoz_kmeans_fit,
    kmeans_predict as mldoz_kmeans_predict,
)
from lib.nndoz.nndoz import (
    linear_init as nndoz_linear_init,
    linear_forward as nndoz_linear_forward,
    linear_backward as nndoz_linear_backward,
    relu_forward as nndoz_relu_forward,
    relu_backward as nndoz_relu_backward,
    sigmoid_forward as nndoz_sigmoid_forward,
    sigmoid_backward as nndoz_sigmoid_backward,
    tanh_forward as nndoz_tanh_forward,
    tanh_backward as nndoz_tanh_backward,
    softmax_forward as nndoz_softmax,
    sequential_forward as nndoz_sequential,
    get_params as nndoz_params,
    get_param_count as nndoz_param_count,
    batchnorm_init as nndoz_bn_init,
    batchnorm_forward as nndoz_bn_forward,
    batchnorm_backward as nndoz_bn_backward,
    dropout_forward as nndoz_dropout_forward,
    dropout_backward as nndoz_dropout_backward,
    shared_forward as nndoz_shared_forward,
    shared_backward as nndoz_shared_backward,
    embedding_init as nndoz_embedding_init,
    embedding_forward as nndoz_embedding_forward,
    embedding_backward as nndoz_embedding_backward,
    l2_distance as nndoz_l2_dist,
    cosine_similarity as nndoz_cosine_sim,
    perceptron_fit as nndoz_perceptron_fit,
    perceptron_predict as nndoz_perceptron_predict,
    mlp_init as nndoz_mlp_init,
    mlp_train as nndoz_mlp_train,
    mlp_predict as nndoz_mlp_predict,
)
from lib.preprocessdoz.preprocessdoz import (
    pca_fit as preproc_pca_fit,
    pca_transform as preproc_pca_transform,
    pca_fit_transform as preproc_pca_fit_transform,
    pca_explained_ratio as preproc_pca_ratio,
    impute_mean as preproc_impute_mean,
    impute_median as preproc_impute_median,
    impute_constant as preproc_impute_const,
    drop_nulls as preproc_drop_nulls,
    null_count as preproc_null_count,
    write_xlsx as preproc_write_xlsx,
    logreg_multi_fit as preproc_logreg_multi_fit,
    logreg_multi_predict as preproc_logreg_multi_predict,
    logreg_multi_proba as preproc_logreg_multi_proba,
    feature_importance as preproc_feat_importance,
    validation_curve as preproc_val_curve,
)
from lib.datadoz.datadoz import (
    from_lists as datadoz_from_lists,
    make_loader as datadoz_loader,
    get_batches as datadoz_batches,
    train_test_split as datadoz_split,
    read_csv as datadoz_read_csv,
    read_xlsx as datadoz_read_xlsx,
    read_tsv as datadoz_read_tsv,
    read_json as datadoz_read_json,
    write_csv as datadoz_write_csv,
    write_json as datadoz_write_json,
    rename_col as datadoz_rename_col,
    drop_col as datadoz_drop_col,
    add_col as datadoz_add_col,
    filter_rows as datadoz_filter_rows,
    get_column as datadoz_col,
    get_row as datadoz_row,
    num_rows as datadoz_nrows,
    num_cols as datadoz_ncols,
    column_names as datadoz_col_names,
)
from lib.randomdoz.randomdoz import (
    seed as randomdoz_seed,
    random as randomdoz_random,
    uniform as randomdoz_uniform,
    randint as randomdoz_randint,
    choice as randomdoz_choice,
    sample as randomdoz_sample,
    shuffle as randomdoz_shuffle,
    gauss as randomdoz_gauss,
)
from lib.sklearndoz.linear_models import (
    LinearRegression as sk_LinearRegression,
    Ridge as sk_Ridge,
    Lasso as sk_Lasso,
    ElasticNet as sk_ElasticNet,
    LogisticRegression as sk_LogisticRegression,
    SGDClassifier as sk_SGDClassifier,
    SGDRegressor as sk_SGDRegressor,
)
from lib.sklearndoz.tree import (
    DecisionTreeClassifier as sk_DTC,
    DecisionTreeRegressor as sk_DTR,
)
from lib.sklearndoz.ensemble import (
    RandomForestClassifier as sk_RFC,
    RandomForestRegressor as sk_RFR,
    GradientBoostingClassifier as sk_GBC,
    GradientBoostingRegressor as sk_GBR,
    AdaBoostClassifier as sk_AdaBoost,
    BaggingClassifier as sk_Bagging,
)
from lib.sklearndoz.neighbors import (
    KNeighborsClassifier as sk_KNC,
    KNeighborsRegressor as sk_KNR,
    NearestCentroid as sk_NearestCentroid,
)
from lib.sklearndoz.naive_bayes import (
    GaussianNB as sk_GaussianNB,
    MultinomialNB as sk_MultinomialNB,
    BernoulliNB as sk_BernoulliNB,
)
from lib.sklearndoz.svm import (
    SVC as sk_SVC,
    LinearSVC as sk_LinearSVC,
    SVR as sk_SVR,
    LinearSVR as sk_LinearSVR,
)
from lib.sklearndoz.cluster import (
    KMeans as sk_KMeans,
    DBSCAN as sk_DBSCAN,
    AgglomerativeClustering as sk_Agglomerative,
    GaussianMixture as sk_GMM,
)
from lib.sklearndoz.decomposition import (
    PCA as sk_PCA,
    TruncatedSVD as sk_TruncatedSVD,
    NMF as sk_NMF,
    FastICA as sk_FastICA,
)
from lib.sklearndoz.preprocessing import (
    StandardScaler as sk_StandardScaler,
    MinMaxScaler as sk_MinMaxScaler,
    RobustScaler as sk_RobustScaler,
    LabelEncoder as sk_LabelEncoder,
    OneHotEncoder as sk_OneHotEncoder,
    PolynomialFeatures as sk_PolyFeatures,
    SimpleImputer as sk_SimpleImputer,
    Normalizer as sk_Normalizer,
)
from lib.sklearndoz.model_selection import (
    train_test_split as sk_train_test_split,
    KFold as sk_KFold,
    StratifiedKFold as sk_StratifiedKFold,
    cross_val_score as sk_cross_val_score,
    GridSearchCV as sk_GridSearchCV,
    RandomizedSearchCV as sk_RandomizedSearchCV,
)
from lib.sklearndoz.neural_network import (
    MLPClassifier as sk_MLPC,
    MLPRegressor as sk_MLPR,
)
from lib.sklearndoz.pipeline import (
    Pipeline as sk_Pipeline,
    make_pipeline as sk_make_pipeline,
    FeatureUnion as sk_FeatureUnion,
)
from lib.sklearndoz.feature_selection import (
    SelectKBest as sk_SelectKBest,
    VarianceThreshold as sk_VarianceThreshold,
    RFE as sk_RFE,
    f_classif as sk_f_classif,
    f_regression as sk_f_regression,
    mutual_info_classif as sk_mutual_info,
)
from lib.sklearndoz.dummy import (
    DummyClassifier as sk_DummyClassifier,
    DummyRegressor as sk_DummyRegressor,
)
from lib.graphdoz.graphdoz import (
    bfs as graphdoz_bfs,
    dfs as graphdoz_dfs,
    astar as graphdoz_astar,
    make_graph as graphdoz_make_graph,
    make_directed_graph as graphdoz_make_directed,
    adjacency_matrix as graphdoz_adj_matrix,
    path_cost as graphdoz_path_cost,
)
from lib.geneticdoz.geneticdoz import (
    generate_population as gendoz_gen_pop,
    evaluate_fitness as gendoz_eval_fitness,
    select_parents as gendoz_select_parents,
    crossover as gendoz_crossover,
    crossover_uniform as gendoz_crossover_uniform,
    mutate as gendoz_mutate,
    mutate_real as gendoz_mutate_real,
    next_generation as gendoz_next_gen,
    run as gendoz_run,
    knapsack_fitness as gendoz_knapsack_fitness,
)
from lib.acodoz.acodoz import (
    run as acodoz_run,
    make_dist_matrix as acodoz_make_dist,
)
from lib.psodoz.psodoz import run as psodoz_run
from lib.qlearndoz.qlearndoz import (
    make_qtable as qldoz_make_qtable,
    choose_action as qldoz_choose_action,
    update as qldoz_update,
    train as qldoz_train,
    get_policy as qldoz_get_policy,
    get_q_value as qldoz_get_q_value,
)
from lib.mldoz.naive_bayes import (
    nb_fit as mldoz_nb_fit,
    nb_predict as mldoz_nb_predict,
    nb_predict_proba as mldoz_nb_proba,
    nb_predict_batch as mldoz_nb_batch,
)
from lib.mldoz.boosting import (
    adaboost_fit as mldoz_adaboost_fit,
    adaboost_predict as mldoz_adaboost_predict,
    adaboost_predict_batch as mldoz_adaboost_batch,
    gbm_fit as mldoz_gbm_fit,
    gbm_predict as mldoz_gbm_predict,
    gbm_predict_batch as mldoz_gbm_batch,
)
from lib.minimaxdoz.minimaxdoz import (
    minimax_search as mmx_minimax,
    alphabeta_search as mmx_alphabeta,
    make_game as mmx_make_game,
    game_minimax as mmx_game_minimax,
    game_alphabeta as mmx_game_alphabeta,
    ttt_best_move as mmx_ttt_best_move,
)
from lib.mldoz.metrics_ext import (
    roc_curve as mldoz_roc_curve,
    auc as mldoz_auc,
    auc_roc as mldoz_auc_roc,
    precision_recall_curve as mldoz_pr_curve,
    f1_score as mldoz_f1,
    accuracy as mldoz_accuracy,
    confusion_matrix_vals as mldoz_conf_mat,
)
from lib.mldoz.svm_kernel import (
    ksvm_fit as mldoz_ksvm_fit,
    ksvm_predict as mldoz_ksvm_predict,
    ksvm_predict_batch as mldoz_ksvm_batch,
    ksvm_decision as mldoz_ksvm_decision,
)
from lib.mldoz.ensemble import (
    bagging_fit as mldoz_bag_fit,
    bagging_predict as mldoz_bag_predict,
    bagging_predict_batch as mldoz_bag_batch,
    rf_fit as mldoz_rf_fit,
    rf_predict as mldoz_rf_predict,
    rf_predict_batch as mldoz_rf_batch,
    xgb_fit as mldoz_xgb_fit,
    xgb_predict as mldoz_xgb_predict,
    xgb_predict_batch as mldoz_xgb_batch,
)
from lib.mldoz.pipeline import (
    train_val_test_split as mldoz_tvt_split,
    kfold_split as mldoz_kfold_split,
    kfold_cv as mldoz_kfold_cv,
    grid_search as mldoz_grid_search,
    normalize_minmax as mldoz_normalize,
    normalize_apply as mldoz_normalize_apply,
    standardize as mldoz_standardize,
    standardize_apply as mldoz_standardize_apply,
)
from lib.geneticdoz.geneticdoz import (
    generate_population_order as ga_gen_order,
    crossover_order as ga_crossover_order,
    mutate_order as ga_mutate_order,
    run_order as ga_run_order,
    generate_population_real as ga_gen_real,
    run_real as ga_run_real,
)
from lib.plotdoz.plotdoz import (
    learning_curve as plotdoz_learning_curve,
    multi_curve as plotdoz_multi_curve,
    histogram as plotdoz_histogram,
    bar_chart as plotdoz_bar,
    scatter as plotdoz_scatter,
    scatter_classes as plotdoz_scatter_classes,
    line_chart as plotdoz_line,
    confusion_matrix as plotdoz_confusion_matrix,
    heatmap as plotdoz_heatmap,
    cluster_scatter as plotdoz_cluster_scatter,
    learning_curve_ascii as plotdoz_lc_ascii,
    decision_boundary as plotdoz_decision_boundary,
)


# ── Namespace registry — Modulo.funcion() dispatch ───────────────────────────

_NAMESPACES = {
    'Tensordoz': {
        'dot':       lambda args: math_utilsdoz.dot(args[0], args[1]),
        'transpose': lambda args: math_utilsdoz.transpose(args[0]),
        'reshape':   lambda args: math_utilsdoz.reshape(args[0], int(args[1]), int(args[2])),
        'flatten':   lambda args: math_utilsdoz.flatten(args[0]),
        'add':       lambda args: math_utilsdoz.add(args[0], args[1]),
        'sub':       lambda args: math_utilsdoz.sub(args[0], args[1]),
        'scale':     lambda args: math_utilsdoz.scale(args[0], args[1]),
        'zeros':     lambda args: math_utilsdoz.zeros(int(args[0]), int(args[1])),
        'ones':      lambda args: math_utilsdoz.ones(int(args[0]), int(args[1])),
        'concat':    lambda args: math_utilsdoz.concat(args[0], args[1], int(args[2]) if len(args) > 2 else 0),
    },
    'Mathdoz': {
        'relu':      lambda args: math_utilsdoz.relu(args[0]),
        'sigmoid':   lambda args: math_utilsdoz.sigmoid(args[0]),
        'exp':       lambda args: math_utilsdoz.exp_doz(args[0]),
        'log':       lambda args: math_utilsdoz.log_doz(args[0]),
        'sin':       lambda args: math_utilsdoz.sin_doz(args[0]),
        'cos':       lambda args: math_utilsdoz.cos_doz(args[0]),
        'tan':       lambda args: math_utilsdoz.tan_doz(args[0]),
        'sqrt':      lambda args: math_utilsdoz.sqrt_doz(args[0]),
        'abs':       lambda args: math_utilsdoz.abs_doz(args[0]),
        'pow':       lambda args: math_utilsdoz.pow_doz(args[0], args[1]),
        'factorial': lambda args: math_utilsdoz.factorial_doz(args[0]),
    },
    'IOdoz': {
        'leer':    lambda args: read_file_doz(args[0]),
        'escribir': lambda args: write_file_doz(args[0], args[1]),
    },
    'Inspectdoz': {
        'spy':   lambda args: inspectdoz_spy(args[0]),
        'shape': lambda args: inspectdoz_shape(args[0]),
    },
    'Optimdoz': {
        'sgd':        lambda args: optimdoz_sgd(args[0], args[1], *args[2:]),
        'adam':       lambda args: optimdoz_adam(args[0], args[1], args[2], args[3], int(args[4]), *args[5:]),
        'zeros_like': lambda args: optimdoz_zeros(args[0]),
    },
    'Metricsdoz': {
        'mse':                  lambda args: math_utilsdoz.mse(args[0], args[1]),
        'mse_grad':             lambda args: math_utilsdoz.mse_grad(args[0], args[1]),
        'cross_entropy':        lambda args: lossdoz_ce(args[0], args[1]),
        'cross_entropy_grad':   lambda args: lossdoz_ce_grad(args[0], args[1]),
        'bce':                  lambda args: lossdoz_bce(args[0], args[1]),
        'bce_grad':             lambda args: lossdoz_bce_grad(args[0], args[1]),
        'contrastive_loss':     lambda args: lossdoz_contrastive(
                                    args[0], args[1],
                                    args[2] if len(args) > 2 else 1.0,
                                ),
        'contrastive_grad':     lambda args: lossdoz_contrastive_grad(
                                    args[0], args[1], args[2],
                                    args[3] if len(args) > 3 else 1.0,
                                ),
        'triplet_loss':         lambda args: lossdoz_triplet(
                                    args[0], args[1], args[2],
                                    args[3] if len(args) > 3 else 0.2,
                                ),
        'triplet_grad':         lambda args: lossdoz_triplet_grad(
                                    args[0], args[1], args[2],
                                    args[3] if len(args) > 3 else 0.2,
                                ),
        'kl_divergence':        lambda args: lossdoz_kl(args[0], args[1]),
        'kl_grad':              lambda args: lossdoz_kl_grad(args[0], args[1]),
        'huber_loss':           lambda args: lossdoz_huber(
                                    args[0], args[1],
                                    args[2] if len(args) > 2 else 1.0,
                                ),
        'accuracy':             lambda args: metricsdoz_accuracy(args[0], args[1]),
        'precision':            lambda args: metricsdoz_precision(args[0], args[1], args[2] if len(args) > 2 else 1),
        'recall':               lambda args: metricsdoz_recall(args[0], args[1], args[2] if len(args) > 2 else 1),
        'f1_score':             lambda args: metricsdoz_f1(args[0], args[1], args[2] if len(args) > 2 else 1),
        'confusion_matrix':     lambda args: metricsdoz_confusion(args[0], args[1], args[2] if len(args) > 2 else None),
        'rmse':                 lambda args: metricsdoz_rmse(args[0], args[1]),
        'mae':                  lambda args: metricsdoz_mae(args[0], args[1]),
        'r2_score':             lambda args: metricsdoz_r2(args[0], args[1]),
    },
    'NNdoz': {
        # ── Capas básicas ──
        'linear_init':        lambda args: nndoz_linear_init(int(args[0]), int(args[1])),
        'linear_forward':     lambda args: nndoz_linear_forward(args[0], args[1]),
        'linear_backward':    lambda args: nndoz_linear_backward(args[0], args[1], args[2]),
        # ── Activaciones ──
        'relu_forward':       lambda args: list(nndoz_relu_forward(args[0])),
        'relu_backward':      lambda args: nndoz_relu_backward(args[0], args[1]),
        'sigmoid_forward':    lambda args: nndoz_sigmoid_forward(args[0]),
        'sigmoid_backward':   lambda args: nndoz_sigmoid_backward(args[0], args[1]),
        'tanh_forward':       lambda args: nndoz_tanh_forward(args[0]),
        'tanh_backward':      lambda args: nndoz_tanh_backward(args[0], args[1]),
        'softmax':            lambda args: nndoz_softmax(args[0]),
        # ── Regularización ──
        'batchnorm_init':     lambda args: nndoz_bn_init(
                                  int(args[0]),
                                  args[1] if len(args) > 1 else 1e-5,
                              ),
        'batchnorm_forward':  lambda args: nndoz_bn_forward(
                                  args[0], args[1],
                                  bool(args[2]) if len(args) > 2 else True,
                              ),
        'batchnorm_backward': lambda args: nndoz_bn_backward(args[0], args[1]),
        'dropout_forward':    lambda args: nndoz_dropout_forward(
                                  args[0],
                                  args[1] if len(args) > 1 else 0.5,
                                  bool(args[2]) if len(args) > 2 else True,
                              ),
        'dropout_backward':   lambda args: nndoz_dropout_backward(args[0], args[1]),
        # ── Siamesas / embeddings ──
        'shared_forward':     lambda args: nndoz_shared_forward(args[0], args[1], args[2]),
        'shared_backward':    lambda args: nndoz_shared_backward(args[0], args[1], args[2], args[3], args[4]),
        'embedding_init':     lambda args: nndoz_embedding_init(int(args[0]), int(args[1])),
        'embedding_forward':  lambda args: nndoz_embedding_forward(args[0], args[1]),
        'embedding_backward': lambda args: nndoz_embedding_backward(args[0], args[1], args[2], args[3]),
        'l2_distance':        lambda args: nndoz_l2_dist(args[0], args[1]),
        'cosine_similarity':  lambda args: nndoz_cosine_sim(args[0], args[1]),
        # ── MLP completo ──
        'sequential':         lambda args: list(nndoz_sequential(args[0], args[1])),
        'get_params':         lambda args: nndoz_params(args[0]),
        'param_count':        lambda args: nndoz_param_count(args[0]),
        'perceptron_fit':     lambda args: nndoz_perceptron_fit(
                                  args[0], args[1],
                                  args[2] if len(args) > 2 else 0.1,
                                  int(args[3]) if len(args) > 3 else 100,
                              ),
        'perceptron_predict': lambda args: nndoz_perceptron_predict(args[0], args[1]),
        'mlp_init':           lambda args: nndoz_mlp_init(args[0]),
        'mlp_train':          lambda args: nndoz_mlp_train(
                                  args[0], args[1], args[2],
                                  args[3] if len(args) > 3 else 0.01,
                                  int(args[4]) if len(args) > 4 else 1000,
                                  args[5] if len(args) > 5 else 'mse',
                              ),
        'mlp_predict':        lambda args: nndoz_mlp_predict(args[0], args[1]),
    },
    'Autodoz': {
        'tensor':       lambda args: autodoz_tensor(args[0]),
        'zeros':        lambda args: autodoz_zeros(int(args[0]), int(args[1]) if len(args) > 1 else None),
        'ones':         lambda args: autodoz_ones(int(args[0]), int(args[1]) if len(args) > 1 else None),
        'backward':     lambda args: args[0].backward() or args[0],
        'zero_grad':    lambda args: args[0].zero_grad() or args[0],
        'grad':         lambda args: args[0].grad,
        'data':         lambda args: args[0].data,
        'shape':        lambda args: list(args[0].shape),
        'relu':         lambda args: args[0].relu(),
        'sigmoid':      lambda args: args[0].sigmoid(),
        'tanh':         lambda args: args[0].tanh(),
        'log':          lambda args: args[0].log(),
        'exp':          lambda args: args[0].exp(),
        'sum':          lambda args: args[0].sum(),
        'mean':         lambda args: args[0].mean(),
        'matmul':       lambda args: args[0].matmul(args[1]),
        'T':            lambda args: args[0].T(),
    },
    'Datadoz': {
        'from_lists':        lambda args: datadoz_from_lists(args[0], args[1]),
        'make_loader':       lambda args: datadoz_loader(args[0], int(args[1]), bool(args[2]) if len(args) > 2 else False),
        'get_batches':       lambda args: datadoz_batches(args[0]),
        'train_test_split':  lambda args: datadoz_split(args[0], args[1], args[2] if len(args) > 2 else 0.2),
        'read_csv':          lambda args: datadoz_read_csv(args[0], args[1] if len(args) > 1 else ','),
        'read_xlsx':         lambda args: datadoz_read_xlsx(args[0]),
        'read_tsv':          lambda args: datadoz_read_tsv(args[0]),
        'read_json':         lambda args: datadoz_read_json(args[0]),
        'write_csv':         lambda args: datadoz_write_csv(args[0], args[1], args[2] if len(args) > 2 else ','),
        'write_json':        lambda args: datadoz_write_json(args[0], args[1]),
        'rename_col':        lambda args: datadoz_rename_col(args[0], args[1], args[2]),
        'drop_col':          lambda args: datadoz_drop_col(args[0], args[1]),
        'add_col':           lambda args: datadoz_add_col(args[0], args[1], args[2]),
        'filter_rows':       lambda args: datadoz_filter_rows(args[0], args[1], args[2], args[3]),
        'columna':           lambda args: datadoz_col(args[0], args[1]),
        'fila':              lambda args: datadoz_row(args[0], int(args[1])),
        'num_filas':         lambda args: datadoz_nrows(args[0]),
        'num_columnas':      lambda args: datadoz_ncols(args[0]),
        'columnas':          lambda args: datadoz_col_names(args[0]),
    },
    'Plotdoz': {
        'learning_curve':   lambda args: plotdoz_learning_curve(args[0], args[1] if len(args) > 1 else None, *args[2:]),
        'multi_curve':      lambda args: plotdoz_multi_curve(args[0], *args[1:]),
        'histogram':        lambda args: plotdoz_histogram(args[0], *args[1:]),
        'bar_chart':        lambda args: plotdoz_bar(args[0], args[1], *args[2:]),
        'scatter':          lambda args: plotdoz_scatter(args[0], args[1], *args[2:]),
        'scatter_classes':  lambda args: plotdoz_scatter_classes(args[0], args[1], args[2], *args[3:]),
        'line_chart':       lambda args: plotdoz_line(args[0], args[1], *args[2:]),
        'confusion_matrix':  lambda args: plotdoz_confusion_matrix(args[0], args[1], *args[2:]),
        'heatmap':           lambda args: plotdoz_heatmap(args[0], *args[1:]),
        'cluster_scatter':      lambda args: plotdoz_cluster_scatter(args[0], args[1], args[2], *args[3:]),
        'learning_curve_ascii': lambda args: plotdoz_lc_ascii(args[0], *args[1:]),
        'decision_boundary':    lambda args: plotdoz_decision_boundary(
                                    args[0], args[1],
                                    (lambda modelo, fn: (lambda pts: fn(modelo, pts)))(args[2], args[3]),
                                    *args[4:],
                                ),
    },
    'Mldoz': {
        'linreg_fit':           lambda args: mldoz_linreg_fit(
                                    args[0], args[1],
                                    args[2] if len(args) > 2 else 0.01,
                                    int(args[3]) if len(args) > 3 else 500,
                                ),
        'linreg_predict':       lambda args: mldoz_linreg_predict(args[0], args[1]),
        'logreg_fit':           lambda args: mldoz_logreg_fit(
                                    args[0], args[1],
                                    args[2] if len(args) > 2 else 0.1,
                                    int(args[3]) if len(args) > 3 else 500,
                                ),
        'logreg_predict':       lambda args: mldoz_logreg_predict(
                                    args[0], args[1],
                                    args[2] if len(args) > 2 else 0.5,
                                ),
        'logreg_predict_proba': lambda args: mldoz_logreg_proba(args[0], args[1]),
        'svm_fit':              lambda args: mldoz_svm_fit(
                                    args[0], args[1],
                                    args[2] if len(args) > 2 else 0.01,
                                    int(args[3]) if len(args) > 3 else 1000,
                                    args[4] if len(args) > 4 else 0.01,
                                ),
        'svm_predict':          lambda args: mldoz_svm_predict(args[0], args[1]),
        'tree_clf_fit':         lambda args: mldoz_tree_clf_fit(
                                    args[0], args[1],
                                    int(args[2]) if len(args) > 2 else 5,
                                ),
        'tree_reg_fit':         lambda args: mldoz_tree_reg_fit(
                                    args[0], args[1],
                                    int(args[2]) if len(args) > 2 else 5,
                                ),
        'tree_predict':         lambda args: mldoz_tree_predict(args[0], args[1]),
        'knn_fit':              lambda args: mldoz_knn_fit(
                                    args[0], args[1],
                                    int(args[2]) if len(args) > 2 else 3,
                                ),
        'knn_predict':          lambda args: mldoz_knn_predict(args[0], args[1]),
        'knn_predict_clf':      lambda args: mldoz_knn_clf(args[0], args[1]),
        'knn_predict_reg':      lambda args: mldoz_knn_reg(args[0], args[1]),
        'kmeans_fit':           lambda args: mldoz_kmeans_fit(
                                    args[0],
                                    int(args[1]),
                                    int(args[2]) if len(args) > 2 else 100,
                                ),
        'kmeans_predict':       lambda args: mldoz_kmeans_predict(args[0], args[1]),
        # Naive Bayes (Grokking ML Cap 8)
        'nb_fit':               lambda args: mldoz_nb_fit(args[0], args[1]),
        'nb_predict':           lambda args: mldoz_nb_predict(args[0], args[1]),
        'nb_predict_proba':     lambda args: mldoz_nb_proba(args[0], args[1]),
        'nb_predict_batch':     lambda args: mldoz_nb_batch(args[0], args[1]),
        # AdaBoost (Grokking ML Cap 12)
        'adaboost_fit':         lambda args: mldoz_adaboost_fit(
                                    args[0], args[1],
                                    int(args[2]) if len(args) > 2 else 50,
                                ),
        'adaboost_predict':     lambda args: mldoz_adaboost_predict(args[0], args[1]),
        'adaboost_batch':       lambda args: mldoz_adaboost_batch(args[0], args[1]),
        # Gradient Boosting (Grokking ML Cap 12)
        'gbm_fit':              lambda args: mldoz_gbm_fit(
                                    args[0], args[1],
                                    int(args[2]) if len(args) > 2 else 100,
                                    args[3] if len(args) > 3 else 0.1,
                                    int(args[4]) if len(args) > 4 else 3,
                                ),
        'gbm_predict':          lambda args: mldoz_gbm_predict(args[0], args[1]),
        'gbm_predict_batch':    lambda args: mldoz_gbm_batch(args[0], args[1]),
        # ROC / AUC / métricas (Grokking ML Cap 7)
        'roc_curve':            lambda args: mldoz_roc_curve(args[0], args[1]),
        'auc':                  lambda args: mldoz_auc(args[0], args[1]),
        'auc_roc':              lambda args: mldoz_auc_roc(args[0], args[1]),
        'pr_curve':             lambda args: mldoz_pr_curve(args[0], args[1]),
        'f1':                   lambda args: mldoz_f1(args[0], args[1]),
        'accuracy':             lambda args: mldoz_accuracy(args[0], args[1]),
        'conf_mat':             lambda args: mldoz_conf_mat(args[0], args[1]),
        # SVM con kernel (Grokking ML Cap 11)
        'ksvm_fit':             lambda args: mldoz_ksvm_fit(
                                    args[0], args[1],
                                    args[2] if len(args) > 2 else 'rbf',
                                    args[3] if len(args) > 3 else 1.0,
                                    args[4] if len(args) > 4 else 0.5,
                                    int(args[5]) if len(args) > 5 else 3,
                                ),
        'ksvm_predict':         lambda args: mldoz_ksvm_predict(args[0], args[1]),
        'ksvm_batch':           lambda args: mldoz_ksvm_batch(args[0], args[1]),
        'ksvm_decision':        lambda args: mldoz_ksvm_decision(args[0], args[1]),
        # Bagging / Random Forest (Grokking ML Cap 12)
        'bagging_fit':          lambda args: mldoz_bag_fit(
                                    args[0], args[1],
                                    int(args[2]) if len(args) > 2 else 10,
                                    int(args[3]) if len(args) > 3 else 5,
                                ),
        'bagging_predict':      lambda args: mldoz_bag_predict(args[0], args[1]),
        'bagging_batch':        lambda args: mldoz_bag_batch(args[0], args[1]),
        'rf_fit':               lambda args: mldoz_rf_fit(
                                    args[0], args[1],
                                    int(args[2]) if len(args) > 2 else 10,
                                    int(args[3]) if len(args) > 3 else 5,
                                ),
        'rf_predict':           lambda args: mldoz_rf_predict(args[0], args[1]),
        'rf_batch':             lambda args: mldoz_rf_batch(args[0], args[1]),
        # XGBoost-style (Grokking ML Cap 12)
        'xgb_fit':              lambda args: mldoz_xgb_fit(
                                    args[0], args[1],
                                    int(args[2]) if len(args) > 2 else 100,
                                    args[3] if len(args) > 3 else 0.1,
                                    int(args[4]) if len(args) > 4 else 3,
                                    args[5] if len(args) > 5 else 1.0,
                                ),
        'xgb_predict':          lambda args: mldoz_xgb_predict(args[0], args[1]),
        'xgb_batch':            lambda args: mldoz_xgb_batch(args[0], args[1]),
        # Pipeline utilities (Grokking ML Cap 13)
        'tvt_split':            lambda args: mldoz_tvt_split(
                                    args[0], args[1],
                                    args[2] if len(args) > 2 else 0.15,
                                    args[3] if len(args) > 3 else 0.15,
                                ),
        'kfold_split':          lambda args: mldoz_kfold_split(
                                    args[0], args[1],
                                    int(args[2]) if len(args) > 2 else 5,
                                ),
        'normalize':            lambda args: mldoz_normalize(args[0]),
        'normalize_apply':      lambda args: mldoz_normalize_apply(args[0], args[1]),
        'standardize':          lambda args: mldoz_standardize(args[0]),
        'standardize_apply':    lambda args: mldoz_standardize_apply(args[0], args[1]),
    },
    # ── Preprocessdoz: PCA, imputación, logreg multiclase, feature importance ──
    'Preprocessdoz': {
        'pca_fit':              lambda args: preproc_pca_fit(args[0], int(args[1]) if len(args) > 1 else 2),
        'pca_transform':        lambda args: preproc_pca_transform(args[0], args[1]),
        'pca_fit_transform':    lambda args: preproc_pca_fit_transform(args[0], int(args[1]) if len(args) > 1 else 2),
        'pca_explained_ratio':  lambda args: preproc_pca_ratio(args[0]),
        'impute_mean':          lambda args: preproc_impute_mean(args[0]),
        'impute_median':        lambda args: preproc_impute_median(args[0]),
        'impute_constant':      lambda args: preproc_impute_const(args[0], args[1] if len(args) > 1 else 0),
        'drop_nulls':           lambda args: preproc_drop_nulls(args[0]),
        'null_count':           lambda args: preproc_null_count(args[0]),
        'write_xlsx':           lambda args: preproc_write_xlsx(args[0], args[1]),
        'logreg_multi_fit':     lambda args: preproc_logreg_multi_fit(
                                    args[0], args[1],
                                    args[2] if len(args) > 2 else 0.1,
                                    int(args[3]) if len(args) > 3 else 500,
                                ),
        'logreg_multi_predict': lambda args: preproc_logreg_multi_predict(args[0], args[1]),
        'logreg_multi_proba':   lambda args: preproc_logreg_multi_proba(args[0], args[1]),
        'feature_importance':   lambda args: preproc_feat_importance(args[0], int(args[1])),
        'validation_curve':     lambda args: preproc_val_curve(
                                    args[0], args[1], args[2], args[3], args[4], args[5],
                                    args[6],
                                    args[7] if len(args) > 7 else None,
                                ),
    },
    # ── Graphdoz: BFS, DFS, A* (Grokking AI Algorithms Caps 2-3) ─────────────
    'Graphdoz': {
        'bfs':              lambda args: graphdoz_bfs(args[0], args[1], args[2]),
        'dfs':              lambda args: graphdoz_dfs(args[0], args[1], args[2]),
        'astar':            lambda args: graphdoz_astar(
                                args[0], args[1], args[2],
                                args[3] if len(args) > 3 else None,
                            ),
        'make_graph':       lambda args: graphdoz_make_graph(args[0]),
        'make_directed':    lambda args: graphdoz_make_directed(args[0]),
        'adj_matrix':       lambda args: graphdoz_adj_matrix(args[0], args[1]),
        'path_cost':        lambda args: graphdoz_path_cost(args[0], args[1]),
    },
    # ── Geneticdoz: Algoritmo Genético (Grokking AI Algorithms Caps 4-5) ──────
    'Geneticdoz': {
        'gen_population':   lambda args: gendoz_gen_pop(int(args[0]), int(args[1])),
        'eval_fitness':     lambda args: gendoz_eval_fitness(args[0], args[1]),
        'select_parents':   lambda args: gendoz_select_parents(
                                args[0],
                                int(args[1]) if len(args) > 1 else 2,
                                args[2] if len(args) > 2 else "tournament",
                            ),
        'crossover':        lambda args: gendoz_crossover(
                                args[0], args[1],
                                int(args[2]) if len(args) > 2 else None,
                            ),
        'crossover_uniform':lambda args: gendoz_crossover_uniform(args[0], args[1]),
        'mutate':           lambda args: gendoz_mutate(
                                args[0],
                                args[1] if len(args) > 1 else 0.01,
                            ),
        'mutate_real':      lambda args: gendoz_mutate_real(
                                args[0],
                                args[1] if len(args) > 1 else 0.1,
                                args[2] if len(args) > 2 else 0.1,
                            ),
        'next_generation':  lambda args: gendoz_next_gen(
                                args[0], args[1],
                                args[2] if len(args) > 2 else 0.01,
                                int(args[3]) if len(args) > 3 else 2,
                            ),
        'run':              lambda args: gendoz_run(
                                args[0], int(args[1]),
                                int(args[2]) if len(args) > 2 else 50,
                                int(args[3]) if len(args) > 3 else 100,
                                args[4] if len(args) > 4 else 0.01,
                                int(args[5]) if len(args) > 5 else 2,
                            ),
        'knapsack_fitness': lambda args: gendoz_knapsack_fitness(
                                args[0], args[1], args[2],
                            ),
        # Cap 5: codificación de orden (permutaciones, TSP)
        'gen_population_order': lambda args: ga_gen_order(int(args[0]), args[1]),
        'crossover_order':   lambda args: ga_crossover_order(args[0], args[1]),
        'mutate_order':      lambda args: ga_mutate_order(
                                 args[0],
                                 args[1] if len(args) > 1 else 0.01,
                             ),
        'run_order':         lambda args: ga_run_order(
                                 args[0], args[1],
                                 int(args[2]) if len(args) > 2 else 50,
                                 int(args[3]) if len(args) > 3 else 100,
                                 args[4] if len(args) > 4 else 0.05,
                                 int(args[5]) if len(args) > 5 else 2,
                             ),
        # Cap 5: codificación real (optimización continua)
        'gen_population_real': lambda args: ga_gen_real(int(args[0]), int(args[1]), args[2]),
        'run_real':          lambda args: ga_run_real(
                                 args[0], int(args[1]), args[2],
                                 int(args[3]) if len(args) > 3 else 50,
                                 int(args[4]) if len(args) > 4 else 100,
                                 args[5] if len(args) > 5 else 0.1,
                                 args[6] if len(args) > 6 else 0.1,
                                 int(args[7]) if len(args) > 7 else 2,
                             ),
    },
    # ── Acodoz: Ant Colony Optimization (Grokking AI Algorithms Cap 6) ────────
    'Acodoz': {
        'run':           lambda args: acodoz_run(
                             args[0],
                             int(args[1]) if len(args) > 1 else 20,
                             int(args[2]) if len(args) > 2 else 100,
                             args[3] if len(args) > 3 else 1.0,
                             args[4] if len(args) > 4 else 2.0,
                             args[5] if len(args) > 5 else 0.5,
                         ),
        'make_dist_matrix': lambda args: acodoz_make_dist(args[0]),
    },
    # ── Psodoz: Particle Swarm Optimization (Grokking AI Algorithms Cap 7) ────
    'Psodoz': {
        'run':           lambda args: psodoz_run(
                             args[0], int(args[1]),
                             int(args[2]) if len(args) > 2 else 30,
                             int(args[3]) if len(args) > 3 else 100,
                             args[4] if len(args) > 4 else None,
                             args[5] if len(args) > 5 else 0.7,
                             args[6] if len(args) > 6 else 1.5,
                             args[7] if len(args) > 7 else 1.5,
                             bool(args[8]) if len(args) > 8 else True,
                         ),
    },
    # ── Qlearndoz: Q-Learning (Grokking AI Algorithms Cap 10) ─────────────────
    'Qlearndoz': {
        'make_qtable':   lambda args: qldoz_make_qtable(
                             args[0], args[1],
                             args[2] if len(args) > 2 else 0.0,
                         ),
        'choose_action': lambda args: qldoz_choose_action(
                             args[0], args[1], args[2],
                             args[3] if len(args) > 3 else 0.1,
                         ),
        'update':        lambda args: qldoz_update(
                             args[0], args[1], args[2], args[3], args[4],
                             args[5] if len(args) > 5 else 0.1,
                             args[6] if len(args) > 6 else 0.9,
                         ),
        'train':         lambda args: qldoz_train(
                             args[0], args[1], args[2],
                             int(args[3]) if len(args) > 3 else 1000,
                             args[4] if len(args) > 4 else 0.1,
                             args[5] if len(args) > 5 else 0.9,
                             args[6] if len(args) > 6 else 0.1,
                         ),
        'get_policy':    lambda args: qldoz_get_policy(args[0]),
        'get_q_value':   lambda args: qldoz_get_q_value(args[0], args[1], args[2]),
    },
    # ── Minimaxdoz: Minimax + Alpha-Beta Pruning (Grokking AI Algorithms Cap 3) ─
    'Minimaxdoz': {
        'minimax':     lambda args: mmx_minimax(
                           args[0], args[1], args[2], args[3], args[4],
                           int(args[5]) if len(args) > 5 else 4,
                           bool(args[6]) if len(args) > 6 else True,
                       ),
        'alphabeta':   lambda args: mmx_alphabeta(
                           args[0], args[1], args[2], args[3], args[4],
                           int(args[5]) if len(args) > 5 else 4,
                           bool(args[6]) if len(args) > 6 else True,
                       ),
        'make_game':   lambda args: mmx_make_game(args[0], args[1], args[2], args[3]),
        'game_minimax':  lambda args: mmx_game_minimax(
                             args[0], args[1],
                             int(args[2]) if len(args) > 2 else 4,
                             bool(args[3]) if len(args) > 3 else True,
                         ),
        'game_alphabeta': lambda args: mmx_game_alphabeta(
                              args[0], args[1],
                              int(args[2]) if len(args) > 2 else 4,
                              bool(args[3]) if len(args) > 3 else True,
                          ),
        'ttt_best_move': lambda args: mmx_ttt_best_move(
                             args[0],
                             int(args[1]) if len(args) > 1 else 9,
                         ),
    },
    'Sklearndoz': {
        # ── Preprocesado (útil antes/después de DL) ──
        'StandardScaler':     lambda args: sk_StandardScaler(),
        'MinMaxScaler':       lambda args: sk_MinMaxScaler(),
        'RobustScaler':       lambda args: sk_RobustScaler(),
        'LabelEncoder':       lambda args: sk_LabelEncoder(),
        'OneHotEncoder':      lambda args: sk_OneHotEncoder(),
        'SimpleImputer':      lambda args: sk_SimpleImputer(args[0] if args else 'mean'),
        'Normalizer':         lambda args: sk_Normalizer(args[0] if args else 'l2'),
        # ── Reducción de dimensionalidad (análisis / visualización) ──
        'PCA':                lambda args: sk_PCA(int(args[0]) if args else 2),
        'NMF':                lambda args: sk_NMF(int(args[0]) if args else 2),
        # ── Clustering (aprendizaje no supervisado) ──
        'KMeans':             lambda args: sk_KMeans(int(args[0]) if args else 8),
        'DBSCAN':             lambda args: sk_DBSCAN(
                                  args[0] if args else 0.5,
                                  int(args[1]) if len(args) > 1 else 5,
                              ),
        'GaussianMixture':    lambda args: sk_GMM(int(args[0]) if args else 1),
        # ── Baselines clásicos (comparación con DL) ──
        'LogisticRegression': lambda args: sk_LogisticRegression(
                                  args[0] if args else 1.0,
                                  int(args[1]) if len(args) > 1 else 100,
                              ),
        'RandomForestClassifier': lambda args: sk_RFC(
                                  int(args[0]) if args else 100,
                              ),
        'KNeighborsClassifier':   lambda args: sk_KNC(int(args[0]) if args else 5),
        'GaussianNB':             lambda args: sk_GaussianNB(),
        # ── Validación de modelos ──
        'train_test_split':   lambda args: sk_train_test_split(
                                  args[0], args[1],
                                  test_size=args[2] if len(args) > 2 else 0.2,
                              ),
        'KFold':              lambda args: sk_KFold(int(args[0]) if args else 5),
        'cross_val_score':    lambda args: sk_cross_val_score(
                                  args[0], args[1], args[2],
                                  cv=int(args[3]) if len(args) > 3 else 5,
                              ),
        # ── Métodos de instancia ──
        'fit':            lambda args: args[0].fit(args[1], args[2] if len(args) > 2 else None),
        'predict':        lambda args: args[0].predict(args[1]),
        'predict_proba':  lambda args: args[0].predict_proba(args[1]),
        'transform':      lambda args: args[0].transform(args[1]),
        'fit_transform':  lambda args: args[0].fit_transform(args[1]),
        'fit_predict':    lambda args: args[0].fit_predict(args[1]),
        'score':          lambda args: args[0].score(args[1], args[2]),
    },
    'Randomdoz': {
        'seed':    lambda args: randomdoz_seed(int(args[0])),
        'random':  lambda args: randomdoz_random(),
        'uniform': lambda args: randomdoz_uniform(args[0], args[1]),
        'randint': lambda args: randomdoz_randint(int(args[0]), int(args[1])),
        'choice':  lambda args: randomdoz_choice(args[0]),
        'sample':  lambda args: randomdoz_sample(args[0], int(args[1])),
        'shuffle': lambda args: randomdoz_shuffle(list(args[0])),
        'gauss':   lambda args: randomdoz_gauss(
            args[0] if len(args) > 0 else 0.0,
            args[1] if len(args) > 1 else 1.0,
        ),
    },
}


# ── Signal for return statement ───────────────────────────────────────────────

class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


# ── Callable types ────────────────────────────────────────────────────────────

class TrezFunction:
    """Named closure: captures params, body AST, and definition environment."""
    def __init__(self, name, params, body_ctx, env):
        self.name = name
        self.params = params
        self.body_ctx = body_ctx
        self.env = env

    def __repr__(self):
        return f"<func {self.name}({', '.join(self.params)})>"


class TrezLambda:
    """Anonymous single-parameter lambda: \\param -> rhs."""
    def __init__(self, param, body_ctx, env):
        self.param = param
        self.body_ctx = body_ctx
        self.env = env

    def __repr__(self):
        return f"<lambda {self.param}>"


class TrezBuiltin:
    """Wraps a native Python callable so it can be passed as a first-class value."""
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def __repr__(self):
        return f"<builtin {self.name}>"


# ── Scope chain ───────────────────────────────────────────────────────────────

class Environment:
    """Linked scopes — functional lookup chain, no global mutable state."""
    def __init__(self, parent=None):
        self.bindings = {}
        self.parent = parent

    def get(self, name):
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.get(name)
        raise UndefinedSymbolError(name)

    def set(self, name, value):
        self.bindings[name] = value

    def update(self, name, value):
        """Update existing binding in nearest enclosing scope; create locally if not found."""
        if name in self.bindings:
            self.bindings[name] = value
            return
        if self.parent and self.parent.has(name):
            self.parent.update(name, value)
            return
        self.bindings[name] = value

    def has(self, name):
        if name in self.bindings:
            return True
        if self.parent:
            return self.parent.has(name)
        return False


# ── Visitor ───────────────────────────────────────────────────────────────────

class TrezVisitor(AntlrTrezVisitor):
    def __init__(self):
        super().__init__()
        self.global_env = Environment()
        self.env = self.global_env
        self._call_depth = 0  # >0 when inside a function/lambda body
        self._register_builtins()

    def _register_builtins(self):
        """Register native math functions as first-class values in global scope."""
        builtins = {
            'relu':      lambda x: math_utilsdoz.relu(x),
            'sigmoid':   lambda x: math_utilsdoz.sigmoid(x),
            'sqrt':      lambda x: math_utilsdoz.sqrt_doz(x),
            'exp':       lambda x: math_utilsdoz.exp_doz(x),
            'log':       lambda x: math_utilsdoz.log_doz(x),
            'sin':       lambda x: math_utilsdoz.sin_doz(x),
            'cos':       lambda x: math_utilsdoz.cos_doz(x),
            'tan':       lambda x: math_utilsdoz.tan_doz(x),
            'abs':       lambda x: math_utilsdoz.abs_doz(x),
            'factorial': lambda x: math_utilsdoz.factorial_doz(x),
        }
        for name, fn in builtins.items():
            self.global_env.set(name, TrezBuiltin(name, fn))

    # ── program ──────────────────────────────────────────────────────────────

    def visitProgram(self, ctx: TrezParser.ProgramContext):
        for stmt in ctx.statement():
            self.visit(stmt)
        return None

    # ── rhs delegation (lambda | expr) ──────────────────────────────────────

    def visitLambdaDef(self, ctx: TrezParser.LambdaDefContext):
        param = ctx.ID().getText()
        return TrezLambda(param, ctx.rhs(), self.env)

    def visitExprRhs(self, ctx: TrezParser.ExprRhsContext):
        return self.visit(ctx.expr())

    # ── statements ───────────────────────────────────────────────────────────

    def visitLet_stmt(self, ctx: TrezParser.Let_stmtContext):
        name = ctx.ID().getText()
        value = self.visit(ctx.rhs())
        self.env.update(name, value)
        return value

    def visitBind_tuple(self, ctx: TrezParser.Bind_tupleContext):
        """let [a, b, c] = rhs  — destructures a list into named bindings."""
        names = [tok.getText() for tok in ctx.ID()]
        value = self.visit(ctx.rhs())
        if not isinstance(value, list):
            raise TrezRuntimeError(
                f"Desestructuración de tupla requiere una lista, recibió {type(value).__name__}."
            )
        if len(value) < len(names):
            raise TrezRuntimeError(
                f"Desestructuración: se esperaban {len(names)} elementos, la lista tiene {len(value)}."
            )
        for name, val in zip(names, value):
            self.env.update(name, val)
        return value

    def visitFunc_def(self, ctx: TrezParser.Func_defContext):
        name = ctx.ID().getText()
        params = [p.getText() for p in ctx.param_list().ID()] if ctx.param_list() else []
        fn = TrezFunction(name, params, ctx.block(), self.env)
        self.env.set(name, fn)
        return fn

    def visitReturn_stmt(self, ctx: TrezParser.Return_stmtContext):
        raise ReturnSignal(self.visit(ctx.rhs()))

    def visitExpr_stmt(self, ctx: TrezParser.Expr_stmtContext):
        result = self.visit(ctx.rhs())
        if result is not None and self._call_depth == 0:
            print(result)
        return result

    def visitIf_stmt(self, ctx: TrezParser.If_stmtContext):
        cond = self.visit(ctx.rhs())
        if cond:
            return self.visit(ctx.block(0))
        if ctx.if_stmt():
            return self.visit(ctx.if_stmt())
        if ctx.block(1) is not None:
            return self.visit(ctx.block(1))
        return None

    def visitWhile_stmt(self, ctx: TrezParser.While_stmtContext):
        while self.visit(ctx.rhs()):
            self.visit(ctx.block())
        return None

    def visitFor_stmt(self, ctx: TrezParser.For_stmtContext):
        var_name = ctx.ID().getText()
        iterable = self.visit(ctx.rhs())
        if isinstance(iterable, (TrezQueue, TrezStack)):
            iterable = iterable.to_list()
        if not isinstance(iterable, (list, str)):
            raise TrezRuntimeError("for..in requiere una lista o texto.")
        for item in iterable:
            loop_env = Environment(self.env)
            loop_env.set(var_name, item)
            saved = self.env
            self.env = loop_env
            try:
                self.visit(ctx.block())
            finally:
                self.env = saved
        return None

    def visitBlock(self, ctx: TrezParser.BlockContext):
        block_env = Environment(self.env)
        saved = self.env
        self.env = block_env
        result = None
        try:
            for stmt in ctx.statement():
                result = self.visit(stmt)
        finally:
            self.env = saved
        return result

    # ── lambda & pipe ─────────────────────────────────────────────────────────

    def visitPipeOp(self, ctx: TrezParser.PipeOpContext):
        """expr |> expr  ≡  right(left) — left-associative"""
        value = self.visit(ctx.expr(0))
        fn    = self.visit(ctx.expr(1))
        return self._apply(fn, [value])

    def _apply(self, fn, args):
        """Call a TrezFunction, TrezLambda, or builtin with args."""
        if isinstance(fn, TrezLambda):
            call_env = Environment(fn.env)
            call_env.set(fn.param, args[0])
            saved = self.env
            self.env = call_env
            self._call_depth += 1
            try:
                result = self.visit(fn.body_ctx)
            finally:
                self.env = saved
                self._call_depth -= 1
            return result
        if isinstance(fn, TrezFunction):
            return self._call_function(fn, args)
        raise TrezRuntimeError(
            f"El operador |> requiere una función en el lado derecho, recibió {type(fn).__name__}."
        )

    # ── expressions ──────────────────────────────────────────────────────────

    def visitNumExpr(self, ctx: TrezParser.NumExprContext):
        val = ctx.getText()
        return float(val) if '.' in val else int(val)

    def visitStringExpr(self, ctx: TrezParser.StringExprContext):
        raw = ctx.getText()[1:-1]
        raw = raw.replace('\\n', '\n').replace('\\t', '\t')
        raw = raw.replace('\\"', '"').replace('\\\\', '\\')
        return raw

    def visitBoolExpr(self, ctx: TrezParser.BoolExprContext):
        return ctx.getText() == 'true'

    def visitVarExpr(self, ctx: TrezParser.VarExprContext):
        name = ctx.ID().getText()
        if hasattr(math_utilsdoz, 'constants') and name in math_utilsdoz.constants:
            return math_utilsdoz.constants[name]
        return self.env.get(name)

    def visitParenExpr(self, ctx: TrezParser.ParenExprContext):
        return self.visit(ctx.rhs())

    def visitPostfixExpr(self, ctx: TrezParser.PostfixExprContext):
        return self.visit(ctx.postfix())

    def visitAtomExpr(self, ctx: TrezParser.AtomExprContext):
        return self.visit(ctx.atom())

    def visitArrayExpr(self, ctx: TrezParser.ArrayExprContext):
        return self.visit(ctx.array())

    def visitArray(self, ctx: TrezParser.ArrayContext):
        return [self.visit(r) for r in ctx.rhs()]

    def visitDictExpr(self, ctx: TrezParser.DictExprContext):
        return self.visit(ctx.dict_())

    def visitDict(self, ctx: TrezParser.DictContext):
        result = {}
        for entry in ctx.dict_entry():
            key_token = entry.getChild(0).getText()
            key = key_token[1:-1] if key_token.startswith('"') else key_token
            result[key] = self.visit(entry.rhs())
        return result

    # ── operators ────────────────────────────────────────────────────────────

    def visitNotExpr(self, ctx: TrezParser.NotExprContext):
        return not bool(self.visit(ctx.expr()))

    def visitAndExpr(self, ctx: TrezParser.AndExprContext):
        left = self.visit(ctx.expr(0))
        if not bool(left):
            return False
        return bool(self.visit(ctx.expr(1)))

    def visitOrExpr(self, ctx: TrezParser.OrExprContext):
        left = self.visit(ctx.expr(0))
        if bool(left):
            return True
        return bool(self.visit(ctx.expr(1)))

    def visitEqExpr(self, ctx: TrezParser.EqExprContext):
        left  = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        return left == right if ctx.getChild(1).getText() == '==' else left != right

    def visitCompareExpr(self, ctx: TrezParser.CompareExprContext):
        left  = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        if op == '<':  return left < right
        if op == '<=': return left <= right
        if op == '>':  return left > right
        return left >= right

    def visitAddSubExpr(self, ctx: TrezParser.AddSubExprContext):
        left  = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                raise TrezRuntimeError("Listas de distinto tamaño en suma/resta.")
            if op == '+':
                return [l + r for l, r in zip(left, right)]
            return [l - r for l, r in zip(left, right)]
        if op == '+':
            return left + right
        return left - right

    def visitMulDivExpr(self, ctx: TrezParser.MulDivExprContext):
        left  = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        if op == '*':
            if isinstance(left, (int, float)) and isinstance(right, list):
                return [left * r for r in right]
            if isinstance(left, list) and isinstance(right, (int, float)):
                return [l * right for l in left]
            return left * right
        if op == '/':
            if right == 0:
                raise TrezRuntimeError("División por cero.")
            return left / right
        return left % right

    def visitPowExpr(self, ctx: TrezParser.PowExprContext):
        return math_utilsdoz.pow_doz(self.visit(ctx.expr(0)), self.visit(ctx.expr(1)))

    def visitUnaryMinusExpr(self, ctx: TrezParser.UnaryMinusExprContext):
        val = self.visit(ctx.expr())
        if isinstance(val, list):
            return [-v for v in val]
        return -val

    # ── postfix ───────────────────────────────────────────────────────────────

    def visitIndexExpr(self, ctx: TrezParser.IndexExprContext):
        obj = self.visit(ctx.postfix())
        idx = self.visit(ctx.expr())
        if isinstance(obj, (list, str)):
            if not isinstance(idx, int):
                idx = int(idx)
            if idx < 0 or idx >= len(obj):
                raise TrezRuntimeError(f"Índice {idx} fuera de rango (tamaño {len(obj)}).")
            return obj[idx]
        if isinstance(obj, dict):
            key = idx if isinstance(idx, str) else str(idx)
            if key not in obj:
                raise TrezRuntimeError(f"Clave '{key}' no existe en el diccionario.")
            return obj[key]
        raise TrezRuntimeError("El operador [] requiere lista, texto o diccionario.")

    def visitMethodCallExpr(self, ctx: TrezParser.MethodCallExprContext):
        method      = ctx.ID().getText()
        module_name = ctx.postfix().getText()
        args        = [self.visit(r) for r in ctx.rhs()]

        # Namespace dispatch — check module name BEFORE evaluating postfix
        if module_name in _NAMESPACES:
            ns = _NAMESPACES[module_name]
            if method not in ns:
                raise TrezRuntimeError(
                    f"El módulo '{module_name}' no tiene la función '{method}'."
                )
            return ns[method](args)

        obj = self.visit(ctx.postfix())
        return self._dispatch_method(obj, method, args)

    def _dispatch_method(self, obj, method, args):
        # ── list ──
        if isinstance(obj, list):
            if method == 'append':   return obj + [args[0]]
            if method == 'head':
                if not obj: raise TrezRuntimeError("head() en lista vacía.")
                return obj[0]
            if method == 'tail':     return obj[1:]
            if method == 'len':      return len(obj)
            if method == 'contains': return args[0] in obj
            if method == 'reverse':  return obj[::-1]
            if method == 'get':
                idx = int(args[0])
                if idx < 0 or idx >= len(obj):
                    raise TrezRuntimeError(f"Índice {idx} fuera de rango.")
                return obj[idx]
            if method == 'slice':
                return obj[int(args[0]):int(args[1])]
            raise TrezRuntimeError(f"Lista no tiene método '{method}'.")

        # ── dict ──
        if isinstance(obj, dict):
            if method == 'get':
                key = args[0] if isinstance(args[0], str) else str(args[0])
                if key not in obj: raise TrezRuntimeError(f"Clave '{key}' no existe.")
                return obj[key]
            if method == 'keys':   return list(obj.keys())
            if method == 'values': return list(obj.values())
            if method == 'has':
                key = args[0] if isinstance(args[0], str) else str(args[0])
                return key in obj
            if method == 'set':
                key = args[0] if isinstance(args[0], str) else str(args[0])
                new_d = dict(obj)
                new_d[key] = args[1]
                return new_d
            raise TrezRuntimeError(f"Diccionario no tiene método '{method}'.")

        # ── Queue ──
        if isinstance(obj, TrezQueue):
            if method == 'enqueue': return obj.enqueue(args[0])
            if method == 'dequeue':
                val, new_q = obj.dequeue()
                return [val, new_q]
            if method == 'peek':    return obj.peek()
            if method == 'isEmpty': return obj.is_empty()
            if method == 'size':    return obj.size()
            if method == 'toList':  return obj.to_list()
            raise TrezRuntimeError(f"Queue no tiene método '{method}'.")

        # ── Stack ──
        if isinstance(obj, TrezStack):
            if method == 'push':    return obj.push(args[0])
            if method == 'pop':
                val, new_s = obj.pop()
                return [val, new_s]
            if method == 'peek':    return obj.peek()
            if method == 'isEmpty': return obj.is_empty()
            if method == 'size':    return obj.size()
            if method == 'toList':  return obj.to_list()
            raise TrezRuntimeError(f"Stack no tiene método '{method}'.")

        # ── string ──
        if isinstance(obj, str):
            if method == 'len':      return len(obj)
            if method == 'contains': return args[0] in obj
            raise TrezRuntimeError(f"Texto no tiene método '{method}'.")

        raise TrezRuntimeError(f"El objeto no soporta métodos (tipo: {type(obj).__name__}).")

    # ── function call ─────────────────────────────────────────────────────────

    def visitFuncCallExpr(self, ctx: TrezParser.FuncCallExprContext):
        func_name = ctx.ID().getText()
        args = [self.visit(r) for r in ctx.rhs()]

        # ── I/O ──
        if func_name == 'leer':     return read_file_doz(args[0])
        if func_name == 'escribir': return write_file_doz(args[0], args[1])
        if func_name in ('mostrar', 'print'):
            print(args[0])
            return None

        # ── structures ──
        if func_name == 'Queue': return make_queue()
        if func_name == 'Stack': return make_stack()

        # ── collection utils ──
        if func_name == 'len':
            obj = args[0]
            if isinstance(obj, (list, str)):              return len(obj)
            if isinstance(obj, (TrezQueue, TrezStack)):   return obj.size()
            raise TrezRuntimeError("len() requiere lista, texto o estructura.")
        if func_name == 'append':
            if not isinstance(args[0], list): raise TrezRuntimeError("append() requiere lista.")
            return args[0] + [args[1]]
        if func_name == 'head':
            if not isinstance(args[0], list) or not args[0]:
                raise TrezRuntimeError("head() requiere lista no vacía.")
            return args[0][0]
        if func_name == 'tail':
            if not isinstance(args[0], list): raise TrezRuntimeError("tail() requiere lista.")
            return args[0][1:]
        if func_name == 'keys':
            if not isinstance(args[0], dict): raise TrezRuntimeError("keys() requiere diccionario.")
            return list(args[0].keys())
        if func_name == 'values':
            if not isinstance(args[0], dict): raise TrezRuntimeError("values() requiere diccionario.")
            return list(args[0].values())
        if func_name == 'range':
            if len(args) == 1: return list(_range_doz(0, int(args[0]), 1))
            if len(args) == 2: return list(_range_doz(int(args[0]), int(args[1]), 1))
            return list(_range_doz(int(args[0]), int(args[1]), int(args[2])))
        if func_name == 'str': return str(args[0])
        if func_name == 'int': return int(args[0])
        if func_name == 'num':
            try:
                v = float(str(args[0]))
                return int(v) if v == int(v) else v
            except Exception:
                raise TrezRuntimeError(f"num() no pudo convertir '{args[0]}' a número.")

        # ── math ──
        math_map = {
            'relu':      lambda: math_utilsdoz.relu(args[0]),
            'sigmoid':   lambda: math_utilsdoz.sigmoid(args[0]),
            'dot':       lambda: math_utilsdoz.dot(args[0], args[1]),
            'transpose': lambda: math_utilsdoz.transpose(args[0]),
            'mse':       lambda: math_utilsdoz.mse(args[0], args[1]),
            'mse_grad':  lambda: math_utilsdoz.mse_grad(args[0], args[1]),
            'abs':       lambda: math_utilsdoz.abs_doz(args[0]),
            'sqrt':      lambda: math_utilsdoz.sqrt_doz(args[0]),
            'pow':       lambda: math_utilsdoz.pow_doz(args[0], args[1]),
            'exp':       lambda: math_utilsdoz.exp_doz(args[0]),
            'log':       lambda: math_utilsdoz.log_doz(args[0]),
            'sin':       lambda: math_utilsdoz.sin_doz(args[0]),
            'cos':       lambda: math_utilsdoz.cos_doz(args[0]),
            'tan':       lambda: math_utilsdoz.tan_doz(args[0]),
            'factorial': lambda: math_utilsdoz.factorial_doz(args[0]),
        }
        if func_name in math_map:
            return math_map[func_name]()

        # ── user-defined function, lambda, or builtin ──
        fn = self.env.get(func_name)
        if isinstance(fn, (TrezFunction, TrezLambda, TrezBuiltin)):
            return self._apply(fn, args)
        raise TrezRuntimeError(f"'{func_name}' no es una función.")

    def _call_function(self, fn: TrezFunction, args):
        if len(args) != len(fn.params):
            raise TrezRuntimeError(
                f"'{fn.name}' espera {len(fn.params)} arg(s), recibió {len(args)}."
            )
        call_env = Environment(fn.env)
        for param, val in zip(fn.params, args):
            call_env.set(param, val)
        call_env.set(fn.name, fn)  # self-reference for recursion
        saved = self.env
        self.env = call_env
        self._call_depth += 1
        result = None
        try:
            for stmt in fn.body_ctx.statement():
                self.visit(stmt)
        except ReturnSignal as r:
            result = r.value
        finally:
            self.env = saved
            self._call_depth -= 1
        return result

    def _apply(self, fn, args):
        """Unified call for TrezFunction, TrezLambda, and TrezBuiltin."""
        if isinstance(fn, TrezBuiltin):
            return fn.fn(*args)
        if isinstance(fn, TrezFunction):
            return self._call_function(fn, args)
        if isinstance(fn, TrezLambda):
            call_env = Environment(fn.env)
            call_env.set(fn.param, args[0])
            saved = self.env
            self.env = call_env
            self._call_depth += 1
            try:
                result = self.visit(fn.body_ctx)
            finally:
                self.env = saved
                self._call_depth -= 1
            return result
        raise TrezRuntimeError(
            f"|> requiere una función en el lado derecho, recibió {type(fn).__name__}."
        )


# ── native range (zero external deps) ────────────────────────────────────────

def _range_doz(start, stop, step):
    if step == 0:
        raise TrezRuntimeError("range() step no puede ser 0.")
    i = start
    while (step > 0 and i < stop) or (step < 0 and i > stop):
        yield i
        i += step
