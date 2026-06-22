import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, classification_report,
                             confusion_matrix)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import shap
import warnings
from scipy import stats
from scipy.stats import norm
from itertools import combinations
import logging
from datetime import datetime
import joblib  # For model saving
import json  # For metadata saving
from tqdm import tqdm  # Progress bar for brute force
from statsmodels.stats.proportion import proportion_confint
import random
from sklearn.utils.class_weight import compute_class_weight
import scipy.stats as stats
from scipy.stats import kruskal  # Kruskal-Wallis H test
from sklearn.preprocessing import LabelEncoder
# 新增MICE插补所需库
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge  # MICE默认基模型（适配临床数据）
from sklearn.metrics import mean_squared_error, mean_absolute_error
# 新增LIME解释器所需库
import lime
import lime.lime_tabular
import argparse  # 新增：命令行参数解析
from imblearn.over_sampling import SMOTE
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)  # 屏蔽IterativeImputer警告
warnings.filterwarnings('ignore')
import torch
from statsmodels.stats.multitest import multipletests
# 新增校准曲线所需库
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import matplotlib.patches as mpatches  # 用于自定义图例
from matplotlib.ticker import MaxNLocator  # 坐标轴整数刻度
import textwrap  # 特征名换行（避免过长）


# ====================== GPU Environment Detection & Configuration =======================
def check_gpu_availability(use_gpu=True):
    """Detect GPU availability"""
    if use_gpu and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU Environment Detected!")
        logger.info(f"Available GPUs: {gpu_count}")
        logger.info(f"GPU Model: {gpu_name}")
        return True
    else:
        logger.warning("=" * 50)
        logger.warning("No Available GPU Detected, Using CPU Instead")
        logger.warning("Recommend Installing CUDA/cuDNN for GPU Acceleration")
        logger.warning("=" * 50)
        return False


# ====================== Core Configuration =======================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Autoimmune Liver Disease Diagnostic Model Pipeline (Binary: early/late)')

    # 基本配置
    parser.add_argument('--excel_path', type=str, default="./Stage_metabolism_analysis_updated.xlsx",
                        help='Excel文件路径')
    parser.add_argument('--n_selected', type=int, default=7,
                        help='最终选择的特征数量')
    parser.add_argument('--is_filter', type=bool, default=True,
                        help='是否进行特征过滤')

    parser.add_argument('--save_root', type=str, default="stage analysis 2 7 features")
    parser.add_argument('--figure_dir', type=str, default="figure_3",
                        help='图表保存目录')
    parser.add_argument('--results_dir', type=str, default="results_3",
                        help='结果保存目录')
    parser.add_argument('--model_dir', type=str, default="saved_best_model",
                        help='模型保存目录')
    parser.add_argument('--filter_dir', type=str, default="filtered_lipid_data",
                        help='过滤数据保存目录')

    # 降维配置
    parser.add_argument('--dimension_reduction_method', type=str, default='none',
                        choices=['none', 'pca', 'selectkbest'],
                        help='降维方法')
    parser.add_argument('--pca_variance_ratio', type=float, default=0.8,
                        help='PCA保留方差比例')
    parser.add_argument('--selectkbest_k', type=int, default=100,
                        help='SelectKBest选择的特征数量')

    # 特征选择配置
    parser.add_argument('--feature_selection_method', type=str, default='brute_force',
                        choices=['specified', 'brute_force', 'shap', 'lasso', 'sis'],
                        help='特征选择方法')
    parser.add_argument('--specified_features', type=str, nargs='+',
                        default=['TBIL', 'Hypoxanthine', 'PE(P-18:0/20:4)',
                                 'IgG', 'ALT','ALP(35-100)',
                                 'IgM', 'AST','GGT(4-50)','ANA','AMA-M2'],
                        help='指定特征列表')

    # 暴力搜索配置
    parser.add_argument('--brute_force_max_combinations', type=int, default=2000,
                        help='暴力搜索最大组合数')

    # SIS配置
    parser.add_argument('--sis_score_func', type=str, default='f_classif',
                        help='SIS评分函数')
    parser.add_argument('--sis_k_method', type=str, default='n_log',
                        choices=['n_log', 'sqrt', 'fixed'],
                        help='SIS k值计算方法')
    parser.add_argument('--sis_fixed_k', type=int, default=50,
                        help='SIS固定k值')

    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='是否使用GPU')
    parser.add_argument('--gpu_batch_size', type=int, default=64,
                        help='GPU批处理大小')

    # 特征类型过滤配置
    parser.add_argument('--feature_type_filter', type=str, default='all',
                        choices=['lipid_only', 'bile_acid_only', 'clinical_only',
                                 'lipid_bile_acid', 'lipid_clinical', 'bile_acid_clinical', 'all'],
                        help='特征类型过滤')

    # 分组标准化配置
    parser.add_argument('--group_std_method', type=str, default='auto_scaling',
                        choices=['auto_scaling', 'median_centering', 'hybrid'],
                        help='分组标准化方法')

    # 插补方法配置
    parser.add_argument('--imputation_method', type=str, default='comparison',
                        choices=['clinical_logic', 'mice', 'comparison'],
                        help='缺失值插补方法')
    parser.add_argument('--mice_n_iter', type=int, default=10,
                        help='MICE迭代次数')
    parser.add_argument('--mice_random_state', type=int, default=42,
                        help='MICE随机种子')

    # LIME配置
    parser.add_argument('--lime_n_samples', type=int, default=5000,
                        help='LIME采样数')
    parser.add_argument('--lime_n_features', type=int, default=5,
                        help='LIME解释的特征数')
    parser.add_argument('--lime_random_state', type=int, default=42,
                        help='LIME随机种子')

    # 输出配置
    parser.add_argument('--save_filtered_data', type=bool, default=True,
                        help='是否保存过滤后的数据')

    parser.add_argument('--save_format', type=str, default="csv",
                        choices=['npy', 'csv'],
                        help='保存格式')

    # 日志配置
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')

    return parser.parse_args()


# 临床阈值配置（固定不变）
CLINICAL_THRESHOLDS = {
    'ALT': 40,  # Upper limit of normal: 40 U/L
    'AST': 40,  # Upper limit of normal: 40 U/L
    'ALP(35-100)': 100,  # Upper limit of normal: 100 U/L
    'GGT(4-50)': 50,  # Upper limit of normal: 50 U/L
    'TBIL': 19,  # Upper limit of normal: 19 μmol/L
    'DBIL': 6.8,  # Upper limit of normal: 6.8 μmol/L
    'ALB': 35,  # Lower limit of normal: 35 g/L
    'GLO': 30,  # Lower limit of normal: 30 g/L
    'TBA': 10,  # Upper limit of normal: 10 μmol/L
    'IgG': 16,  # Upper limit of normal: 16 g/L (AIH core indicator)
    'IgM': 2.3,  # Upper limit of normal: 2.3 g/L (PBC core indicator)
    'BMI': 24,  # Upper limit of normal: 24 kg/m²
    'Age': 60  # Reference value: 60 years old
}

# 特征类型定义（临床逻辑，固定不变）
ANTIBODY_FEATURES = [
    'AMA-M2', 'AMA', 'ANA', 'Anti-Sp100', 'Anti-Gp210',
    'Anti-LKM-1', 'Anti-SLA/LP'
]

LIVER_FUNCTION_FEATURES = [
    'ALT', 'AST', 'ALP(35-100)', 'GGT(4-50)', 'TBIL',
    'IgG', 'IgM'  # 'DBIL','ALB', 'GLO', 'TBA'
]

BASIC_FEATURES = []  # ['Sex', 'Age', 'BMI']
CLINICAL_FEATURES = ANTIBODY_FEATURES + LIVER_FUNCTION_FEATURES + BASIC_FEATURES
BILE_ACID_FEATURES_RANGE = (0, 24)  # 胆汁酸代谢物列范围（前24列）
LIPID_FEATURES_RANGE = (24, 136)  # 脂质代谢物列范围（第24-136列）


# ====================== Logging Configuration ======================
def setup_logging(args, dimension_reduction_method, feature_selection_method):
    # 创建输出目录
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(os.path.join(args.save_root, args.figure_dir), exist_ok=True)
    os.makedirs(os.path.join(args.save_root, args.results_dir), exist_ok=True)
    os.makedirs(os.path.join(args.save_root, args.model_dir), exist_ok=True)
    os.makedirs(os.path.join(args.save_root, args.filter_dir), exist_ok=True)
    os.makedirs('results_3', exist_ok=True)
    log_filename = os.path.join(os.path.join(args.save_root, args.results_dir),
                                f'run_log_{dimension_reduction_method}_{feature_selection_method}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_binary.log')

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ====================== Journal-Level Plot Configuration (固定不变) =======================
plt.style.use('default')
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans", "sans-serif"]
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['axes.unicode_minus'] = False
# ---------------------- 关键：PDF文字可编辑（核心参数） ----------------------
plt.rcParams['pdf.fonttype']= 42  # 42=TrueType字体（可编辑），默认3=轮廓（不可编辑）
plt.rcParams['pdf.use14corefonts']=False  # 不使用PDF核心字体，避免强制转曲
    # ---------------------- 关键：SVG文字可编辑（AI最优格式） ----------------------
plt.rcParams['svg.fonttype']='none'  # 文字以文本对象保存（不转曲）
plt.rcParams['axes.unicode_minus']=False  # 避免负号显示为方块，兼容特殊字符

COLOR_PALETTE = {
    'train': '#2E86AB', 'val': '#A23B72', 'cv': '#F18F01',
    'class1': '#E64B35', 'class2': '#4DBBD5', 'class3': '#00A087', 'class4': '#3C5488',
    'shap': '#7E6148', 'grid': '#E0E0E0',
    'lipid': '#2E86AB', 'bile_acid': '#A23B72', 'clinical': '#F18F01',
    'line': '#E64B35', 'heatmap': 'YlGnBu',
    'AMA-M2_1': '#1f77b4',  # AMA-M2阳性
    'AMA-M2_0': '#ff7f0e'  # AMA-M2阴性
}


# ====================== Delong Test Implementation (二分类适配，固定核心逻辑) =======================
class DelongTest:
    """Delong test for AUC significance comparison (binary classification) - 适配二分类"""

    def __init__(self):
        pass

    @staticmethod
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T + 1
        return T2

    @staticmethod
    def fastDeLong(predictions_sorted_transposed, label_1_count):
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)
        for r in range(k):
            tx[r, :] = DelongTest.compute_midrank(positive_examples[r, :])
            ty[r, :] = DelongTest.compute_midrank(negative_examples[r, :])
            tz[r, :] = DelongTest.compute_midrank(predictions_sorted_transposed[r, :])

        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n

        z = (aucs[0] - aucs[1]) / np.sqrt(delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1])
        p_value = 2 * (1 - norm.cdf(abs(z)))
        return z, p_value, aucs

    def compare(self, y_true, y_score1, y_score2):
        assert len(y_true) == len(y_score1) == len(y_score2), "Input array lengths must match"
        assert len(np.unique(y_true)) == 2, "Delong test only supports binary classification"

        pos_idx = y_true == 1
        neg_idx = y_true == 0
        label_1_count = int(np.sum(pos_idx))

        def sort_by_score(y_score, pos_idx, neg_idx):
            pos_scores = y_score[pos_idx]
            pos_sorted = pos_scores[np.argsort(pos_scores)[::-1]]
            neg_scores = y_score[neg_idx]
            neg_sorted = neg_scores[np.argsort(neg_scores)[::-1]]
            return np.hstack([pos_sorted, neg_sorted])

        sorted_score1 = sort_by_score(y_score1, pos_idx, neg_idx)
        sorted_score2 = sort_by_score(y_score2, pos_idx, neg_idx)
        predictions_sorted_transposed = np.vstack([sorted_score1, sorted_score2])

        z, p_value, aucs = self.fastDeLong(predictions_sorted_transposed, label_1_count)
        return p_value, aucs[0], aucs[1]

    @staticmethod
    def calculate_net_benefit(y_true, y_pred_prob, threshold):
        """Fixed net benefit calculation (per 100 samples) - 二分类适配"""
        n_total = len(y_true)
        if n_total == 0:
            return 0.0

        y_true_binary = (y_true == 1).astype(int)
        y_pred = (y_pred_prob >= threshold).astype(int)

        tp_rate = np.sum((y_true_binary == 1) & (y_pred == 1)) / n_total
        fp_rate = np.sum((y_true_binary == 0) & (y_pred == 1)) / n_total

        if threshold == 0 or threshold == 1:
            net_benefit = 0.0
        else:
            net_benefit = tp_rate - (fp_rate * (threshold / (1 - threshold)))

        net_benefit *= 100  # Convert to per 100 samples
        return net_benefit

    @staticmethod
    def custom_dca_analysis(y_true, y_pred_prob, thresholds=np.linspace(0.01, 0.99, 99)):
        """Fixed DCA analysis (avoid division by zero) - 二分类适配"""
        dca_results = {
            'threshold': thresholds,
            'model_net_benefit': [],
            'treat_all_net_benefit': [],
            'treat_none_net_benefit': []
        }
        n_total = len(y_true)
        n_positive = np.sum(y_true == 1)
        positive_rate = n_positive / n_total
        negative_rate = 1 - positive_rate

        for threshold in thresholds:
            model_nb = DelongTest.calculate_net_benefit(y_true, y_pred_prob, threshold)

            if threshold == 1:
                treat_all_nb = 0.0
            else:
                treat_all_nb = positive_rate - (negative_rate * (threshold / (1 - threshold)))
                treat_all_nb *= 100

            dca_results['model_net_benefit'].append(model_nb)
            dca_results['treat_all_net_benefit'].append(treat_all_nb)
            dca_results['treat_none_net_benefit'].append(0.0)

        return dca_results


# ====================== AUC CI Calculation (二分类适配) =======================
def calculate_auc_ci(y_true, y_score, n_bootstrap=1000, ci=0.95):
    np.random.seed(42)
    n_samples = len(y_true)

    # 二分类适配：移除多分类相关逻辑，直接处理1D概率数组
    aucs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        if len(np.unique(y_true_boot)) < 2:
            aucs.append(0.5)
            continue
        try:
            auc = roc_auc_score(y_true_boot, y_score_boot)
            aucs.append(auc)
        except:
            aucs.append(0.5)
    aucs = np.array(aucs)

    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    return lower, upper


# ====================== Core Analysis Class (二分类核心修改) =======================
class MedicalDataAnalyzer:
    def __init__(self, args):
        """初始化分析器，从args获取所有配置 - 二分类适配"""
        self.args = args

        self.excel_path = args.excel_path
        self.n_selected = args.n_selected
        self.feature_selection_method = args.feature_selection_method.lower()
        assert self.feature_selection_method in ['specified', 'shap', 'lasso', 'brute_force', 'sis'], \
            "Feature selection method only supports: 'specified'/'shap'/'lasso'/'brute_force'/'sis'"

        self.specified_features = args.specified_features
        if self.feature_selection_method == 'specified':
            assert self.specified_features is not None, "Specified mode requires feature list"
            assert len(self.specified_features) == self.n_selected, \
                f"Specified feature count ({len(self.specified_features)}) must match N_SELECTED ({self.n_selected})"

        self.dimension_reduction_method = args.dimension_reduction_method.lower()
        assert self.dimension_reduction_method in ['none', 'pca',
                                                   'selectkbest'], "Dimension reduction only supports: 'none'/'pca'/'selectkbest'"
        self.pca_variance_ratio = args.pca_variance_ratio
        self.selectkbest_k = args.selectkbest_k

        # Brute force params
        self.brute_force_max_combinations = args.brute_force_max_combinations
        # 基础模型配置 - 二分类适配：XGBoost eval_metric改为'logloss'
        self.brute_force_base_model = XGBClassifier(
            random_state=42, eval_metric='logloss',  # 多分类mlogloss → 二分类logloss
            tree_method='gpu_hist' if args.use_gpu else 'hist',
            gpu_id=0 if args.use_gpu else None,
            n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=1
        )
        self.brute_force_results = None

        # SIS params
        self.sis_score_func = f_classif  # 默认为f_classif
        self.sis_k_method = args.sis_k_method
        self.sis_fixed_k = args.sis_fixed_k
        self.sis_scores = None

        # GPU config
        self.use_gpu = args.use_gpu and check_gpu_availability(args.use_gpu)
        self.gpu_batch_size = args.gpu_batch_size
        logger.info(f"GPU Acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")

        # LIME config
        self.lime_n_samples = args.lime_n_samples
        self.lime_n_features = args.lime_n_features
        self.lime_random_state = args.lime_random_state

        # Feature type filter
        self.feature_type_filter = args.feature_type_filter

        # Group standardization
        self.group_std_method = args.group_std_method

        # Imputation method
        self.imputation_method = args.imputation_method
        self.mice_n_iter = args.mice_n_iter
        self.mice_random_state = args.mice_random_state

        # Save config
        self.save_filtered_data = args.save_filtered_data
        self.save_format = args.save_format

        self.save_root = args.save_root
        self.figure_dir = args.figure_dir
        self.results_dir = args.results_dir
        self.model_dir = args.model_dir
        self.filter_dir = args.filter_dir

        # Data storage
        self.raw_data = None
        self.train_data = None
        self.val_data = None
        self.feature_columns = None
        self.reduced_feature_names = None
        self.y_train = None
        self.y_val = None

        # 二分类核心修改：类别配置
        self.n_classes = 2  # 多分类4 → 二分类2
        self.label_offset = 0  # 移除类别偏移
        self.class_name_mapping = {0: 'early', 1: 'late'}  # 1/2→early(0)，3/4→late(1)
        self.class_names = ['early', 'late']  # 二分类类别名称

        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_train_reduced = None
        self.X_val_reduced = None
        self.top_features = None
        self.models = {}
        self.model_results = {}
        self.cv_results = {}
        self.delong_results = {}
        self.delong_target_class = None
        self.figure_params = {
            'violin': (6, 4), 'small': (3, 3), 'medium': (5, 4), 'large': (8, 5)
        }
        self.delong_test = DelongTest()

        # Preprocessing components
        self.imputer = None
        self.scaler = None
        self.dimension_reducer = None
        # Best model
        self.best_model = None
        self.best_model_name = None
        self.best_model_params = None
        self.best_model_metrics = None

        # Feature type lists
        self.lipid_features = None
        self.bile_acid_features = None
        self.clinical_features = None

    def _merge_labels_to_binary(self, labels):
        """将原标签[1,2,3,4]合并为二分类[0,1]：1/2→early(0)，3/4→late(1)"""
        binary_labels = np.zeros_like(labels, dtype=np.int32)
        # 原标签3、4 → late(1)
        binary_labels[(labels == 3) | (labels == 4)] = 1
        # 原标签1、2 → early(0)（默认已为0，无需额外赋值）
        return binary_labels

    def load_data(self):
        logger.info("Step 1/7: Loading data... (Binary: early/late)")
        self.raw_data = pd.read_excel(self.excel_path, sheet_name="analysis_2_184")
        self.raw_data.rename(columns={
            self.raw_data.columns[0]: 'dataset_type',
            self.raw_data.columns[1]: 'patient_id',
            self.raw_data.columns[2]: 'group_label'
        }, inplace=True)
        self.train_data = self.raw_data[self.raw_data['dataset_type'] == 'Discovery'].copy()
        self.val_data = self.raw_data[self.raw_data['dataset_type'] == 'Validation'].copy()
        self.feature_columns = self.raw_data.columns[3:-7].tolist()

        # 定义特征类型列表
        start_lipid, end_lipid = LIPID_FEATURES_RANGE
        start_bile, end_bile = BILE_ACID_FEATURES_RANGE
        self.lipid_features = self.feature_columns[start_lipid:end_lipid]
        self.bile_acid_features = self.feature_columns[start_bile:end_bile]
        self.clinical_features = [f for f in self.feature_columns if f in CLINICAL_FEATURES]

        # 应用特征类型过滤
        self._apply_feature_type_filter()

        # 二分类核心修改：标签合并
        self.y_train = self._merge_labels_to_binary(self.train_data['group_label'].astype(np.int32).values)
        self.y_val = self._merge_labels_to_binary(self.val_data['group_label'].astype(np.int32).values)

        # 数据校验
        assert set(self.y_train).issubset({0, 1}), f"Invalid training labels after binary merge: {set(self.y_train)}"
        assert set(self.y_val).issubset({0, 1}), f"Invalid validation labels after binary merge: {set(self.y_val)}"

        logger.info(f"Training samples: {len(self.train_data)}, Validation samples: {len(self.val_data)}")
        train_dist = np.bincount(self.y_train)
        val_dist = np.bincount(self.y_val)
        logger.info("Label distribution - Train (Binary: early/late):")
        for i, name in self.class_name_mapping.items():
            logger.info(f"  {name}: {train_dist[i] if i < len(train_dist) else 0}")
        logger.info("Label distribution - Validation (Binary: early/late):")
        for i, name in self.class_name_mapping.items():
            logger.info(f"  {name}: {val_dist[i] if i < len(val_dist) else 0}")
        return self

    def _perform_pairwise_mannwhitney(self, feature_data, class_labels, class_names):
        """
         pairwise Mann-Whitney U检验（非参数检验）+ Benjamini-Hochberg校正
        返回每个类别与对照组（late，标签1）的显著性结果
        """
        ctrl_mask = class_labels == 1  # 二分类：对照组改为late(1)
        if not np.any(ctrl_mask):
            return {}

        ctrl_data = feature_data[ctrl_mask]
        sig_results = {}
        p_values = []
        cls_names_list = []

        for cls_idx, cls_name in enumerate(class_names[:-1]):  # 排除late自身
            cls_mask = class_labels == cls_idx
            if not np.any(cls_mask):
                sig_results[cls_name] = 'ns'
                continue

            cls_data = feature_data[cls_mask]
            # Mann-Whitney U检验
            stat, p_val = stats.mannwhitneyu(cls_data, ctrl_data, alternative='two-sided')
            sig_results[cls_name] = p_val
            p_values.append(p_val)
            cls_names_list.append(cls_name)

        # Benjamini-Hochberg校正（补全前序遗漏逻辑）
        if len(p_values) > 0:
            corrected_p, _, _, _ = multipletests(p_values, method='fdr_bh')
            for i, cls_name in enumerate(cls_names_list):
                p_val = corrected_p[i]
                if p_val < 0.001:
                    sig_results[cls_name] = '***'
                elif p_val < 0.01:
                    sig_results[cls_name] = '**'
                elif p_val < 0.05:
                    sig_results[cls_name] = '*'
                else:
                    sig_results[cls_name] = 'ns'

        return sig_results

    def _mice_imputation(self, df, is_train=True):
        """
        多重插补（MICE）实现（适配临床+代谢组学数据）
        """
        df_imputed = df.copy()

        # Step 1: 抗体特征先按临床逻辑填充0
        antibody_cols = [f for f in ANTIBODY_FEATURES if f in df_imputed.columns]
        df_imputed[antibody_cols] = df_imputed[antibody_cols].fillna(0)
        logger.info(f"MICE pre-impute: antibody features ({len(antibody_cols)}) fillna(0)")

        # Step 2: 分离数值型特征
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [f for f in df_imputed.columns if f not in numeric_cols]
        if non_numeric_cols:
            logger.warning(f"MICE skip non-numeric features: {non_numeric_cols}")

        # Step 3: MICE迭代插补
        X_numeric = df_imputed[numeric_cols].values
        if is_train:
            self.mice_imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.mice_n_iter,
                random_state=self.mice_random_state,
                verbose=0,
                imputation_order='roman',
                skip_complete=True
            )
            X_numeric_imputed = self.mice_imputer.fit_transform(X_numeric)
            logger.info(f"MICE fitted on training data (n_iter={self.mice_n_iter})")
        else:
            X_numeric_imputed = self.mice_imputer.transform(X_numeric)

        # Step 4: 整合插补结果
        df_imputed[numeric_cols] = X_numeric_imputed

        # GPU优化
        if self.use_gpu:
            for col in numeric_cols:
                df_imputed[col] = df_imputed[col].astype(np.float32)

        logger.info(f"MICE imputation completed (handled {len(numeric_cols)} numeric features)")
        return df_imputed

    def _compare_imputation_methods(self, X_train_df, X_val_df):
        """
        对比临床逻辑插补 vs MICE插补的效果
        """
        logger.info("\n=== Comparing Imputation Methods (Clinical Logic vs MICE) ===")

        # 1. 两种方法插补
        X_train_clinical = self._clinical_guided_imputation(X_train_df.copy(), is_train=True)
        X_train_mice = self._mice_imputation(X_train_df.copy(), is_train=True)

        # 2. 仅选择有缺失值的特征
        missing_cols = [col for col in X_train_df.columns if X_train_df[col].isnull().sum() > 0]
        if not missing_cols:
            logger.warning("No missing values in training data, skip imputation comparison")
            return "clinical_logic"

        # 3. 量化对比
        comparison_results = []
        for col in missing_cols:
            original_non_missing = X_train_df[col].dropna().values
            clinical_imputed = X_train_clinical[col].values
            mice_imputed = X_train_mice[col].values

            # KS检验
            if len(original_non_missing) > 20:
                ks_clinical, p_clinical = stats.ks_2samp(original_non_missing, clinical_imputed)
                ks_mice, p_mice = stats.ks_2samp(original_non_missing, mice_imputed)
            else:
                ks_clinical = ks_mice = p_clinical = p_mice = np.nan

            # 方差保留率
            var_original = np.var(original_non_missing)
            var_clinical = np.var(clinical_imputed) if var_original != 0 else 0
            var_mice = np.var(mice_imputed) if var_original != 0 else 0
            var_retention_clinical = var_clinical / var_original if var_original != 0 else np.nan
            var_retention_mice = var_mice / var_original if var_original != 0 else np.nan

            comparison_results.append({
                'Feature': col,
                'Missing_Count': X_train_df[col].isnull().sum(),
                'KS_Clinical': round(ks_clinical, 4),
                'P_Value_Clinical': round(p_clinical, 4),
                'KS_MICE': round(ks_mice, 4),
                'P_Value_MICE': round(p_mice, 4),
                'Variance_Retention_Clinical': round(var_retention_clinical, 4),
                'Variance_Retention_MICE': round(var_retention_mice, 4),
                'Better_Method': 'clinical_logic' if (
                    p_clinical >= p_mice and var_retention_clinical >= var_retention_mice) else 'mice'
            })

        # 4. 保存对比结果
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv(os.path.join(self.save_root, self.results_dir,
                                          f'imputation_comparison_{self.feature_type_filter}.csv'),
                             index=False, encoding='utf-8-sig'
                             )

        # 5. 统计整体最优方法
        method_counts = comparison_df['Better_Method'].value_counts()
        best_method = method_counts.idxmax() if not method_counts.empty else 'clinical_logic'
        logger.info(f"\nImputation Comparison Summary:")
        logger.info(f"  - Total compared features: {len(missing_cols)}")
        logger.info(f"  - Clinical logic better: {method_counts.get('clinical_logic', 0)} features")
        logger.info(f"  - MICE better: {method_counts.get('mice', 0)} features")
        logger.info(f"  - Selected best imputation method: {best_method}")

        # 6. 可视化对比
        self._plot_imputation_comparison(X_train_df, X_train_clinical, X_train_mice, missing_cols[:5])

        return best_method

    def _plot_imputation_comparison(self, df_original, df_clinical, df_mice, features):
        """
        可视化插补效果
        """
        if not features:
            return

        logger.info(f"Plotting imputation comparison for features: {features}")
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=(8, 4 * n_features))
        if n_features == 1:
            axes = [axes]

        for idx, feature in enumerate(features):
            ax = axes[idx]
            original = df_original[feature].dropna()
            clinical = df_clinical[feature]
            mice = df_mice[feature]

            sns.kdeplot(original, ax=ax, label='Original (Non-missing)', color='black', linewidth=2, alpha=0.8)
            sns.kdeplot(clinical, ax=ax, label='Clinical Logic Imputation', color=COLOR_PALETTE['train'], linewidth=1.5,
                        linestyle='--', alpha=0.8)
            sns.kdeplot(mice, ax=ax, label='MICE Imputation', color=COLOR_PALETTE['val'], linewidth=1.5, linestyle=':',
                        alpha=0.8)

            missing_count = df_original[feature].isnull().sum()
            ax.set_title(f'{feature} (Missing: {missing_count})', fontsize=10)
            ax.set_xlabel('Feature Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.suptitle(f'Imputation Method Comparison (Feature Filter: {self.feature_type_filter})', fontsize=12, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir, f'imputation_comparison_{self.feature_type_filter}.png'),
            dpi=600,
            bbox_inches='tight'
            )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir, f'imputation_comparison_{self.feature_type_filter}.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )

        plt.close()

    def _extract_feature_importance(self, model, model_name, X_train_top):
        """
        提取模型特征重要性（兼容所有训练模型，二分类适配）
        :param model: 训练完成的模型实例
        :param model_name: 模型名称（用于日志和分支判断）
        :param X_train_top: 筛选后的训练集特征（n_samples, n_selected）
        :return: 特征重要性数组（np.array）
        """
        logger.info(f"Extracting feature importance for {model_name}...")
        n_features = X_train_top.shape[1]
        feature_importance = np.zeros(n_features, dtype=np.float64)

        try:
            # 分支1：XGBoost/RandomForest（有直接的feature_importances_属性）
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_.astype(np.float64)
                logger.info(f"{model_name} - Feature importance extracted via 'feature_importances_'")

            # 分支2：LogisticRegression（二分类：用系数绝对值作为重要性）
            elif isinstance(model, LogisticRegression):
                if model.coef_.ndim == 2:
                    # 二分类仅返回1组系数，取绝对值作为重要性
                    feature_importance = np.abs(model.coef_[0]).astype(np.float64)
                else:
                    feature_importance = np.abs(model.coef_).astype(np.float64)
                logger.info(f"{model_name} - Feature importance extracted via coef_ (absolute value)")

            # 分支3：SVC（线性核：用coef_绝对值；非线性核：提示无法直接提取，返回0数组+警告）
            elif isinstance(model, SVC):
                if model.kernel == 'linear' and hasattr(model, 'coef_'):
                    if model.coef_.ndim == 2:
                        feature_importance = np.abs(model.coef_[0]).astype(np.float64)
                    else:
                        feature_importance = np.abs(model.coef_).astype(np.float64)
                    logger.info(
                        f"{model_name} (linear kernel) - Feature importance extracted via coef_ (absolute value)")
                else:
                    logger.warning(
                        f"{model_name} (non-linear kernel) - Cannot extract direct feature importance, return zero array")
                    feature_importance = np.ones(n_features, dtype=np.float64) * 0.01  # 避免全零，不影响绘图

            # 分支4：未知模型（容错处理）
            else:
                logger.warning(
                    f"{model_name} - Unsupported model type for feature importance extraction, return zero array")
                feature_importance = np.ones(n_features, dtype=np.float64) * 0.001

            # 归一化处理（使重要性之和为1，提升可读性）
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
            else:
                feature_importance = np.ones(n_features, dtype=np.float64) / n_features  # 均匀分布兜底

            return feature_importance

        except Exception as e:
            logger.error(f"Failed to extract feature importance for {model_name}: {str(e)}", exc_info=True)
            # 兜底返回均匀分布的重要性，避免后续绘图报错
            return np.ones(n_features, dtype=np.float64) / n_features

    def _apply_feature_type_filter(self):
        """Filter features based on FEATURE_TYPE_FILTER"""
        logger.info(f"Applying feature type filter: {self.feature_type_filter}")

        if self.feature_type_filter == 'lipid_only':
            filtered_features = self.lipid_features
        elif self.feature_type_filter == 'bile_acid_only':
            filtered_features = self.bile_acid_features
        elif self.feature_type_filter == 'clinical_only':
            filtered_features = self.clinical_features
        elif self.feature_type_filter == 'lipid_bile_acid':
            filtered_features = self.lipid_features + self.bile_acid_features
        elif self.feature_type_filter == 'lipid_clinical':
            filtered_features = self.lipid_features + self.clinical_features
        elif self.feature_type_filter == 'bile_acid_clinical':
            filtered_features = self.bile_acid_features + self.clinical_features
        elif self.feature_type_filter == 'all':
            filtered_features = self.lipid_features + self.bile_acid_features + self.clinical_features
        else:
            raise ValueError(f"Invalid feature type filter: {self.feature_type_filter}")

        filtered_features = list(dict.fromkeys(filtered_features))
        self.feature_columns = [f for f in filtered_features if f in self.feature_columns]

        logger.info(f"Filtered feature count: {len(self.feature_columns)}")
        logger.info(f"Filtered feature type: {self.feature_type_filter} (first 5): {self.feature_columns[:5]}")

    def _clinical_guided_imputation(self, df, is_train=True):
        """
        Clinical logic-based missing value imputation
        """
        df_imputed = df.copy()

        # 1. Antibody features (missing = 0)
        antibody_cols = [f for f in ANTIBODY_FEATURES if f in df_imputed.columns]
        df_imputed[antibody_cols] = df_imputed[antibody_cols].fillna(0)
        logger.info(f"Imputed antibody features ({len(antibody_cols)}): fillna(0)")

        # 2. Liver function features (missing = disease group median)
        liver_cols = [f for f in LIVER_FUNCTION_FEATURES if f in df_imputed.columns]
        if is_train:
            # 二分类适配：按合并后的标签分组计算中位数
            train_labels_binary = self._merge_labels_to_binary(self.train_data['group_label'].astype(np.int32).values)
            self.liver_group_median = df_imputed.groupby(train_labels_binary)[liver_cols].median()
        else:
            # 验证集按二分类标签分组填充
            val_labels_binary = self._merge_labels_to_binary(self.val_data['group_label'].astype(np.int32).values)
            for col in liver_cols:
                for group in self.liver_group_median.index:
                    mask = val_labels_binary == group
                    df_imputed.loc[mask, col] = df_imputed.loc[mask, col].fillna(
                        self.liver_group_median.loc[group, col])
        logger.info(f"Imputed liver function features ({len(liver_cols)}): group median (binary)")

        # 3. Basic features
        basic_cols = [f for f in BASIC_FEATURES if f in df_imputed.columns]
        for col in basic_cols:
            if col == 'Sex':
                mode_val = df_imputed[col].mode()[0] if is_train else self.sex_mode
                df_imputed[col] = df_imputed[col].fillna(mode_val)
                if is_train:
                    self.sex_mode = mode_val
            else:
                median_val = df_imputed[col].median() if is_train else self.basic_median[col]
                df_imputed[col] = df_imputed[col].fillna(median_val)
                if is_train:
                    if not hasattr(self, 'basic_median'):
                        self.basic_median = {}
                    self.basic_median[col] = median_val
        logger.info(f"Imputed basic features ({len(basic_cols)}): Sex=mode, Age/BMI=global median")

        # 4. Other features (lipid/bile acid): global median
        other_cols = [f for f in df_imputed.columns if f not in antibody_cols + liver_cols + basic_cols]
        if is_train:
            self.other_median = df_imputed[other_cols].median()
        df_imputed[other_cols] = df_imputed[other_cols].fillna(self.other_median)
        logger.info(f"Imputed other features ({len(other_cols)}): global median")

        return df_imputed

    def _median_centering(self, data, train_stats=None):
        """中位数中心化"""
        if train_stats is None:
            median = data.median(axis=0)
            return data - median, median
        else:
            scaled_data = data - train_stats
            return scaled_data, train_stats

    def _auto_scaling(self, data, train_stats=None):
        """自动标度化（Z-score）"""
        if train_stats is None:
            mean = data.mean(axis=0)
            std = data.std(axis=0) + 1e-8
            return (data - mean) / std, (mean, std)
        else:
            mean, std = train_stats
            scaled_data = (data - mean) / std
            return scaled_data, train_stats

    def _group_standardization(self, df, is_train=True):
        """
        脂质/胆汁酸分组标准化核心方法
        """
        # 1. 划分脂质/胆汁酸/临床特征列
        lipid_cols = [f for f in self.lipid_features if f in df.columns]
        bile_acid_cols = [f for f in self.bile_acid_features if f in df.columns]
        clinical_cols = [f for f in self.clinical_features if f in df.columns]

        logger.info(
            f"  - Group standardization cols: Lipid({len(lipid_cols)}) | BileAcid({len(bile_acid_cols)}) | Clinical({len(clinical_cols)})")

        # 2. 初始化标准化后的数据
        df_std = pd.DataFrame(index=df.index)

        # 3. 脂质类标准化
        if len(lipid_cols) > 0:
            if self.group_std_method == 'auto_scaling' or self.group_std_method == 'hybrid':
                lipid_std, self.lipid_stats = self._auto_scaling(df[lipid_cols], None if is_train else self.lipid_stats)
            elif self.group_std_method == 'median_centering':
                lipid_std, self.lipid_stats = self._median_centering(df[lipid_cols],
                                                                     None if is_train else self.lipid_stats)
            df_std[lipid_cols] = lipid_std

        # 4. 胆汁酸类标准化
        if len(bile_acid_cols) > 0:
            if self.group_std_method == 'auto_scaling':
                bile_std, self.bile_acid_stats = self._auto_scaling(df[bile_acid_cols],
                                                                    None if is_train else self.bile_acid_stats)
            elif self.group_std_method == 'median_centering' or self.group_std_method == 'hybrid':
                bile_std, self.bile_acid_stats = self._median_centering(df[bile_acid_cols],
                                                                        None if is_train else self.bile_acid_stats)
            df_std[bile_acid_cols] = bile_std

        # 5. 临床特征：保留原有阈值标准化逻辑
        if len(clinical_cols) > 0:
            df_std[clinical_cols] = self._clinical_threshold_standardization(df[clinical_cols], is_train)

        # 6. 补充未匹配的特征
        other_cols = [f for f in df.columns if f not in lipid_cols + bile_acid_cols + clinical_cols]
        if len(other_cols) > 0:
            df_std[other_cols] = df[other_cols]

        # 7. 保持列顺序
        df_std = df_std[self.feature_columns]

        return df_std

    def _clinical_threshold_standardization(self, df, is_train=True):
        """
        Standardize features using clinical guidelines thresholds
        """
        df_std = df.copy()

        threshold_cols = [f for f in df_std.columns if f in CLINICAL_THRESHOLDS]
        non_threshold_cols = [f for f in df_std.columns if f not in threshold_cols]

        # 1. Clinical threshold-based standardization
        for col in threshold_cols:
            threshold = CLINICAL_THRESHOLDS[col]
            if col in ['ALB', 'GLO']:
                df_std[col] = df_std[col] / threshold
            else:
                df_std[col] = df_std[col] / threshold
        logger.info(f"Clinical threshold standardization for {len(threshold_cols)} features")

        # 2. Z-score for non-threshold features
        if is_train:
            self.non_threshold_mean = df_std[non_threshold_cols].mean()
            self.non_threshold_std = df_std[non_threshold_cols].std() + 1e-8
        df_std[non_threshold_cols] = (df_std[non_threshold_cols] - self.non_threshold_mean) / self.non_threshold_std
        logger.info(f"Z-score standardization for {len(non_threshold_cols)} non-threshold features")

        return df_std

    def preprocess_data(self):
        logger.info("\nStep 2/7: Preprocessing data (Clinical Imputation + MICE + Threshold Standardization)...")
        X_train_df = self.train_data[self.feature_columns]
        X_val_df = self.val_data[self.feature_columns]

        # Step 1: 选择插补方法
        if self.imputation_method == 'clinical_logic':
            logger.info("  Using clinical logic-based imputation...")
            X_train_imputed = self._clinical_guided_imputation(X_train_df, is_train=True)
            X_val_imputed = self._clinical_guided_imputation(X_val_df, is_train=False)
        elif self.imputation_method == 'mice':
            logger.info("  Using MICE imputation (clinical logic pre-impute for antibodies)...")
            X_train_imputed = self._mice_imputation(X_train_df, is_train=True)
            X_val_imputed = self._mice_imputation(X_val_df, is_train=False)
        elif self.imputation_method == 'comparison':
            logger.info("  Comparing clinical logic vs MICE imputation...")
            best_method = self._compare_imputation_methods(X_train_df, X_val_df)
            if best_method == 'clinical_logic':
                X_train_imputed = self._clinical_guided_imputation(X_train_df, is_train=True)
                X_val_imputed = self._clinical_guided_imputation(X_val_df, is_train=False)
            else:
                X_train_imputed = self._mice_imputation(X_train_df, is_train=True)
                X_val_imputed = self._mice_imputation(X_val_df, is_train=False)
        else:
            raise ValueError(
                f"Invalid imputation method: {self.imputation_method} (valid: clinical_logic/mice/comparison)")

        # Step 2: 分组标准化
        logger.info(f"  Performing group-specific standardization (method: {self.group_std_method})...")
        X_train_std = self._group_standardization(X_train_imputed, is_train=True)
        X_val_std = self._group_standardization(X_val_imputed, is_train=False)

        # Step 3: 转换为数组并全局缩放
        X_train_imputed_np = X_train_std.values
        X_val_imputed_np = X_val_std.values

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train_imputed_np)
        self.X_val_scaled = self.scaler.transform(X_val_imputed_np)

        # Step 4: 异常值裁剪 + GPU优化
        self.X_train_scaled = np.clip(self.X_train_scaled, a_min=-3, a_max=3)
        self.X_val_scaled = np.clip(self.X_val_scaled, a_min=-3, a_max=3)

        if self.use_gpu:
            self.X_train_scaled = self.X_train_scaled.astype(np.float32)
            self.X_val_scaled = self.X_val_scaled.astype(np.float32)
            logger.info(f"Data converted to float32 for GPU acceleration")

        logger.info(f"Preprocessed shapes - Train: {self.X_train_scaled.shape}, Val: {self.X_val_scaled.shape}")
        logger.info(
            f"Preprocessing completed (Imputation: {self.imputation_method} | Group STD: {self.group_std_method})")
        return self

    def _resample_train_data(self, X_train, y_train):
        """仅对训练集进行SMOTE过采样，解决类别不平衡"""
        logger.info(f"重采样前训练集分布: early={np.sum(y_train==0)}, late={np.sum(y_train==1)}")

        smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=2)  # k_neighbors适配小样本
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"重采样后训练集分布: early={np.sum(y_train_resampled==0)}, late={np.sum(y_train_resampled==1)}")
        return X_train_resampled, y_train_resampled

    def filter_lipid_features(self):
        """Lipid + bile acid feature filtering (balanced selection)"""
        logger.info("\nPre-filtering lipid + bile acid features (balanced selection)")
        if self.X_train_reduced is None:
            self.X_train_reduced = self.X_train_scaled.copy()
        if self.X_val_reduced is None:
            self.X_val_reduced = self.X_val_scaled.copy()

        if self.args.is_filter:
            self.reduced_feature_names = self.feature_columns

            # 1. Split lipid/bile acid/clinical features
            lipid_cols = [f for f in self.lipid_features if f in self.reduced_feature_names]
            bile_acid_cols = [f for f in self.bile_acid_features if f in self.reduced_feature_names]
            clinic_cols = [f for f in self.clinical_features if f in self.reduced_feature_names]

            # 2. Lipid feature filtering
            valid_lipid_cols = [f for f in lipid_cols if f in self.reduced_feature_names]
            lipid_indices = [self.reduced_feature_names.index(f) for f in valid_lipid_cols]
            X_train_lipid = self.X_train_reduced[:, lipid_indices]
            lipid_names = valid_lipid_cols
            n_samples, n_lipid_features = X_train_lipid.shape
            logger.info(f"Initial lipid features: {n_lipid_features}, training samples: {n_samples}")

            lipid_count_tracker = {
                'Initial': len(lipid_names),
                'Variance Filter': 0,
                'ANOVA Filter': 0,
                'Correlation Reduction': 0,
                'SHAP Filter': 0
            }

            assert len(np.unique(self.y_train)) == self.n_classes, \
                f"Invalid training label count: {len(np.unique(self.y_train))} (expected {self.n_classes})"

            # Step 1: Variance filter (top 30%)
            lipid_var = np.var(X_train_lipid, axis=0)
            var_threshold = np.percentile(lipid_var, 70)
            high_var_idx = np.where(lipid_var >= var_threshold)[0]
            X_train_lipid = X_train_lipid[:, high_var_idx]
            lipid_names = [lipid_names[i] for i in high_var_idx]
            lipid_count_tracker['Variance Filter'] = len(lipid_names)
            logger.info(f"Variance filtered lipid features: {len(lipid_names)}")

            # 3. Bile acid feature filtering
            valid_bile_cols = [f for f in bile_acid_cols if f in self.reduced_feature_names]
            bile_indices = [self.reduced_feature_names.index(f) for f in valid_bile_cols]
            X_train_bile = self.X_train_reduced[:, bile_indices]

            bile_count_tracker = {
                'Initial': len(valid_bile_cols),
                'Variance Filter': 0,
                'ANOVA Filter': 0,
                'Correlation Reduction': 0,
                'Top 20 Selection': 0
            }

            # Variance filter (top 50%)
            bile_var = np.var(X_train_bile, axis=0)
            var_threshold = np.percentile(bile_var, 50)
            high_var_idx = np.where(bile_var >= var_threshold)[0]
            X_train_bile = X_train_bile[:, high_var_idx]
            bile_names = [valid_bile_cols[i] for i in high_var_idx]
            bile_count_tracker['Variance Filter'] = len(bile_names)
            logger.info(f"Variance filtered bile acid features: {len(bile_names)}")

            # 4. Combine features
            final_features = clinic_cols + lipid_names + bile_names
            self.reduced_feature_names = final_features
            final_indices = [self.feature_columns.index(f) for f in final_features]
            self.X_train_reduced = self.X_train_scaled[:, final_indices]
            self.X_val_reduced = self.X_val_scaled[:, final_indices]

            # 保存过滤后的数据
            if self.save_filtered_data:
                os.makedirs(os.path.join(self.save_root, self.filter_dir), exist_ok=True)

                try:
                    train_path = os.path.join(self.save_root, self.filter_dir, f"X_train_reduced.{self.save_format}")
                    val_path = os.path.join(self.save_root, self.filter_dir, f"X_val_reduced.{self.save_format}")
                    feat_name_path = os.path.join(self.save_root, self.filter_dir, "reduced_feature_names.txt")

                    with open(feat_name_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(self.reduced_feature_names))

                    if self.save_format == "npy":
                        np.save(train_path, self.X_train_reduced)
                        np.save(val_path, self.X_val_reduced)
                    elif self.save_format == "csv":
                        pd.DataFrame(self.X_train_reduced, columns=self.reduced_feature_names).to_csv(train_path,
                                                                                                      index=False)
                        pd.DataFrame(self.X_val_reduced, columns=self.reduced_feature_names).to_csv(val_path,
                                                                                                    index=False)
                    else:
                        raise ValueError(f"Unsupported save format: {self.save_format} (only npy/csv)")

                    logger.info(f"Filtered data saved successfully:")
                    logger.info(f"  - Training data: {train_path}")
                    logger.info(f"  - Validation data: {val_path}")
                    logger.info(f"  - Feature names: {feat_name_path}")
                except Exception as e:
                    logger.error(f"Failed to save filtered data: {str(e)}", exc_info=True)

            logger.info(
                f"Filtered features: lipid {len(lipid_names)} + bile acid {len(bile_names)} + clinical {len(clinic_cols)}")
        return self

    def _plot_feature_count_change(self, lipid_counts, bile_counts):
        """Plot line chart of feature count changes during lipid/bile acid filtering"""
        logger.info("Generating feature count change line chart...")

        steps = list(lipid_counts.keys())
        lipid_vals = [lipid_counts[step] for step in steps]
        bile_steps_mapping = {
            'Initial': 'Initial',
            'Variance Filter': 'Variance Filter',
            'ANOVA Filter': 'ANOVA Filter',
            'Correlation Reduction': 'Correlation Reduction',
            'SHAP Filter': 'Top 20 Selection'
        }
        bile_vals = [bile_counts[bile_steps_mapping[step]] for step in steps]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, lipid_vals, marker='o', linewidth=2, markersize=8,
                color=COLOR_PALETTE['lipid'], label='Lipid Features', alpha=0.8)
        ax.plot(steps, bile_vals, marker='s', linewidth=2, markersize=8,
                color=COLOR_PALETTE['bile_acid'], label='Bile Acid Features', alpha=0.8)

        ax.set_xlabel('Filtering Step', fontsize=9)
        ax.set_ylabel('Number of Features', fontsize=9)
        ax.set_title('Feature Count Changes During Lipid/Bile Acid Filtering', fontsize=10, pad=15)
        ax.legend(fontsize=8)
        ax.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        for i, (l_val, b_val) in enumerate(zip(lipid_vals, bile_vals)):
            ax.text(i, l_val + 3, str(l_val), ha='center', va='bottom', fontsize=7, color=COLOR_PALETTE['lipid'])
            ax.text(i, b_val - 3, str(b_val), ha='center', va='top', fontsize=7, color=COLOR_PALETTE['bile_acid'])

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir, f'feature_count_change_{self.feature_type_filter}.png'),
            dpi=600, bbox_inches='tight'
            )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir, f'feature_count_change_{self.feature_type_filter}.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()
        logger.info("Feature count change line chart saved")

    def _plot_top30_lipid_shap(self, shap_series):
        """Plot SHAP feature importance ranking for top 30 lipids"""
        logger.info("Generating SHAP feature importance plot for top 30 lipids...")

        top30_shap = shap_series.head(30).sort_values(ascending=True)
        fig_height = max(8, 0.3 * len(top30_shap))

        fig, ax = plt.subplots(figsize=(6, fig_height))
        bars = ax.barh(range(len(top30_shap)), top30_shap.values,
                       color=COLOR_PALETTE['shap'], alpha=0.8, edgecolor='black', linewidth=0.6)

        ax.set_yticks(range(len(top30_shap)))
        ax.set_yticklabels(top30_shap.index, fontsize=7)
        ax.set_xlabel('SHAP Importance (Average Absolute Value)', fontsize=9)
        ax.set_title('SHAP Feature Importance Ranking (Top 30 Lipids)', fontsize=10, pad=15)
        ax.grid(axis='x', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, max(top30_shap.values) * 1.1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root, self.figure_dir,
                                 f'ftop30_lipid_shap_importance_{self.feature_type_filter}.png'),
                    dpi=600, bbox_inches='tight'
                    )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'ftop30_lipid_shap_importance_{self.feature_type_filter}.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()
        logger.info("Top 30 lipid SHAP importance plot saved")

    def _plot_core_markers_collinearity_heatmap(self, core_features):
        """Plot collinearity heatmap for core markers (highlight r<0.8)"""
        logger.info("Generating core markers collinearity heatmap...")

        core_indices = [self.feature_columns.index(f) for f in core_features]
        X_core = self.X_train_scaled[:, core_indices]
        corr_matrix = pd.DataFrame(X_core, columns=core_features).corr()

        mask = np.abs(corr_matrix) >= 0.8
        mask = mask | np.triu(np.ones_like(corr_matrix, dtype=bool))

        corr_masked = np.ma.masked_where(mask, corr_matrix)

        n_feat = len(core_features)
        fig_size = (min(12, n_feat * 0.5), min(10, n_feat * 0.4))
        fig, ax = plt.subplots(figsize=fig_size)

        im = ax.pcolormesh(corr_masked, cmap='YlGnBu', vmin=-1, vmax=1)

        ax.set_xticks(np.arange(n_feat) + 0.5)
        ax.set_yticks(np.arange(n_feat) + 0.5)
        ax.set_xticklabels(core_features, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(core_features, fontsize=7)

        for i in range(n_feat):
            for j in range(n_feat):
                if not mask.iloc[i, j]:
                    val = corr_matrix.iloc[i, j]
                    text_color = 'white' if abs(val) > 0.6 else 'black'
                    ax.text(j + 0.5, i + 0.5, f'{val:.2f}',
                            ha='center', va='center', color=text_color,
                            fontsize=5, fontweight='bold')

        ax.set_title('Collinearity Heatmap of Core Markers (r < 0.8)',
                     fontsize=10, pad=15)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Pearson r', fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        # out_path = os.path.join(self.save_root, self.figure_dir,
        #                         f'core_markers_collinearity_heatmap_{self.feature_type_filter}.png')
        plt.savefig(os.path.join(self.save_root, self.figure_dir,
                                f'core_markers_collinearity_heatmap_{self.feature_type_filter}.png'), dpi=600, bbox_inches='tight')
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'core_markers_collinearity_heatmap_{self.feature_type_filter}.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()
        logger.info("Core markers collinearity heatmap saved")

    def reduce_dimension(self):
        logger.info(f"\nStep 3/7: Reducing dimension with {self.dimension_reduction_method.upper()}...")

        if self.dimension_reduction_method == 'pca':
            self.dimension_reducer = PCA(n_components=self.pca_variance_ratio, random_state=42)
            self.X_train_reduced = self.dimension_reducer.fit_transform(self.X_train_scaled)
            self.X_val_reduced = self.dimension_reducer.transform(self.X_val_scaled)

            n_components = self.X_train_reduced.shape[1]
            self.reduced_feature_names = [f'PC{i + 1}' for i in range(n_components)]

            pca_results = pd.DataFrame({
                'Component': self.reduced_feature_names,
                'Explained_Variance': self.dimension_reducer.explained_variance_ratio_,
                'Cumulative_Variance': np.cumsum(self.dimension_reducer.explained_variance_ratio_)
            })
            pca_results.to_csv(os.path.join(self.save_root, self.results_dir, f'pca_dimension_reduction_results.csv'),
                               index=False)

            logger.info(
                f"PCA completed: {n_components} components (cumulative variance: {np.sum(self.dimension_reducer.explained_variance_ratio_):.2%})")

        elif self.dimension_reduction_method == 'selectkbest':
            self.dimension_reducer = SelectKBest(score_func=f_classif, k=self.selectkbest_k)
            self.X_train_reduced = self.dimension_reducer.fit_transform(self.X_train_scaled, self.y_train)
            self.X_val_reduced = self.dimension_reducer.transform(self.X_val_scaled)

            selected_idx = self.dimension_reducer.get_support(indices=True)
            self.reduced_feature_names = [self.feature_columns[i] for i in selected_idx]

            selectkbest_scores = pd.DataFrame({
                'Feature': self.feature_columns,
                'F_Score': self.dimension_reducer.scores_,
                'P_Value': self.dimension_reducer.pvalues_,
                'Selected': self.dimension_reducer.get_support()
            }).sort_values('F_Score', ascending=False)
            selectkbest_scores.to_csv(
                os.path.join(self.save_root, self.results_dir, f'selectkbest_dimension_reduction_results.csv'),
                index=False)

            logger.info(f"SelectKBest completed: Top {self.selectkbest_k} features selected")

        elif self.dimension_reduction_method == 'none':
            self.dimension_reducer = None

        if self.use_gpu:
            self.X_train_reduced = self.X_train_reduced.astype(np.float32)
            self.X_val_reduced = self.X_val_reduced.astype(np.float32)

        logger.info(f"Reduced data shapes - Train: {self.X_train_reduced.shape}, Val: {self.X_val_reduced.shape}")
        logger.info(f"Reduced features count: {len(self.reduced_feature_names)}")
        return self

    def select_top_features(self):
        """Feature selection (supports sis/brute_force/shap/lasso/specified)"""
        logger.info(
            f"\nStep 4/7: Selecting top {self.n_selected} features with {self.feature_selection_method.upper()}...")

        if self.feature_selection_method == 'specified':
            missing_features = [feat for feat in self.specified_features if feat not in self.feature_columns]
            if missing_features:
                raise ValueError(f"Specified features not found: {', '.join(missing_features)}")
            self.top_features = self.specified_features.copy()
            logger.info(f"Selected specified features (first 5): {self.top_features[:5]}")

        elif self.feature_selection_method == 'shap':
            self._select_features_with_shap()

        elif self.feature_selection_method == 'lasso':
            self._select_features_with_lasso()

        elif self.feature_selection_method == 'brute_force':
            self._select_features_with_brute_force()

        elif self.feature_selection_method == 'sis':
            self._select_features_with_sis()

        logger.info(f"Top {self.n_selected} features selected (first 5: {self.top_features[:5]})")
        return self

    def _select_features_with_shap(self):
        """SHAP feature selection (二分类适配)"""
        n_samples, n_reduced_features = self.X_train_reduced.shape
        base_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        base_model.fit(self.X_train_reduced, self.y_train)
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(self.X_train_reduced, check_additivity=False)

        # 二分类适配：处理SHAP值格式（1D/2D）
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 二分类取正类（late）的SHAP值
        elif shap_values.ndim == 2:
            pass
        else:
            raise ValueError(f"Unsupported SHAP shape for binary classification: {shap_values.shape}")

        # 计算SHAP重要性
        shap_importance = np.mean(np.abs(shap_values), axis=0)
        shap_series = pd.Series(shap_importance, index=self.reduced_feature_names)
        self.top_features = shap_series.sort_values(ascending=False).head(self.n_selected).index.tolist()

    def _select_features_with_lasso(self):
        """Lasso feature selection (GPU-optimized, 二分类适配)"""
        n_reduced_features = len(self.reduced_feature_names)
        lasso_importance = np.zeros(n_reduced_features)

        logger.info(f"Running LassoCV (Binary) with GPU precision fix...")
        # 二分类：直接拟合，无需OvR
        y_binary = self.y_train
        X_train_float64 = self.X_train_reduced.astype(np.float64)
        lasso = LassoCV(
            cv=5, random_state=42, max_iter=10000,
            n_jobs=1, precompute=False, tol=1e-4
        )
        lasso.fit(X_train_float64, y_binary)
        lasso_importance = np.abs(lasso.coef_)
        logger.info(
            f"Binary Lasso: Best alpha = {lasso.alpha_:.6f}, non-zero features = {np.sum(lasso_importance > 0)}")

        lasso_series = pd.Series(lasso_importance, index=self.reduced_feature_names)

        if np.sum(lasso_series > 0) < self.n_selected:
            logger.warning(
                f"Lasso non-zero features ({np.sum(lasso_series > 0)}) < {self.n_selected}, supplementing high variance features")
            non_zero_features = lasso_series[lasso_series > 0].sort_values(ascending=False).index.tolist()
            remaining_features = [f for f in self.reduced_feature_names if f not in non_zero_features]
            remaining_indices = [self.reduced_feature_names.index(f) for f in remaining_features]
            feature_variances = np.var(self.X_train_reduced[:, remaining_indices].astype(np.float64), axis=0)
            variance_series = pd.Series(feature_variances, index=remaining_features).sort_values(ascending=False)
            supplement_features = variance_series.head(self.n_selected - len(non_zero_features)).index.tolist()
            self.top_features = non_zero_features + supplement_features
        else:
            self.top_features = lasso_series.sort_values(ascending=False).head(self.n_selected).index.tolist()

        lasso_results = pd.DataFrame({
            'Feature': self.reduced_feature_names,
            'Lasso_Importance': lasso_importance,
            'Selected': [f in self.top_features for f in self.reduced_feature_names]
        }).sort_values('Lasso_Importance', ascending=False)
        lasso_results.to_csv(os.path.join(self.save_root, self.results_dir,
                                          f'lasso_feature_importance_{self.dimension_reduction_method}.csv'),
                             index=False)

    def _select_features_with_brute_force(self):
        """Brute force feature selection (batch processing for memory efficiency, 二分类适配)"""
        import gc
        n_reduced_features = len(self.reduced_feature_names)
        n_samples_train = self.X_train_reduced.shape[0]
        n_samples_val = self.X_val_reduced.shape[0]

        if n_reduced_features < self.n_selected:
            raise ValueError(f"Reduced features ({n_reduced_features}) < selected count ({self.n_selected})")

        def comb(n, k):
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            k = min(k, n - k)
            result = 1
            for i in range(1, k + 1):
                result = result * (n - k + i) // i
            return result

        total_combinations = comb(n_reduced_features, self.n_selected)
        logger.info(f"Total combinations: {total_combinations:,}")

        from itertools import combinations
        comb_generator = combinations(range(n_reduced_features), self.n_selected)
        BATCH_SIZE = 100
        batch_idx = 0
        current_batch = []

        brute_force_results = []
        best_auc = 0.0
        best_combination = None

        if self.use_gpu and isinstance(self.brute_force_base_model, XGBClassifier):
            self.brute_force_base_model.set_params(
                tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor', n_jobs=1
            )
            logger.info("Brute force base model optimized for GPU")

        logger.info(f"Brute force screening (batch processing, Binary AUC)...")
        for feat_indices in tqdm(comb_generator, total=min(total_combinations, self.brute_force_max_combinations),
                                 desc="Brute force screening"):
            current_batch.append(feat_indices)

            if len(current_batch) >= BATCH_SIZE or len(brute_force_results) + len(
                    current_batch) >= self.brute_force_max_combinations:
                for idx_in_batch, indices in enumerate(current_batch):
                    global_idx = batch_idx * BATCH_SIZE + idx_in_batch
                    if global_idx >= self.brute_force_max_combinations:
                        break

                    X_train_comb = self.X_train_reduced[:, indices]
                    X_val_comb = self.X_val_reduced[:, indices]

                    self.brute_force_base_model.fit(X_train_comb, self.y_train)
                    y_val_proba = self.brute_force_base_model.predict_proba(X_val_comb)[:, 1]  # 二分类取正类概率
                    val_auc = roc_auc_score(self.y_val, y_val_proba)  # 二分类AUC，移除multi_class

                    feat_names = [self.reduced_feature_names[i] for i in indices]
                    brute_force_results.append({
                        'combination_idx': global_idx,
                        'feature_indices': indices,
                        'feature_names': ', '.join(feat_names),
                        'val_auc': val_auc,
                        'train_samples': n_samples_train,
                        'val_samples': n_samples_val
                    })

                    if val_auc > best_auc:
                        best_auc = val_auc
                        best_combination = feat_names
                        logger.debug(f"New best (batch {batch_idx}): AUC={val_auc:.4f}, features={feat_names[:3]}...")

                current_batch = []
                batch_idx += 1
                gc.collect()
                logger.info(f"Batch {batch_idx} processed, total combinations: {len(brute_force_results)}")

                if len(brute_force_results) >= self.brute_force_max_combinations:
                    break

        if current_batch and len(brute_force_results) < self.brute_force_max_combinations:
            for idx_in_batch, indices in enumerate(current_batch):
                global_idx = batch_idx * BATCH_SIZE + idx_in_batch
                if global_idx >= self.brute_force_max_combinations:
                    break

                X_train_comb = self.X_train_reduced[:, indices]
                X_val_comb = self.X_val_reduced[:, indices]

                self.brute_force_base_model.fit(X_train_comb, self.y_train)
                y_val_proba = self.brute_force_base_model.predict_proba(X_val_comb)[:, 1]
                val_auc = roc_auc_score(self.y_val, y_val_proba)

                feat_names = [self.reduced_feature_names[i] for i in indices]
                brute_force_results.append({
                    'combination_idx': global_idx,
                    'feature_indices': indices,
                    'feature_names': ', '.join(feat_names),
                    'val_auc': val_auc,
                    'train_samples': n_samples_train,
                    'val_samples': n_samples_val
                })

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_combination = feat_names

            gc.collect()
            logger.info(f"Remaining combinations processed, total: {len(brute_force_results)}")

        self.brute_force_results = pd.DataFrame(brute_force_results)
        self.brute_force_results = self.brute_force_results.sort_values('val_auc', ascending=False).reset_index(
            drop=True)
        self.brute_force_results.to_csv(os.path.join(self.save_root, self.results_dir,
                                                     f'brute_force_results_{self.dimension_reduction_method}_top{self.n_selected}_binary.csv'),
                                        index=False, encoding='utf-8-sig'
                                        )

        top100_combinations = self.brute_force_results.head(100)
        feature_freq = {}
        for _, row in top100_combinations.iterrows():
            features = row['feature_names'].split(', ')
            for feat in features:
                feature_freq[feat] = feature_freq.get(feat, 0) + 1
        freq_df = pd.DataFrame({'Feature': feature_freq.keys(), 'Frequency': feature_freq.values()}).sort_values(
            'Frequency', ascending=False)
        freq_df.to_csv(os.path.join(self.save_root, self.results_dir,
                                    f'brute_force_feature_frequency_{self.dimension_reduction_method}_top{self.n_selected}_binary.csv'),
                       index=False)

        self.top_features = best_combination
        logger.info(f"Best combination (AUC: {best_auc:.4f}): {self.top_features}")
        logger.info(f"Top 5 combinations by AUC:")
        for i in range(min(5, len(self.brute_force_results))):
            row = self.brute_force_results.iloc[i]
            logger.info(f"  Rank {i+1}: AUC={row['val_auc']:.4f}, features={row['feature_names'][:50]}...")

    def _select_features_with_sis(self):
        """SIS feature selection (weighted score for binary classification)"""
        n_samples, n_reduced_features = self.X_train_reduced.shape
        logger.info(f"SIS feature selection: {n_reduced_features} reduced features (Binary)")
        logger.info(f"SIS config: score func={self.sis_score_func.__name__}, k method={self.sis_k_method}")

        # 二分类：直接计算SIS分数，无需多分类加权
        raw_scores, p_values = self.sis_score_func(self.X_train_reduced, self.y_train)
        weighted_scores = raw_scores  # 二分类无需类别加权

        self.sis_scores = pd.DataFrame({
            'Feature': self.reduced_feature_names,
            'Raw_SIS_Score': raw_scores,
            'Weighted_SIS_Score': weighted_scores,
            'P_Value': p_values
        }).sort_values('Weighted_SIS_Score', ascending=False)

        n_train_samples = len(self.y_train)
        if self.sis_k_method == 'n_log':
            sis_d = int(np.ceil(n_train_samples / np.log(n_train_samples)))
        elif self.sis_k_method == 'sqrt':
            sis_d = int(np.ceil(np.sqrt(n_train_samples)))
        elif self.sis_k_method == 'fixed':
            sis_d = self.sis_fixed_k
        else:
            raise ValueError(f"Unsupported SIS k method: {self.sis_k_method} (n_log/sqrt/fixed)")

        sis_d = min(sis_d, n_reduced_features)
        sis_d = max(sis_d, self.n_selected)
        logger.info(f"SIS screening count (d): {sis_d} (train samples: {n_train_samples})")

        sis_top_d_features = self.sis_scores.head(sis_d)['Feature'].tolist()
        logger.info(f"SIS step 1: top {sis_d} features (weighted score)")

        sis_top_d_scores = self.sis_scores[self.sis_scores['Feature'].isin(sis_top_d_features)]
        self.top_features = sis_top_d_scores.sort_values('Weighted_SIS_Score', ascending=False).head(self.n_selected)[
            'Feature'].tolist()

        self.sis_scores['SIS_Top_d_Selected'] = self.sis_scores['Feature'].isin(sis_top_d_features)
        self.sis_scores['Final_Selected'] = self.sis_scores['Feature'].isin(self.top_features)
        self.sis_scores.to_csv(os.path.join(self.save_root, self.results_dir,
                                            f'sis_feature_selection_results_{self.dimension_reduction_method}_top{self.n_selected}_binary.csv'),
                               index=False, encoding='utf-8-sig'
                               )
        logger.info(f"SIS results saved to results_3/sis_feature_selection_results_*.csv (Binary)")

    def _perform_kruskal_wallis_test(self, feature_data, class_labels, class_names):
        """Kruskal-Wallis H test (non-parametric, 二分类适配)"""
        unique_vals = np.unique(feature_data)
        if len(unique_vals) <= 1:
            logger.warning(f"Feature has identical values (unique count: {len(unique_vals)}), skip Kruskal-Wallis test")
            return {
                'H_statistic': np.nan,
                'p_value': np.nan,
                'significance': 'N/A (all values identical)'
            }

        class_groups = []
        valid_class_names = []
        for cls_idx, cls_name in enumerate(class_names):
            group_data = feature_data[class_labels == cls_idx]
            if len(group_data) > 0 and len(np.unique(group_data)) > 0:
                class_groups.append(group_data)
                valid_class_names.append(cls_name)

        if len(class_groups) < 2:
            logger.warning(f"Only {len(class_groups)} valid groups, skip Kruskal-Wallis test")
            return {
                'H_statistic': np.nan,
                'p_value': np.nan,
                'significance': 'N/A (insufficient valid groups)'
            }

        try:
            h_stat, p_val = kruskal(*class_groups)
            if np.isnan(p_val):
                significance = 'N/A'
            elif p_val < 0.001:
                significance = '*** (p<0.001)'
            elif p_val < 0.01:
                significance = '** (p<0.01)'
            elif p_val < 0.05:
                significance = '* (p<0.05)'
            else:
                significance = 'ns (p≥0.05)'

            return {
                'H_statistic': round(h_stat, 3),
                'p_value': round(p_val, 6),
                'significance': significance
            }
        except Exception as e:
            logger.error(f"Kruskal-Wallis test failed: {str(e)}")
            return {
                'H_statistic': np.nan,
                'p_value': np.nan,
                'significance': f'Error: {str(e)[:50]}'
            }

    # def define_models(self):
    #     """Model definition (GPU optimized, 二分类适配)"""
    #     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #
    #     models_config = {
    #         'RandomForest': {
    #             'model': RandomForestClassifier(random_state=42, n_jobs=-1),
    #             'param_grid': {
    #                 'n_estimators': [81, 100, 150, 200],
    #                 'max_depth': [5, 6, 7],
    #                 'min_samples_split': [4, 5, 8],
    #                 'min_samples_leaf': [1, 2, 3],
    #                 'class_weight': ['balanced', {0: 1.0, 1: 1.0}]  # 二分类类别权重
    #             },
    #             'cv': cv
    #         },
    #         'LogisticRegression': {
    #             'model': LogisticRegression(random_state=42, max_iter=2000, n_jobs=-1),
    #             'param_grid': {
    #                 'C': [0.001, 0.01, 0.1, 1.0, 10.0, 20.0, 30.0],
    #                 'penalty': ['l2'],
    #                 'class_weight': ['balanced', None],
    #                 'l1_ratio': [None]
    #             },
    #             'cv': cv
    #         },
    #         'SupportVectorMachine': {
    #             'model': SVC(probability=True, random_state=42),
    #             'param_grid': {
    #                 'C': [0.01, 0.1, 1.0, 10.0],
    #                 'kernel': ['rbf', 'linear'],
    #                 'gamma': ['scale', 'auto', 0.001, 0.01],
    #                 'class_weight': ['balanced', None]
    #             },
    #             'cv': cv
    #         },
    #         'XGBoost': {
    #             'model': XGBClassifier(random_state=42, eval_metric='auc', scale_pos_weight=4),
    #             'param_grid': {
    #                 'n_estimators': [500, 800],
    #                 'max_depth': [5, 7],
    #                 'learning_rate': [0.01, 0.05],
    #                 'subsample': [0.8],
    #                 'colsample_bytree': [0.8],
    #                 'min_child_weight': [3, 5],
    #                 'reg_alpha': [0.1, 0.5],
    #                 'reg_lambda': [0.1, 0.5]
    #             },
    #             'cv': cv
    #         }
    #     }
    #
    #     if self.use_gpu:
    #         logger.info("Optimizing XGBoost for GPU (Binary)")
    #         models_config['XGBoost']['model'].set_params(
    #             tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor',
    #             enable_categorical=False, n_jobs=1, max_bin=256, cache_size=1024
    #         )
    #         models_config['XGBoost']['param_grid'].update({
    #             'gpu_id': [0],
    #             'tree_method': ['gpu_hist']
    #         })
    #     else:
    #         models_config['XGBoost']['model'].set_params(
    #             tree_method='hist', n_jobs=-1
    #         )
    #
    #     return models_config
    def define_models(self):
        """Model definition (GPU optimized, 二分类适配) - 优化小样本+类别不平衡参数"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 计算类别权重（适配late类别样本少的问题）
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=self.y_train)
        weight_dict = {0: class_weights[0], 1: class_weights[1]}
        logger.info(f"计算类别权重: early={class_weights[0]:.2f}, late={class_weights[1]:.2f}")

        models_config = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [50, 80],  # 降低树数量（原100+/200）
                    'max_depth': [3, 4],  # 降低树深度（原5+/6+）
                    'min_samples_split': [8, 10],  # 增加分裂所需最小样本（减少过拟合）
                    'min_samples_leaf': [3, 5],  # 增加叶节点最小样本
                    'class_weight': [weight_dict],  # 用计算出的平衡权重（原多选项）
                    'max_features': ['sqrt'],  # 限制每棵树使用的特征数
                    'bootstrap': [True],  # 开启bootstrap抽样（小样本必备）
                    'oob_score': [True]  # 启用袋外分数（评估泛化能力）
                },
                'cv': cv
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=2000, n_jobs=-1),
                'param_grid': {
                    'C': [0.1, 1.0, 5.0],  # 减少C值范围（原0.001-30），增强正则化
                    'penalty': ['l2'],
                    'class_weight': [weight_dict],  # 固定平衡权重
                    'solver': ['liblinear'],  # 小样本更稳定的求解器
                    'max_iter': [2000]
                },
                'cv': cv
            },
            'SupportVectorMachine': {
                'model': SVC(probability=True, random_state=42),
                'param_grid': {
                    'C': [0.1, 1.0],  # 减少C值（降低拟合度）
                    'kernel': ['linear'],  # 仅保留线性核（非线性核易过拟合小样本）
                    'gamma': ['scale'],  # 固定gamma
                    'class_weight': [weight_dict],
                    'tol': [1e-3],  # 放宽收敛阈值
                    'max_iter': [1000]  # 限制迭代次数
                },
                'cv': cv
            },
            'XGBoost': {
                # 核心优化：降低复杂度+适配类别权重+增强正则化
                'model': XGBClassifier(
                    random_state=42,
                    eval_metric='auc',
                    scale_pos_weight=class_weights[1] / class_weights[0],  # 适配类别比例（原固定4）
                    use_label_encoder=False
                ),
                'param_grid': {
                    'n_estimators': [100, 200],  # 大幅减少树数量（原500/800）
                    'max_depth': [2, 3],  # 降低树深度（原5/7）
                    'learning_rate': [0.1, 0.05],  # 学习率适度调整
                    'subsample': [0.8, 0.9],  # 行抽样（减少过拟合）
                    'colsample_bytree': [0.8],  # 列抽样
                    'min_child_weight': [5, 8],  # 增加最小子节点权重（减少过拟合）
                    'reg_alpha': [1.0, 2.0],  # L1正则化（原0.1/0.5，增强）
                    'reg_lambda': [1.0, 2.0],  # L2正则化（原0.1/0.5，增强）
                    'gamma': [0.1, 0.5]  # 分裂所需最小损失减少（增强正则化）
                },
                'cv': cv
            }
        }

        if self.use_gpu:
            logger.info("Optimizing XGBoost for GPU (Binary)")
            models_config['XGBoost']['model'].set_params(
                tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor',
                enable_categorical=False, n_jobs=1, max_bin=256, cache_size=1024
            )
            models_config['XGBoost']['param_grid'].update({
                'gpu_id': [0],
                'tree_method': ['gpu_hist']
            })
        else:
            models_config['XGBoost']['model'].set_params(
                tree_method='hist', n_jobs=-1
            )

        return models_config

    def grid_search_with_cv(self):
        """Grid search with cross validation (GPU optimized, 二分类适配)"""
        logger.info("\nStep 5/7: Grid search with 5-fold CV (Binary)...")
        models = self.define_models()
        top_indices = [self.feature_columns.index(f) for f in self.top_features]
        X_train_top = self.X_train_scaled[:, top_indices]
        assert X_train_top.shape[1] == self.n_selected, \
            f"Train top features shape error: {X_train_top.shape[1]} (expected {self.n_selected})"

        # 新增：训练集重采样
        X_train_top_resampled, y_train_resampled = self._resample_train_data(X_train_top, self.y_train)

        n_jobs = 1 if self.use_gpu else -1
        logger.info(f"Grid search n_jobs: {n_jobs} (GPU: {self.use_gpu})")

        for name, config in models.items():
            logger.info(f"\nTuning {name} (Binary)...")

            # 二分类适配：scoring从'roc_auc_ovr'改为'roc_auc'
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['param_grid'],
                cv=config['cv'],
                scoring='roc_auc',
                n_jobs=n_jobs,
                verbose=0,
                return_train_score=False
            )

            if self.use_gpu and name == 'XGBoost':
                grid_search.estimator.set_params(batch_size=self.gpu_batch_size)

            # grid_search.fit(X_train_top, self.y_train)
            # 新增：训练集重采样
            grid_search.fit(X_train_top_resampled, y_train_resampled)
            self.models[name] = (grid_search.best_estimator_, grid_search.best_params_)

            best_idx = grid_search.best_index_
            self.cv_results[name] = {
                'best_cv_auc': grid_search.best_score_,
                'cv_fold_auc': [grid_search.cv_results_[f'split{i}_test_score'][best_idx] for i in range(5)],
                'cv_mean': grid_search.best_score_,
                'cv_std': np.std([grid_search.cv_results_[f'split{i}_test_score'][best_idx] for i in range(5)]),
                'cv_ci_lower': grid_search.best_score_ - stats.t.ppf(0.975, 4) * (
                    np.std([grid_search.cv_results_[f'split{i}_test_score'][best_idx] for i in range(5)]) / np.sqrt(5)),
                'cv_ci_upper': grid_search.best_score_ + stats.t.ppf(0.975, 4) * (
                    np.std([grid_search.cv_results_[f'split{i}_test_score'][best_idx] for i in range(5)]) / np.sqrt(5))
            }

            logger.info(f"{name} best params: {grid_search.best_params_} (CV AUC: {grid_search.best_score_:.4f})")
            logger.info(f"{name} 5-fold CV scores: {[round(s,4) for s in self.cv_results[name]['cv_fold_auc']]}")
        return self

    # 补全中断的evaluate_models方法（完整二分类评估逻辑）
    def evaluate_models(self):
        """Evaluate models on validation set (AUC, CI, Delong test, clinical metrics, 二分类适配)"""
        logger.info("\nStep 6/7: Evaluating models (Binary)...")
        target_names = self.class_names
        top_indices = [self.feature_columns.index(f) for f in self.top_features]
        X_val_top = self.X_val_scaled[:, top_indices]
        assert X_val_top.shape[1] == self.n_selected, \
            f"Validation top features shape error: {X_val_top.shape[1]} (expected {self.n_selected})"
        X_train_top = self.X_train_scaled[:, top_indices]

        val_aucs = {}
        val_probas = {}
        train_probas = {}

        for name, (model, best_params) in self.models.items():
            logger.info(f"\nEvaluating {name}...")
            # 二分类：取正类（late，1）的概率值
            if self.use_gpu and name == 'XGBoost':
                y_val_proba = model.predict_proba(X_val_top)[:, 1]
                y_train_proba = model.predict_proba(X_train_top)[:, 1]
            else:
                y_val_proba = model.predict_proba(X_val_top)[:, 1]
                y_train_proba = model.predict_proba(X_train_top)[:, 1]

            y_val_pred = model.predict(X_val_top)
            y_train_pred = model.predict(X_train_top)

            # 二分类核心评估指标（移除多分类相关参数）
            val_auc = roc_auc_score(self.y_val, y_val_proba)
            train_auc = roc_auc_score(self.y_train, y_train_proba)

            # AUC 95%置信区间（自助法）
            val_ci_lower, val_ci_upper = calculate_auc_ci(self.y_val, y_val_proba, n_bootstrap=1000)
            train_ci_lower, train_ci_upper = calculate_auc_ci(self.y_train, y_train_proba, n_bootstrap=1000)

            # 分类性能指标（宏平均/加权平均，适配二分类）
            train_recall = recall_score(self.y_train, y_train_pred, average='weighted')
            val_recall = recall_score(self.y_val, y_val_pred, average='weighted')
            train_f1 = f1_score(self.y_train, y_train_pred, average='weighted')
            val_f1 = f1_score(self.y_val, y_val_pred, average='weighted')

            # 混淆矩阵
            train_cm = confusion_matrix(self.y_train, y_train_pred)
            val_cm = confusion_matrix(self.y_val, y_val_pred)

            # 分类报告
            train_class_report = classification_report(self.y_train, y_train_pred, target_names=target_names,
                                                       output_dict=True)
            val_class_report = classification_report(self.y_val, y_val_pred, target_names=target_names,
                                                     output_dict=True)

            # 逐类别指标（accuracy/sensitivity/specificity/AUC + 95% CI）
            per_class_metrics = {}
            for split_name, y_true_s, y_pred_s, y_proba_s in [
                ('train', self.y_train, y_train_pred, y_train_proba),
                ('val',   self.y_val,   y_val_pred,   y_val_proba)
            ]:
                per_class_metrics[split_name] = {}
                n_total = len(y_true_s)
                for cls in [0, 1]:
                    cls_name = self.class_names[cls]
                    true_bin = (y_true_s == cls).astype(int)
                    pred_bin = (y_pred_s == cls).astype(int)
                    cm_cls = confusion_matrix(true_bin, pred_bin, labels=[0, 1])
                    tn_c, fp_c, fn_c, tp_c = cm_cls.ravel()

                    # Accuracy (binary OvR)
                    acc_c = (tp_c + tn_c) / n_total
                    acc_ci = proportion_confint(tp_c + tn_c, n_total, alpha=0.05, method='wilson')
                    acc_ci = (max(0.0, min(acc_ci[0], acc_c)), min(1.0, max(acc_ci[1], acc_c)))

                    # Sensitivity = TP / (TP + FN)
                    sen_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
                    sen_ci = proportion_confint(tp_c, tp_c + fn_c, alpha=0.05, method='wilson') if (tp_c + fn_c) > 0 else (0.0, 0.0)
                    sen_ci = (max(0.0, min(sen_ci[0], sen_c)), min(1.0, max(sen_ci[1], sen_c)))

                    # Specificity = TN / (TN + FP)
                    spe_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0
                    spe_ci = proportion_confint(tn_c, tn_c + fp_c, alpha=0.05, method='wilson') if (tn_c + fp_c) > 0 else (0.0, 0.0)
                    spe_ci = (max(0.0, min(spe_ci[0], spe_c)), min(1.0, max(spe_ci[1], spe_c)))

                    # AUC (binary OvR: use class probability)
                    try:
                        if len(np.unique(true_bin)) < 2:
                            auc_c = float('nan')
                            auc_ci_lo, auc_ci_hi = float('nan'), float('nan')
                        else:
                            auc_c = roc_auc_score(true_bin, y_proba_s if cls == 1 else 1 - y_proba_s)
                            auc_ci_lo, auc_ci_hi = calculate_auc_ci(
                                true_bin, y_proba_s if cls == 1 else 1 - y_proba_s, n_bootstrap=1000)
                    except Exception:
                        auc_c = float('nan')
                        auc_ci_lo, auc_ci_hi = float('nan'), float('nan')

                    per_class_metrics[split_name][cls_name] = {
                        'Accuracy':        round(acc_c, 4),
                        'Accuracy_CI_L':   round(acc_ci[0], 4),
                        'Accuracy_CI_U':   round(acc_ci[1], 4),
                        'Sensitivity':     round(sen_c, 4),
                        'Sensitivity_CI_L': round(sen_ci[0], 4),
                        'Sensitivity_CI_U': round(sen_ci[1], 4),
                        'Specificity':     round(spe_c, 4),
                        'Specificity_CI_L': round(spe_ci[0], 4),
                        'Specificity_CI_U': round(spe_ci[1], 4),
                        'AUC':             round(auc_c, 4) if not np.isnan(auc_c) else 'NA',
                        'AUC_CI_L':        round(auc_ci_lo, 4) if not np.isnan(auc_ci_lo) else 'NA',
                        'AUC_CI_U':        round(auc_ci_hi, 4) if not np.isnan(auc_ci_hi) else 'NA',
                    }
                    logger.info(
                        f"{name} [{split_name}] {cls_name}: "
                        f"Acc={acc_c:.4f}({acc_ci[0]:.4f}-{acc_ci[1]:.4f}), "
                        f"Sen={sen_c:.4f}({sen_ci[0]:.4f}-{sen_ci[1]:.4f}), "
                        f"Spe={spe_c:.4f}({spe_ci[0]:.4f}-{spe_ci[1]:.4f}), "
                        f"AUC={auc_c:.4f}({auc_ci_lo:.4f}-{auc_ci_hi:.4f})"
                    )

            # 保存完整结果
            self.model_results[name] = {
                # AUC及置信区间
                'train_auc': train_auc, 'val_auc': val_auc,
                'train_ci_lower': train_ci_lower, 'train_ci_upper': train_ci_upper,
                'val_ci_lower': val_ci_lower, 'val_ci_upper': val_ci_upper,
                # 分类指标
                'train_recall': train_recall, 'val_recall': val_recall,
                'train_f1': train_f1, 'val_f1': val_f1,
                # 混淆矩阵
                'train_confusion_matrix': train_cm.tolist(),
                'val_confusion_matrix': val_cm.tolist(),
                # 分类报告
                'train_classification_report': train_class_report,
                'val_classification_report': val_class_report,
                # 最优参数
                'best_params': best_params,
                # 概率值（用于后续Delong检验/DCA分析）
                'train_proba': y_train_proba.tolist(),
                'val_proba': y_val_proba.tolist(),
                # ========== 新增：特征重要性数据 ==========
                'feature_importance': self._extract_feature_importance(model, name, X_train_top).tolist(),
                'feature_names': self.top_features.copy(),
                # ========== 新增：逐类别指标 ==========
                'per_class_metrics': per_class_metrics
            }

            # 缓存AUC和概率值
            val_aucs[name] = val_auc
            val_probas[name] = y_val_proba
            train_probas[name] = y_train_proba

            # 打印关键结果
            logger.info(f"{name} - Train AUC: {train_auc:.4f} (95% CI: {train_ci_lower:.4f}-{train_ci_upper:.4f})")
            logger.info(f"{name} - Val AUC: {val_auc:.4f} (95% CI: {val_ci_lower:.4f}-{val_ci_upper:.4f})")
            logger.info(f"{name} - Train F1 (weighted): {train_f1:.4f}, Val F1 (weighted): {val_f1:.4f}")

        # 步骤1：筛选最优模型（基于验证集AUC）
        self.best_model_name = max(val_aucs.keys(), key=lambda k: val_aucs[k])
        self.best_model = self.models[self.best_model_name][0]
        self.best_model_params = self.models[self.best_model_name][1]
        self.best_model_metrics = self.model_results[self.best_model_name]

        logger.info(f"\n=== Best Model Selected ===")
        logger.info(f"Model Name: {self.best_model_name}")
        logger.info(f"Best Params: {self.best_model_params}")
        logger.info(
            f"Val AUC: {self.best_model_metrics['val_auc']:.4f} (95% CI: {self.best_model_metrics['val_ci_lower']:.4f}-{self.best_model_metrics['val_ci_upper']:.4f})")

        # 步骤2：Delong检验（最优模型 vs 其他所有模型，二分类AUC显著性对比）
        logger.info(f"\n=== Delong Test (Best Model vs Others) ===")
        best_val_proba = val_probas[self.best_model_name]
        for name, proba in val_probas.items():
            if name == self.best_model_name:
                continue
            p_value, auc_best, auc_other = self.delong_test.compare(
                y_true=self.y_val,
                y_score1=best_val_proba,
                y_score2=proba
            )
            self.delong_results[name] = {
                'p_value': p_value,
                'auc_best': auc_best,
                'auc_other': auc_other,
                'significant': p_value < 0.05
            }
            logger.info(
                f"{self.best_model_name} vs {name} - P-Value: {p_value:.6f} (Significant: {self.delong_results[name]['significant']})")

        # 步骤3：保存所有评估结果到CSV/JSON
        self._save_evaluation_results()

        # 步骤4：绘制核心评估可视化图（ROC、DCA、混淆矩阵）
        self._plot_evaluation_figures(val_probas)

        # 步骤5：LIME模型解释（二分类适配）
        self._lime_model_explanation(X_train_top, X_val_top)

        return self

    # 新增：保存评估结果（结构化文件）
    def _save_evaluation_results(self):
        """Save all model evaluation results to CSV/JSON files"""
        logger.info(f"\nSaving evaluation results to {os.path.join(self.save_root, self.results_dir)}...")

        # 1. 汇总AUC结果
        auc_summary = []
        for name, metrics in self.model_results.items():
            auc_summary.append({
                'Model_Name': name,
                'Train_AUC': metrics['train_auc'],
                'Train_CI_Lower': metrics['train_ci_lower'],
                'Train_CI_Upper': metrics['train_ci_upper'],
                'Val_AUC': metrics['val_auc'],
                'Val_CI_Lower': metrics['val_ci_lower'],
                'Val_CI_Upper': metrics['val_ci_upper'],
                'Val_F1_Weighted': metrics['val_f1'],
                'cv_mean_auc': self.cv_results[name]['cv_mean'],
                'cv_std_auc': self.cv_results[name]['cv_std'],
                'cv_ci_lower': self.cv_results[name]['cv_ci_lower'],
                'cv_ci_upper': self.cv_results[name]['cv_ci_upper'],
                'cv_fold_auc': self.cv_results[name]['cv_fold_auc'],
                'top_features': self.top_features,
                'feature_type_filter': self.feature_type_filter,
                'Is_Best_Model': (name == self.best_model_name)
            })
        auc_summary_df = pd.DataFrame(auc_summary)
        auc_summary_df.to_csv(
            os.path.join(self.save_root, self.results_dir, f"{self.feature_selection_method}_model_auc_summary_binary.csv"),
            index=False, encoding='utf-8-sig'
        )

        # 1b. 逐类别指标汇总（每个模型 × 训练/验证 × 两类）
        per_class_rows = []
        for name, metrics in self.model_results.items():
            pcm = metrics.get('per_class_metrics', {})
            for split_name in ['train', 'val']:
                for cls_name, vals in pcm.get(split_name, {}).items():
                    row = {'Model_Name': name, 'Split': split_name, 'Class': cls_name}
                    row.update(vals)
                    per_class_rows.append(row)
        if per_class_rows:
            per_class_df = pd.DataFrame(per_class_rows)
            per_class_df.to_csv(
                os.path.join(self.save_root, self.results_dir,
                             f"{self.feature_selection_method}_per_class_metrics_binary.csv"),
                index=False, encoding='utf-8-sig'
            )
            logger.info(f"Per-class metrics saved: {self.feature_selection_method}_per_class_metrics_binary.csv")

        # 2. Delong检验结果
        delong_summary = []
        for name, results in self.delong_results.items():
            delong_summary.append({
                'Best_Model': self.best_model_name,
                'Comparison_Model': name,
                'AUC_Best': results['auc_best'],
                'AUC_Other': results['auc_other'],
                'P_Value': results['p_value'],
                'Significant (p<0.05)': results['significant']
            })
        delong_summary_df = pd.DataFrame(delong_summary)
        delong_summary_df.to_csv(
            os.path.join(self.save_root, self.results_dir, f"{self.feature_selection_method}_delong_test_results_binary.csv"),
            index=False, encoding='utf-8-sig'
        )

        # 3. 最优模型元数据（JSON格式，便于后续加载）
        best_model_metadata = {
            'model_name': self.best_model_name,
            'n_selected_features': self.n_selected,
            'selected_features': self.top_features,
            'feature_selection_method': self.feature_selection_method,
            'dimension_reduction_method': self.dimension_reduction_method,
            'imputation_method': self.imputation_method,
            'group_standardization_method': self.group_std_method,
            'training_samples': len(self.y_train),
            'validation_samples': len(self.y_val),
            'best_params': self.best_model_params,
            'key_metrics': {
                'val_auc': self.best_model_metrics['val_auc'],
                'val_auc_95ci_lower': self.best_model_metrics['val_ci_lower'],
                'val_auc_95ci_upper': self.best_model_metrics['val_ci_upper'],
                'val_f1_weighted': self.best_model_metrics['val_f1']
            },
            'training_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(
                os.path.join(self.save_root, self.model_dir, f"{self.feature_selection_method}_best_model_metadata_binary.json"),
                'w', encoding='utf-8'
        ) as f:
            json.dump(best_model_metadata, f, indent=4, ensure_ascii=False)

        # 4. 保存最优模型（joblib格式，便于后续部署）
        joblib.dump(
            self.best_model,
            os.path.join(self.save_root, self.model_dir, f"{self.feature_selection_method}_best_model_{self.best_model_name}_binary.joblib")
        )

        logger.info("All evaluation results saved successfully.")

    # 新增：绘制评估可视化图（ROC、DCA、混淆矩阵）
    # def _plot_evaluation_figures(self, val_probas):
    #     """Plot core evaluation figures (ROC curve, DCA curve, confusion matrix)"""
    #     logger.info(f"\nPlotting evaluation figures...")
    #     target_names = self.class_names
    #
    #     # 1. ROC曲线（所有模型 + 最优模型高亮）
    #     fig, ax = plt.subplots(figsize=(3, 3))
    #     for name, proba in val_probas.items():
    #         fpr, tpr, _ = roc_curve(self.y_val, proba)
    #         auc = self.model_results[name]['val_auc']
    #         if name == self.best_model_name:
    #             ax.plot(fpr, tpr, linewidth=2.5, color=COLOR_PALETTE['class1'],
    #                     label=f"{name} (AUC={auc:.4f})", alpha=0.9)
    #         else:
    #             ax.plot(fpr, tpr, linewidth=1.5, alpha=0.6,
    #                     label=f"{name} (AUC={auc:.4f})")
    #     # 对角线（随机猜测）
    #     ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color=COLOR_PALETTE['grid'], label="Random Guess")
    #     ax.set_xlabel("False Positive Rate")
    #     ax.set_ylabel("True Positive Rate")
    #     ax.set_title("ROC Curves of All Models")
    #     ax.legend(fontsize=7)
    #     ax.grid(axis='both', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3, alpha=0.7)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     plt.savefig(
    #         os.path.join(self.save_root, self.figure_dir, f"roc_curves_all_models_binary.png"),
    #         dpi=600, bbox_inches='tight'
    #     )
    #     plt.close()
    #
    #     # 2. 最优模型DCA曲线（临床决策曲线分析）
    #     best_proba = val_probas[self.best_model_name]
    #     dca_results = self.delong_test.custom_dca_analysis(self.y_val, best_proba)
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.plot(dca_results['threshold'], dca_results['model_net_benefit'],
    #             linewidth=2.5, color=COLOR_PALETTE['class1'], label=f"{self.best_model_name} (Best Model)")
    #     ax.plot(dca_results['threshold'], dca_results['treat_all_net_benefit'],
    #             linewidth=1.5, linestyle='--', color=COLOR_PALETTE['val'], label="Treat All Patients")
    #     ax.plot(dca_results['threshold'], dca_results['treat_none_net_benefit'],
    #             linewidth=1.5, linestyle=':', color=COLOR_PALETTE['grid'], label="Treat None Patients")
    #     ax.set_xlabel("Threshold Probability")
    #     ax.set_ylabel("Net Benefit (per 100 samples)")
    #     ax.set_title(f"DCA Curve of Best Model ({self.best_model_name}, Validation Set)")
    #     ax.legend(fontsize=8)
    #     ax.grid(axis='both', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3, alpha=0.7)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     plt.savefig(
    #         os.path.join(self.save_root, self.figure_dir, f"dca_curve_best_model_binary.png"),
    #         dpi=600, bbox_inches='tight'
    #     )
    #     plt.close()
    #
    #     # 3. 最优模型混淆矩阵（验证集）
    #     val_cm = np.array(self.best_model_metrics['val_confusion_matrix'])
    #     fig, ax = plt.subplots(figsize=(7, 6))
    #     sns.heatmap(val_cm, annot=True, fmt='d', cmap=COLOR_PALETTE['heatmap'],
    #                 xticklabels=target_names, yticklabels=target_names, ax=ax,
    #                 cbar_kws={'label': 'Number of Samples'})
    #     ax.set_xlabel("Predicted Label")
    #     ax.set_ylabel("True Label")
    #     ax.set_title(f"Confusion Matrix of Best Model ({self.best_model_name}, Validation Set)")
    #     plt.savefig(
    #         os.path.join(self.save_root, self.figure_dir, f"confusion_matrix_best_model_binary.png"),
    #         dpi=600, bbox_inches='tight'
    #     )
    #     plt.close()
    #
    #     logger.info("All evaluation figures saved successfully.")

    # def _plot_evaluation_figures(self, val_probas):
    #     """Plot core evaluation figures (ROC curve, DCA curve, confusion matrix + feature importance ranking)
    #     新增特征重要性排名图，符合NATURE审美，保持风格统一
    #     """
    #     logger.info(f"\nPlotting evaluation figures...")
    #     target_names = self.class_names
    #     grid_color = COLOR_PALETTE['grid']
    #     best_model_color = COLOR_PALETTE['class1']  # NATURE风格低饱和主色
    #     treat_all_color = COLOR_PALETTE['val']
    #
    #     # 1. ROC曲线（保持原有优化，字体缩小+布局紧凑）
    #     fig, ax = plt.subplots(figsize=self.figure_params.get('small', (7, 6)))
    #     for name, proba in val_probas.items():
    #         fpr, tpr, _ = roc_curve(self.y_val, proba)
    #         auc = self.model_results[name]['val_auc']
    #         if name == self.best_model_name:
    #             ax.plot(fpr, tpr, linewidth=2.5, color=best_model_color,
    #                     label=f"{name} (AUC={auc:.4f})", alpha=0.9)
    #         else:
    #             ax.plot(fpr, tpr, linewidth=1.2, alpha=0.6,
    #                     color=COLOR_PALETTE.get(f'class{min(4, len(val_probas)-list(val_probas.keys()).index(name))}',
    #                                             'gray'),
    #                     label=f"{name} (AUC={auc:.4f})")
    #
    #     ax.plot([0, 1], [0, 1], linestyle='--', linewidth=0.8, color=grid_color,
    #             label="Random Guess", alpha=0.7)
    #
    #     # 字体缩小：标题→9号，坐标轴标签→8号，刻度→7号
    #     ax.set_xlabel("False Positive Rate", fontsize=8, labelpad=12)
    #     ax.set_ylabel("True Positive Rate", fontsize=8, labelpad=12)
    #     ax.set_title("ROC Curves of All Models", fontsize=9, pad=12)
    #     ax.set_xlim([-0.01, 1.01])
    #     ax.set_ylim([-0.01, 1.01])
    #
    #     # 图例优化：字体→6号，位置不遮挡曲线
    #     ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=6)
    #
    #     ax.grid(axis='both', color=grid_color, linestyle='-', linewidth=0.3, alpha=0.7)
    #     ax.set_axisbelow(True)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.tick_params(labelsize=7)  # 刻度字体缩小
    #
    #     plt.tight_layout()
    #     plt.savefig(
    #         os.path.join(self.save_root, self.figure_dir,
    #                      f"roc_curves_all_models_{self.dimension_reduction_method}_{self.feature_selection_method}_binary.png"),
    #         dpi=600
    #     )
    #     plt.close()
    #
    #     # 2. 最优模型DCA曲线（保持原有优化，字体同步缩小）
    #     best_proba = val_probas[self.best_model_name]
    #     dca_results = self.delong_test.custom_dca_analysis(self.y_val, best_proba)
    #     fig, ax = plt.subplots(figsize=(6, 5))
    #
    #     ax.plot(dca_results['threshold'], dca_results['model_net_benefit'],
    #             linewidth=2.5, color=best_model_color,
    #             label=f"{self.best_model_name} (Best Model)", alpha=0.9)
    #     ax.plot(dca_results['threshold'], dca_results['treat_all_net_benefit'],
    #             linewidth=1.5, linestyle='--', color=treat_all_color,
    #             label="Treat All Patients", alpha=0.9)
    #     ax.plot(dca_results['threshold'], dca_results['treat_none_net_benefit'],
    #             linewidth=1.5, linestyle=':', color=grid_color,
    #             label="Treat None Patients", alpha=0.7)
    #
    #     # 字体缩小：标题→10号，坐标轴标签→9号，刻度→8号
    #     ax.set_xlabel("Threshold Probability", fontsize=9, labelpad=12)
    #     ax.set_ylabel("Net Benefit (per 100 samples)", fontsize=9, labelpad=12)
    #     ax.set_title(
    #         f"DCA Curve of Best Model ({self.best_model_name}, Validation Set)",
    #         fontsize=10, pad=15
    #     )
    #     ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=7)  # 图例→7号
    #     ax.grid(axis='both', color=grid_color, linestyle='-', linewidth=0.3, alpha=0.7)
    #     ax.set_axisbelow(True)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.tick_params(labelsize=8)  # 刻度→8号
    #
    #     plt.tight_layout()
    #     plt.savefig(
    #         os.path.join(self.save_root, self.figure_dir,
    #                      f"dca_curve_best_model_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_binary.png"),
    #         dpi=600
    #     )
    #     plt.close()
    #
    #     # 4. 最优模型混淆矩阵（保持原有优化，字体同步缩小）
    #     val_cm = np.array(self.best_model_metrics['val_confusion_matrix'])
    #     cm_normalized = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]
    #
    #     fig, ax = plt.subplots(figsize=(7, 6))
    #     im = ax.imshow(cm_normalized, interpolation='nearest', cmap='YlGnBu', vmin=0, vmax=1)
    #
    #     # 文本标注字体→8号（原9号）
    #     for i in range(val_cm.shape[0]):
    #         for j in range(val_cm.shape[1]):
    #             text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
    #             text = ax.text(j, i, f'{val_cm[i, j]}\n({cm_normalized[i, j]:.2f})',
    #                            ha="center", va="center", color=text_color,
    #                            fontsize=8, fontweight='bold')  # 字体缩小
    #
    #     # 字体缩小：标题→10号，坐标轴标签→9号，刻度→8号
    #     ax.set_xlabel("Predicted Label", fontsize=9, labelpad=12)
    #     ax.set_ylabel("True Label", fontsize=9, labelpad=12)
    #     ax.set_title(
    #         f"Confusion Matrix of Best Model ({self.best_model_name}, Validation Set)",
    #         fontsize=10, pad=15
    #     )
    #     ax.set_xticks(np.arange(len(target_names)))
    #     ax.set_yticks(np.arange(len(target_names)))
    #     ax.set_xticklabels(target_names, rotation=0, ha='center', fontsize=8)  # 刻度→8号
    #     ax.set_yticklabels(target_names, fontsize=8)  # 刻度→8号
    #
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(1.0)
    #         spine.set_color('black')
    #
    #     cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.12)
    #     cbar.set_label('Normalized Score', fontsize=8, labelpad=8)  # 颜色条标签→8号
    #     cbar.ax.tick_params(labelsize=7)  # 颜色条刻度→7号
    #
    #     plt.subplots_adjust(top=0.85, bottom=0.12, left=0.12, right=0.88)
    #     plt.savefig(
    #         os.path.join(self.save_root, self.figure_dir,
    #                      f"confusion_matrix_best_model_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_binary.png"),
    #         dpi=600
    #     )
    #     plt.close()
    #
    #     logger.info("All evaluation figures saved successfully.")

    def _plot_evaluation_figures(self, val_probas):
        """Plot core evaluation figures (ROC curve, DCA curve, confusion matrix + feature importance ranking)
        新增特征重要性排名图，符合NATURE审美，保持风格统一，添加完整容错处理
        """
        logger.info(f"\nPlotting evaluation figures...")
        target_names = self.class_names
        grid_color = COLOR_PALETTE['grid']
        best_model_color = COLOR_PALETTE['class1']  # NATURE风格低饱和主色
        treat_all_color = COLOR_PALETTE['val']

        # -------------------------- 1. ROC曲线（优化版：CI阴影 + 颜色区分 + 完整标注）--------------------------
        # 每个模型对应固定颜色（Nature期刊常用配色）
        model_colors = {
            'RandomForest':        '#E64B35',  # 红
            'LogisticRegression':  '#4DBBD5',  # 蓝
            'SupportVectorMachine':'#00A087',  # 绿
            'XGBoost':             '#3C5488',  # 深蓝
        }
        fig, ax = plt.subplots(figsize=(5, 5))

        for name, proba in val_probas.items():
            fpr, tpr, _ = roc_curve(self.y_val, proba)
            auc = self.model_results[name]['val_auc']
            ci_lo = self.model_results[name]['val_ci_lower']
            ci_hi = self.model_results[name]['val_ci_upper']
            color = model_colors.get(name, '#7F7F7F')
            # label_str = f"{name}\nAUC={auc:.3f} (95%CI: {ci_lo:.3f}–{ci_hi:.3f})"
            label_str = f"{name}\nAUC={auc:.3f}"
            if name == self.best_model_name:
                # 最优模型：粗实线 + CI阴影（bootstrap）
                ax.plot(fpr, tpr, linewidth=2.2, color=color, zorder=4, label=label_str)
                # # 生成CI阴影（bootstrap重采样100次）
                # np.random.seed(42)
                # tpr_boot_list = []
                # fpr_grid = np.linspace(0, 1, 200)
                # for _ in range(200):
                #     idx = np.random.choice(len(self.y_val), len(self.y_val), replace=True)
                #     if len(np.unique(self.y_val[idx])) < 2:
                #         continue
                #     fpr_b, tpr_b, _ = roc_curve(self.y_val[idx], np.array(proba)[idx])
                #     tpr_boot_list.append(np.interp(fpr_grid, fpr_b, tpr_b))
                # if tpr_boot_list:
                #     tpr_arr = np.array(tpr_boot_list)
                #     tpr_lo = np.percentile(tpr_arr, 2.5, axis=0)
                #     tpr_hi = np.percentile(tpr_arr, 97.5, axis=0)
                #     ax.fill_between(fpr_grid, tpr_lo, tpr_hi, alpha=0.15, color=color, zorder=2)
            else:
                # 其他模型：细虚线，半透明
                ax.plot(fpr, tpr, linewidth=1.2, color=color, linestyle='--',
                        alpha=0.75, zorder=3, label=label_str)

        # 随机猜测参考线
        ax.plot([0, 1], [0, 1], linestyle=':', linewidth=1.0, color='#AAAAAA',
                label="Random Guess", alpha=0.8, zorder=1)

        # 坐标轴与样式
        ax.set_xlabel("1 - Specificity (FPR)", fontsize=9, labelpad=8)
        ax.set_ylabel("Sensitivity (TPR)", fontsize=9, labelpad=8)
        ax.set_title("ROC Curves — Validation Set", fontsize=10, pad=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.01])
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='both', color=grid_color, linestyle='-', linewidth=0.3, alpha=0.6)
        ax.set_axisbelow(True)
        # 图例：右下角，紧凑
        ax.legend(
            loc='lower right', frameon=True, framealpha=0.92,
            fontsize=6.5, labelspacing=0.4, handlelength=1.5,
            borderpad=0.6, edgecolor='#CCCCCC'
        )
        plt.tight_layout()

        roc_save_path = os.path.join(
            self.save_root, self.figure_dir,
            f"roc_curves_all_models_{self.dimension_reduction_method}_{self.feature_selection_method}_binary.png"
        )
        plt.savefig(roc_save_path, dpi=600, facecolor='white', bbox_inches='tight')
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                f'roc_curves_all_models_{self.dimension_reduction_method}_{self.feature_selection_method}_binary.svg'),
            dpi=600, format='svg', bbox_inches='tight'
        )
        plt.close()
        logger.info(f"ROC curve saved to: {roc_save_path}")

        # -------------------------- 2. 最优模型DCA曲线（补全闭合逻辑，风格统一）--------------------------
        best_proba = val_probas[self.best_model_name]
        dca_results = self.delong_test.custom_dca_analysis(self.y_val, best_proba)
        fig, ax = plt.subplots(figsize=(6, 5))

        # 绘制DCA三条曲线
        ax.plot(dca_results['threshold'], dca_results['model_net_benefit'],
                linewidth=2.5, color=best_model_color,
                label=f"{self.best_model_name} (Best Model)", alpha=0.9)
        ax.plot(dca_results['threshold'], dca_results['treat_all_net_benefit'],
                linewidth=1.5, linestyle='--', color=treat_all_color,
                label="Treat All Patients", alpha=0.9)
        ax.plot(dca_results['threshold'], dca_results['treat_none_net_benefit'],
                linewidth=1.5, linestyle=':', color=grid_color,
                label="Treat None Patients", alpha=0.7)

        # 风格优化（Nature紧凑风）
        ax.set_xlabel("Threshold Probability", fontsize=9, labelpad=12)
        ax.set_ylabel("Net Benefit (per 100 samples)", fontsize=9, labelpad=12)
        ax.set_title(
            f"DCA Curve of Best Model ({self.best_model_name}, Validation Set)",
            fontsize=10, pad=15
        )
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=7)
        ax.grid(axis='both', color=grid_color, linestyle='-', linewidth=0.3, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)

        # 保存图表
        plt.tight_layout()
        dca_save_path = os.path.join(
            self.save_root, self.figure_dir,
            f"dca_curve_best_model_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_binary.png"
        )
        plt.savefig(dca_save_path, dpi=600, facecolor='white', bbox_inches='tight')
        plt.savefig(
            os.path.join(
                self.save_root, self.figure_dir,
                f"dca_curve_best_model_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_binary.svg"),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()
        logger.info(f"DCA curve saved to: {dca_save_path}")

        # -------------------------- 3. 最优模型特征重要性排名图（优化版：特征类型配色 + 百分比标注）--------------------------
        best_model_results = self.model_results.get(self.best_model_name, {})
        feature_importance = best_model_results.get('feature_importance', None)
        feature_names = best_model_results.get('feature_names', None)

        # 数据校验：仅当数据有效时绘制
        if feature_importance is not None and feature_names is not None and len(feature_importance) > 0 and len(
                feature_names) > 0:
            # 转换为数组，确保长度匹配
            feature_importance = np.array(feature_importance, dtype=np.float64)
            feature_names = np.array(feature_names)

            if len(feature_importance) != len(feature_names):
                logger.warning("Feature importance and feature names length mismatch, skip feature importance plot")
            else:
                # 归一化重要性为百分比（sum=100%）
                imp_sum = feature_importance.sum()
                if imp_sum > 0:
                    feature_importance_pct = feature_importance / imp_sum * 100
                else:
                    feature_importance_pct = feature_importance.copy()

                # 构建数据框，按重要性排序
                feat_df = pd.DataFrame({
                    'feature_name': feature_names,
                    'importance': feature_importance,
                    'importance_pct': feature_importance_pct
                }).sort_values(by='importance', ascending=False).reset_index(drop=True)

                # TopN特征（最多显示n_selected个，即选出的所有特征）
                top_n = min(self.n_selected, len(feat_df))
                feat_df_top = feat_df.head(top_n).iloc[::-1].reset_index(drop=True)  # 反转：最重要在顶部

                # 按特征类型分配颜色
                def get_feat_color(fname):
                    if fname in (self.bile_acid_features or []):
                        return COLOR_PALETTE['bile_acid']   # 紫色：胆汁酸
                    elif fname in (self.lipid_features or []):
                        return COLOR_PALETTE['lipid']       # 蓝色：脂质
                    else:
                        return COLOR_PALETTE['clinical']    # 橙色：临床指标

                bar_colors = [get_feat_color(n) for n in feat_df_top['feature_name']]

                # 自适应画布高度（每个特征0.42英寸，最小3英寸）
                fig_height = max(3.0, top_n * 0.42)
                fig, ax = plt.subplots(figsize=(6.5, fig_height))

                bars = ax.barh(
                    range(top_n),
                    feat_df_top['importance_pct'],
                    color=bar_colors,
                    alpha=0.85,
                    edgecolor='white',
                    linewidth=0.3,
                    height=0.65,
                    zorder=2
                )

                # 在每个条形末端标注百分比
                x_max_val = feat_df_top['importance_pct'].max()
                for bar, pct in zip(bars, feat_df_top['importance_pct']):
                    ax.text(
                        bar.get_width() + x_max_val * 0.015,
                        bar.get_y() + bar.get_height() / 2,
                        f'{pct:.1f}%',
                        ha='left', va='center',
                        fontsize=7, color='#333333', fontweight='bold',
                        zorder=3
                    )

                # Y轴特征名（过长则截断）
                y_labels = [n if len(n) <= 22 else f"{n[:19]}..." for n in feat_df_top['feature_name']]
                ax.set_yticks(range(top_n))
                ax.set_yticklabels(y_labels, fontsize=7.5)

                # 坐标轴
                ax.set_xlabel("Relative Importance (%)", fontsize=9, labelpad=8)
                ax.set_title(
                    f"Feature Importance — {self.best_model_name} (Top {top_n})",
                    fontsize=10, pad=10, fontweight='bold'
                )
                ax.set_xlim(0, x_max_val * 1.18)
                ax.tick_params(axis='x', labelsize=7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#CCCCCC')
                ax.grid(axis='x', color=grid_color, linestyle='-', linewidth=0.3, alpha=0.7, zorder=1)
                ax.set_axisbelow(True)

                # 图例：特征类型说明
                legend_patches = [
                    mpatches.Patch(color=COLOR_PALETTE['clinical'],  label='Clinical'),
                    mpatches.Patch(color=COLOR_PALETTE['bile_acid'], label='Bile Acid'),
                    mpatches.Patch(color=COLOR_PALETTE['lipid'],     label='Lipid'),
                ]
                ax.legend(handles=legend_patches, fontsize=7, loc='lower right',
                          frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
                          handlelength=1.2, borderpad=0.5)

                plt.tight_layout()
                feat_imp_save_path = os.path.join(
                    self.save_root, self.figure_dir,
                    f"feature_importance_best_model_{self.dimension_reduction_method}_{self.feature_selection_method}_binary.png"
                )
                plt.savefig(feat_imp_save_path, dpi=600, facecolor='white', bbox_inches='tight')
                plt.savefig(
                    os.path.join(self.save_root, self.figure_dir,
                        f"feature_importance_best_model_{self.dimension_reduction_method}_{self.feature_selection_method}_binary.svg"),
                    dpi=600, format='svg', bbox_inches='tight'
                )
                plt.close()
                logger.info(f"Feature importance plot saved to: {feat_imp_save_path}")

                # 保存CSV
                feat_imp_csv_path = os.path.join(
                    self.save_root, self.results_dir,
                    f"feature_importance_best_model_{self.dimension_reduction_method}_{self.feature_selection_method}_binary.csv"
                )
                feat_df.to_csv(feat_imp_csv_path, index=False, encoding='utf-8-sig')
                logger.info(f"Feature importance data saved to: {feat_imp_csv_path}")
        else:
            # 数据缺失时给出清晰警告，不中断流程
            missing_info = []
            if feature_importance is None: missing_info.append("'feature_importance'")
            if feature_names is None: missing_info.append("'feature_names'")
            if feature_importance is not None and len(feature_importance) == 0: missing_info.append("'feature_importance' is empty")
            if feature_names is not None and len(feature_names) == 0: missing_info.append("'feature_names' is empty")
            logger.warning(
                f"Skip feature importance plot: {', '.join(missing_info)}. "
                f"Hint: Ensure feature importance is extracted and stored correctly in model_results."
            )

        # -------------------------- 4. 最优模型混淆矩阵（保持原有逻辑，风格统一）--------------------------
        val_cm = np.array(self.best_model_metrics['val_confusion_matrix'])
        cm_normalized = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='YlGnBu', vmin=0, vmax=1)

        # 标注混淆矩阵数值（归一化值+原始值）
        for i in range(val_cm.shape[0]):
            for j in range(val_cm.shape[1]):
                text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{val_cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                        ha="center", va="center", color=text_color,
                        fontsize=8, fontweight='bold')

        # 风格优化（Nature紧凑风）
        ax.set_xlabel("Predicted Label", fontsize=9, labelpad=12)
        ax.set_ylabel("True Label", fontsize=9, labelpad=12)
        ax.set_title(
            f"Confusion Matrix of Best Model ({self.best_model_name}, Validation Set)",
            fontsize=10, pad=15
        )
        ax.set_xticks(np.arange(len(target_names)))
        ax.set_yticks(np.arange(len(target_names)))
        ax.set_xticklabels(target_names, rotation=0, ha='center', fontsize=8)
        ax.set_yticklabels(target_names, fontsize=8)

        # 边框优化
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')

        # 颜色条优化
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.12)
        cbar.set_label('Normalized Score', fontsize=8, labelpad=8)
        cbar.ax.tick_params(labelsize=7)

        # 保存图表
        plt.subplots_adjust(top=0.85, bottom=0.12, left=0.12, right=0.88)
        cm_save_path = os.path.join(
            self.save_root, self.figure_dir,
            f"confusion_matrix_best_model_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_binary.png"
        )
        plt.savefig(cm_save_path, dpi=600, facecolor='white')
        plt.savefig(
            os.path.join(
                self.save_root, self.figure_dir,
                f"confusion_matrix_best_model_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_binary.svg"),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()
        logger.info(f"Confusion matrix saved to: {cm_save_path}")

        # 全部完成日志
        logger.info("All available evaluation figures saved successfully.\n")

    # 新增：LIME模型解释（二分类适配）
    def _lime_model_explanation(self, X_train_top, X_val_top):
        """LIME tabular explanation for best model (binary classification, fix KeyError: 1)"""
        logger.info(f"\nPerforming LIME model explanation...")
        if self.best_model is None:
            logger.warning("No best model found, skip LIME explanation.")
            return

        # 1. 初始化LIME解释器（二分类适配）
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_top.astype(np.float64),
            feature_names=self.top_features,
            class_names=self.class_names,
            discretize_continuous=False,
            random_state=self.lime_random_state,
            verbose=False
        )

        # 2. 选择验证集中的典型样本（各1个，early/late）
        early_sample_idx = np.where(self.y_val == 0)[0][0]
        late_sample_idx = np.where(self.y_val == 1)[0][0]
        sample_indices = [early_sample_idx, late_sample_idx]

        for idx in sample_indices:
            sample = X_val_top[idx].astype(np.float64)
            true_label = self.y_val[idx]
            true_label_name = self.class_names[true_label]

            # 3. 生成LIME解释（二分类：不硬编码label，先获取有效标签）
            exp = lime_explainer.explain_instance(
                data_row=sample,
                predict_fn=lambda x: self.best_model.predict_proba(x),
                num_samples=self.lime_n_samples,
                num_features=self.lime_n_features,
                top_labels=1  # 保留top 1个预测标签
            )

            # 修复核心：获取LIME解释中实际存在的标签（避免硬编码label=1导致KeyError）
            valid_label = list(exp.local_exp.keys())[0] if exp.local_exp else true_label
            # 兜底：若仍无有效标签，使用样本真实标签
            valid_label = valid_label if valid_label in [0, 1] else true_label

            # 4. 绘制并保存LIME解释图（使用有效标签valid_label，替代硬编码的1）
            fig = exp.as_pyplot_figure(label=valid_label)
            fig.suptitle(
                f"LIME Explanation - Sample {idx} (True Label: {true_label_name}) | Best Model: {self.best_model_name}",
                fontsize=10, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir, f"lime_explanation_sample_{self.feature_selection_method}_{idx}_binary.png"),
                dpi=600, bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f"lime_explanation_sample_{self.feature_selection_method}_{idx}_binary.svg"),
                dpi=600,
                format='svg',
                bbox_inches='tight'
            )
            plt.close()

            # 5. 保存LIME解释结果（文本格式，使用有效标签valid_label）
            lime_results = exp.as_list(label=valid_label)
            lime_df = pd.DataFrame(lime_results, columns=['Feature_Explanation', 'Contribution_Score'])
            lime_df.to_csv(
                os.path.join(self.save_root, self.results_dir, f"lime_explanation_sample_{self.feature_selection_method}_{idx}_binary.csv"),
                index=False, encoding='utf-8-sig'
            )

        logger.info("LIME model explanation completed and results saved.")

    # def plot_suppl_fig7_model_feature_auc(self):
    #     """Supplementary Figure 7: 模型-特征选择组合AUC对比柱状图（Nature风格）"""
    #     logger.info("绘制Supplementary Figure 7: 模型-特征选择组合AUC对比")
    #     # 1. 构造16组组合数据（4模型×4特征选择，模拟/真实数据适配）
    #     # 若有真实组合结果，替换此处模拟数据；若无则基于现有cv_results扩展
    #     model_list = ['RF', 'SVM', 'LR', 'XGBoost']
    #     feature_selection_list = ['brute force', 'RFE', 'Lasso', 'SIS']
    #     combinations_labels = [f"{m}-{fs}" for m in model_list for fs in feature_selection_list]
    #
    #     # 模拟AUC和标准差（替换为真实cv_results中的mean/std）
    #     np.random.seed(42)
    #     auc_vals = np.array([0.89, 0.82, 0.78, 0.81, 0.75, 0.79, 0.76, 0.74,
    #                          0.77, 0.75, 0.80, 0.73, 0.83, 0.80, 0.79, 0.81])
    #     auc_std = auc_vals * 0.02 + np.random.randn(16) * 0.005  # 标准差
    #
    #     # 2. 定位最优组合（RF-brute force）
    #     best_idx = combinations_labels.index('RF-brute force')
    #     colors = ['#E64B35' if i == best_idx else '#B0B0B0' for i in range(len(combinations_labels))]
    #
    #     # 3. 绘制柱状图（Nature紧凑风格）
    #     fig, ax = plt.subplots(figsize=(10, 5))
    #     bars = ax.bar(
    #         range(len(combinations_labels)), auc_vals,
    #         yerr=auc_std, capsize=3, width=0.6, color=colors,
    #         edgecolor='black', linewidth=0.8, alpha=0.8
    #     )
    #
    #     # 4. 样式优化（Nature期刊要求）
    #     ax.set_xlabel('Model-Feature Selection Combination', fontsize=9, labelpad=10)
    #     ax.set_ylabel('Validation AUC', fontsize=9, labelpad=10)
    #     ax.set_title('Performance Comparison of Model-Feature Selection Combinations', fontsize=10, pad=15)
    #     ax.set_ylim(0.7, 0.95)  # 聚焦AUC范围
    #     ax.set_xticks(range(len(combinations_labels)))
    #     # 横轴标签换行（避免重叠）
    #     ax.set_xticklabels(
    #         [textwrap.fill(lab, 8) for lab in combinations_labels],
    #         rotation=45, ha='right', fontsize=7
    #     )
    #
    #     # 网格/边框优化
    #     ax.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3, alpha=0.7)
    #     ax.set_axisbelow(True)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.tick_params(axis='y', labelsize=8)
    #
    #     # 5. 标注最优组合文本
    #     ax.text(
    #         best_idx, auc_vals[best_idx] + auc_std[best_idx] + 0.01,
    #         'Best', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#E64B35'
    #     )
    #
    #     # 6. 保存
    #     plt.tight_layout()
    #     save_path = os.path.join(self.save_root, self.figure_dir, 'Supplementary_Figure7_Model_Feature_AUC.png')
    #     plt.savefig(save_path, dpi=600, facecolor='white', bbox_inches='tight')
    #     plt.savefig(
    #         os.path.join(self.save_root, self.figure_dir, 'Supplementary_Figure7_Model_Feature_AUC.svg'),
    #         dpi=600,
    #         format='svg',
    #         bbox_inches='tight'
    #     )
    #     plt.close()
    #     logger.info(f"Supplementary Figure 7已保存至: {save_path}")

    def plot_suppl_fig8_core_features_boxplot(self):
        """Supplementary Figure 8: 核心特征早/晚期PBC分布箱线图（带显著性标记）"""
        logger.info("绘制Supplementary Figure 8: 核心特征早/晚期箱线图")
        # 1. 筛选7个核心特征（优先用self.top_features前7个）
        core_features = self.top_features[:7] if len(self.top_features) >= 7 else self.top_features
        if len(core_features) < 1:
            logger.warning("无核心特征，跳过Supplementary Figure 8绘制")
            return

        # 2. 提取早/晚期数据（early:0, late:1）
        train_data_df = pd.DataFrame(self.X_train_scaled, columns=self.feature_columns)
        train_data_df['stage'] = self.y_train  # 0=early,1=late
        early_data = train_data_df[train_data_df['stage'] == 0][core_features]
        late_data = train_data_df[train_data_df['stage'] == 1][core_features]

        # ========== 关键优化1：计算全局最大值，用于统一y轴高度 ==========
        # 收集所有特征的早晚期数据最大值
        all_max_vals = []
        for feat in core_features:
            early_max = early_data[feat].dropna().max() if not early_data[feat].dropna().empty else 0
            late_max = late_data[feat].dropna().max() if not late_data[feat].dropna().empty else 0
            all_max_vals.append(max(early_max, late_max))
        # 全局最大值（乘以1.1，预留显著性标记的空间）
        global_y_max = max(all_max_vals) * 1.1 if all_max_vals else 1.0

        # 3. 绘制多子图箱线图（Nature风格）
        n_feat = len(core_features)
        # ========== 微调画布宽度，避免特征名称挤压 ==========
        fig, axes = plt.subplots(1, n_feat, figsize=(2.5 * n_feat, 4), sharey=False)  # 宽度从2→2.5
        if n_feat == 1: axes = [axes]

        # 定义早晚期配色（提取出来方便复用）
        box_colors = ['#4DBBD5', '#3C5488']  # Early=浅蓝色, Late=深蓝色

        for idx, feat in enumerate(core_features):
            ax = axes[idx]
            # 准备数据
            data_to_plot = [early_data[feat].dropna(), late_data[feat].dropna()]

            # 核心修复部分
            bp = ax.boxplot(
                data_to_plot, labels=['Early', 'Late'], patch_artist=True,
                # boxprops改为全局字典（仅设置公共样式）
                boxprops=dict(edgecolor='black', linewidth=1.0, alpha=0.8),
                whiskerprops=dict(color='black', linewidth=0.8),
                capprops=dict(color='black', linewidth=0.8),
                medianprops=dict(color='#E64B35', linewidth=1.2),  # 红色加粗中位数线
                flierprops=dict(marker='o', markersize=2, color='black', alpha=0.5)
            )

            # 遍历bp['boxes']，逐个设置早晚期箱线的填充色
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(box_colors[i])  # 设置填充色

            # 显著性标注（保留原有逻辑）
            if len(data_to_plot[0]) > 0 and len(data_to_plot[1]) > 0:  # 避免空数据报错
                stat, p_val = stats.mannwhitneyu(data_to_plot[0], data_to_plot[1], alternative='two-sided')
                if p_val < 0.001:
                    sig_label = '***'
                elif p_val < 0.01:
                    sig_label = '**'
                elif p_val < 0.05:
                    sig_label = '*'
                else:
                    sig_label = 'ns'
                # ========== 调整显著性标注位置，基于全局y轴最大值 ==========
                ax.text(1.5, global_y_max * 0.98, sig_label, ha='center', va='bottom', fontsize=8, fontweight='bold')
            else:
                logger.warning(f"特征{feat}的早/晚期数据为空，跳过显著性标注")

            # ========== 关键优化2：优化特征名称显示（减少不必要换行） ==========
            # 调整换行宽度（从10→15），让名称尽量一行显示；若过长则自动换行，且居中
            wrapped_feat = textwrap.fill(feat, width=15)  # 宽度从10→15
            ax.set_title(wrapped_feat, fontsize=8, pad=5, ha='center')  # 强制居中对齐

            # 子图样式优化
            ax.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3, alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7)

            # ========== 关键优化3：统一所有子图的y轴上限 ==========
            ax.set_ylim(bottom=0, top=global_y_max)  # 统一y轴范围

            # 简化x/y轴标签（仅保留第一个/最后一个子图的标签，减少冗余）
            if idx == 0:  # 第一个子图显示y轴标签
                ax.set_ylabel("Normalized Feature Value", fontsize=8)
            else:
                ax.set_ylabel("")

            # 所有子图的x轴标签字体调小，避免挤压
            ax.set_xlabel("")  # 隐藏x轴标签（Early/Late已在boxplot中显示）
            ax.tick_params(axis='x', labelsize=6)  # x轴刻度字体更小

        # 全局标题/标签优化
        fig.suptitle('Core Features Distribution in Early/Late PBC', fontsize=10, y=1.02)
        # 统一设置底部全局x轴标签，替代每个子图的x轴标签
        fig.text(0.5, 0.005, 'PBC Stage Group', ha='center', fontsize=9)
        # 移除重复的y轴全局标签（已在第一个子图显示）

        # 保存（调整tight_layout参数，避免标题被裁剪）
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # 预留顶部标题空间
        save_path = os.path.join(self.save_root, self.figure_dir, 'Supplementary_Figure8_Core_Features_Boxplot.png')
        plt.savefig(save_path, dpi=600, facecolor='white', bbox_inches='tight')
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir, 'Supplementary_Figure8_Core_Features_Boxplot.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()
        logger.info(f"Supplementary Figure 8已保存至: {save_path}")

    def plot_suppl_fig9_core_features_correlation(self):
        """Supplementary Figure 9: 核心特征相关性热力图（验证多重共线性）"""
        logger.info("绘制Supplementary Figure 9: 核心特征相关性热力图")
        # 1. 筛选7个核心特征
        core_features = self.top_features[:7] if len(self.top_features) >= 7 else self.top_features
        if len(core_features) < 2:
            logger.warning("核心特征不足2个，跳过Supplementary Figure 9绘制")
            return

        # 2. 计算皮尔逊相关系数
        train_data_df = pd.DataFrame(self.X_train_scaled, columns=self.feature_columns)
        corr_matrix = train_data_df[core_features].corr(method='pearson')

        # 3. 绘制热力图（Nature风格，突出r<0.7）
        fig, ax = plt.subplots(figsize=(6, 5))
        # 掩码：隐藏r≥0.7的区域（标注警告）
        mask = np.abs(corr_matrix) >= 0.7
        corr_masked = corr_matrix.mask(mask)

        # 热力图绘制
        im = ax.imshow(
            corr_masked, cmap='YlGnBu', vmin=-1, vmax=1,
            aspect='auto', interpolation='none'
        )

        # 标注相关系数
        for i in range(len(core_features)):
            for j in range(len(core_features)):
                if i == j:
                    # 对角线：标注1
                    ax.text(j, i, '1.00', ha='center', va='center', fontsize=8, fontweight='bold')
                elif not mask.iloc[i, j]:
                    # 非掩码区域：标注r值
                    val = corr_matrix.iloc[i, j]
                    text_color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=text_color)
                else:
                    # 掩码区域：标注警告
                    ax.text(j, i, '≥0.7', ha='center', va='center', fontsize=7, color='red', fontweight='bold')

        # 样式优化
        ax.set_xticks(range(len(core_features)))
        ax.set_yticks(range(len(core_features)))
        ax.set_xticklabels([textwrap.fill(f, 15) for f in core_features], rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels([textwrap.fill(f, 15) for f in core_features], fontsize=7)
        ax.set_title('Core Features Pearson Correlation (r < 0.7)', fontsize=10, pad=15)

        # 颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Pearson Correlation Coefficient', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # 边框/网格
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

        # 保存
        plt.tight_layout()
        save_path = os.path.join(self.save_root, self.figure_dir, 'Supplementary_Figure9_Core_Features_Correlation.png')
        plt.savefig(save_path, dpi=600, facecolor='white', bbox_inches='tight')
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir, 'Supplementary_Figure9_Core_Features_Correlation.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()
        logger.info(f"Supplementary Figure 9已保存至: {save_path}")

    def plot_suppl_fig10_calibration_curve(self):
        """Supplementary Figure 10: 最优模型校准曲线（Nature风格）"""
        logger.info("绘制Supplementary Figure 10: 最优模型校准曲线")
        if self.best_model is None:
            logger.warning("无最优模型，跳过Supplementary Figure 10绘制")
            return

        # 1. 提取预测概率和真实标签（修复：使用标准化后的原始数据，而非降维后数据）
        top_indices = [self.feature_columns.index(f) for f in self.top_features]
        X_val_top = self.X_val_scaled[:, top_indices]
        y_val_proba = self.best_model.predict_proba(X_val_top)[:, 1]  # late概率
        y_val_true = self.y_val

        # 2. 计算校准曲线（分10个区间）
        n_bins = 10
        prob_true, prob_pred = calibration_curve(
            y_val_true, y_val_proba, n_bins=n_bins, strategy='quantile'
        )

        # 计算校准误差（ECE：预期校准误差）- 根据实际返回的bin数计算权重
        brier_score = brier_score_loss(y_val_true, y_val_proba)
        actual_n_bins = len(prob_true)  # calibration_curve实际返回的bin数（可能<n_bins）
        bin_edges = np.percentile(y_val_proba, np.linspace(0, 100, actual_n_bins + 1))
        bin_counts = np.histogram(y_val_proba, bins=bin_edges)[0]
        ece = np.sum(np.abs(prob_true - prob_pred) * bin_counts / len(y_val_proba))

        # 3. 绘制校准曲线（同时显示未校准与Isotonic校准后曲线）
        # 对最优模型做Isotonic后校准（使用训练集fit，验证集evaluate）
        X_train_top_calib = self.X_train_scaled[:, top_indices]
        calibrated_model = CalibratedClassifierCV(
            self.best_model, method='isotonic', cv='prefit'
        )
        calibrated_model.fit(X_train_top_calib, self.y_train)
        y_val_proba_calib = calibrated_model.predict_proba(X_val_top)[:, 1]
        prob_true_calib, prob_pred_calib = calibration_curve(
            y_val_true, y_val_proba_calib, n_bins=n_bins, strategy='quantile'
        )
        brier_score_calib = brier_score_loss(y_val_true, y_val_proba_calib)

        fig, ax = plt.subplots(figsize=(5, 5))
        # 理想线（y=x）
        ax.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1, label='Perfect Calibration')
        # # 未校准模型曲线
        # ax.plot(prob_pred, prob_true, marker='o', color='#4DBBD5', linewidth=1.5, markersize=4,
        #         linestyle='--', alpha=0.8, label=f'{self.best_model_name} (Brier={brier_score:.3f})')
        # Isotonic校准后曲线
        ax.plot(prob_pred_calib, prob_true_calib, marker='s', color='#E64B35', linewidth=2, markersize=4,
                label=f'{self.best_model_name} + Isotonic (Brier={brier_score_calib:.3f})')

        # 计算校准后的ECE - 同样根据实际返回的bin数计算权重
        actual_n_bins_calib = len(prob_true_calib)
        bin_edges_calib = np.percentile(y_val_proba_calib, np.linspace(0, 100, actual_n_bins_calib + 1))
        bin_counts_calib = np.histogram(y_val_proba_calib, bins=bin_edges_calib)[0]
        ece_calib = np.sum(np.abs(prob_true_calib - prob_pred_calib) * bin_counts_calib / len(y_val_proba_calib))

        # 样式优化
        ax.set_xlabel('Predicted Probability of Late PBC', fontsize=9, labelpad=10)
        ax.set_ylabel('Actual Observed Probability', fontsize=9, labelpad=10)
        # ax.set_title(f'Calibration Curve\n(ECE={ece:.3f} → {ece_calib:.3f} after Isotonic Calibration)', fontsize=9, pad=12)
        ax.set_title(f'Calibration Curve)', fontsize=9,pad=12)
        ax.legend(fontsize=8, loc='best')
        ax.grid(axis='both', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)

        # 保存
        plt.tight_layout()
        save_path = os.path.join(self.save_root, self.figure_dir, 'Supplementary_Figure10_Calibration_Curve.png')
        plt.savefig(save_path, dpi=600, facecolor='white', bbox_inches='tight')
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir, 'Supplementary_Figure10_Calibration_Curve.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()
        logger.info(f"Supplementary Figure 10已保存至: {save_path}")

    # def plot_mfuzz_metabolic_clustering(self, differential_metabolites):
    #     """
    #     Mfuzz聚类：按代谢变化相似性分组，绘制GC进展中的代谢轨迹 + 代表性代谢物标注
    #     :param differential_metabolites: 差异代谢物列表（需是数据中存在的列名）
    #     """
    #     logger.info("\nStep X: Mfuzz clustering for metabolic trajectories...")
    #     import numpy as np
    #     import pandas as pd
    #     from Mfuzz import ExpressionSet, mfuzz, mestimate  # Mfuzz核心工具
    #     from sklearn.preprocessing import StandardScaler
    #
    #     # ========== 1. 数据准备：提取代谢物-阶段的表达矩阵 ==========
    #     # （关键：需根据你的数据调整阶段映射！这里假设group_label对应阶段：0→NGC, 1→I, 2→II, 3→III, 4→IV）
    #     def map_group_to_stage(group_label):
    #         stage_map = {0: "I", 1: "II", 2: "III", 3: "IV"}
    #         return stage_map.get(group_label, "Unknown")
    #
    #     self.raw_data["stage"] = self.raw_data["group_label"].apply(map_group_to_stage)
    #
    #     # 筛选差异代谢物列（过滤数据中不存在的）
    #     valid_mets = [met for met in differential_metabolites if met in self.raw_data.columns]
    #     if not valid_mets:
    #         logger.error("无匹配的差异代谢物列！请检查输入列表")
    #         return
    #
    #     # 按阶段分组，计算每个代谢物的阶段均值（每个阶段的平均表达）
    #     stage_order = ["I", "II", "III", "IV"]  # 固定阶段顺序
    #     stage_mean_df = self.raw_data.groupby("stage")[valid_mets].mean().reindex(stage_order)
    #     expr_matrix = stage_mean_df.T.values  # 转置：行=代谢物，列=阶段
    #     met_names = valid_mets  # 代谢物名
    #     stage_names = stage_order  # 阶段名
    #
    #     # ========== 2. Mfuzz数据标准化 + 格式转换 ==========
    #     # Mfuzz要求输入z-score标准化后的数据
    #     scaler = StandardScaler()
    #     expr_scaled = scaler.fit_transform(expr_matrix)
    #
    #     # 转换为Mfuzz要求的ExpressionSet对象
    #     eset = ExpressionSet(
    #         data=expr_scaled,
    #         gene_names=met_names,
    #         sample_names=stage_names
    #     )
    #
    #     # ========== 3. Mfuzz聚类参数估计 + 聚类 ==========
    #     m = mestimate(eset)  # 自动估计模糊系数m
    #     logger.info(f"Mfuzz模糊系数m: {m:.2f}")
    #
    #     num_clusters = 3  # 对应你图中的3个Cluster（可根据实际调整）
    #     clusters = mfuzz(eset, c=num_clusters, m=m)  # 执行模糊聚类
    #
    #     # ========== 4. 绘制聚类轨迹图（匹配你提供的图表样式） ==========
    #     fig, axes = plt.subplots(num_clusters, 1, figsize=(8, 4 * num_clusters), sharex=True)
    #     axes = axes.flatten() if num_clusters > 1 else [axes]
    #     cluster_colors = [COLOR_PALETTE["class1"], COLOR_PALETTE["class2"], COLOR_PALETTE["class3"]]  # 3个Cluster的颜色
    #
    #     for idx in range(num_clusters):
    #         ax = axes[idx]
    #         # 提取该Cluster中隶属度>0.5的代谢物（核心成员）
    #         cluster_met_idxs = np.where(clusters.membership[:, idx] > 0.5)[0]
    #         cluster_mets = [met_names[i] for i in cluster_met_idxs]
    #         cluster_expr = expr_scaled[cluster_met_idxs, :]  # 该Cluster的代谢物表达
    #
    #         # 绘制每个代谢物的轨迹（淡色）
    #         for expr in cluster_expr:
    #             ax.plot(stage_names, expr, color=cluster_colors[idx], alpha=0.6, linewidth=1)
    #         # 绘制Cluster中心轨迹（黑色粗线）
    #         ax.plot(stage_names, clusters.center[idx, :], color="black", linewidth=2.5, label="Cluster Center")
    #
    #         # 图表样式调整
    #         ax.set_title(f"Cluster {idx+1}", fontsize=11, pad=10)
    #         ax.set_ylabel("Relative Abundance (Z-score)", fontsize=9)
    #         ax.legend(fontsize=8, loc="best")
    #         ax.grid(axis="y", color=COLOR_PALETTE["grid"], linestyle="-", linewidth=0.3)
    #         ax.spines["top"].set_visible(False)
    #         ax.spines["right"].set_visible(False)
    #
    #     # 统一设置X轴（阶段）
    #     axes[-1].set_xlabel("Stage", fontsize=9)
    #
    #     # ========== 5. 在图右侧添加每个Cluster的代表性代谢物 ==========
    #     fig.subplots_adjust(right=0.78)  # 预留右侧空间
    #     legend_ax = fig.add_axes([0.8, 0.1, 0.18, 0.8])  # 右侧文本轴
    #     legend_ax.axis("off")  # 隐藏坐标轴
    #
    #     for idx in range(num_clusters):
    #         # 提取该Cluster的前5个代表性代谢物
    #         cluster_met_idxs = np.where(clusters.membership[:, idx] > 0.5)[0]
    #         cluster_mets = [met_names[i] for i in cluster_met_idxs][:5]  # 取前5个
    #         # 绘制文本
    #         legend_ax.text(
    #             0, 1 - (idx / num_clusters),
    #                f"Cluster {idx+1}\n" + "\n".join(cluster_mets),
    #             fontsize=8, verticalalignment="top"
    #         )
    #
    #     # 保存图片
    #     save_path = os.path.join(self.save_root, self.figure_dir,
    #                              "mfuzz_metabolic_trajectory_clustering.png")
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=600, bbox_inches="tight")
    #     plt.close()
    #     logger.info(f"Mfuzz聚类图已保存至: {save_path}")

    def write_predictions_to_excel(self):
        """将模型预测结果写入原始Excel并保存新文件到结果目录（修复核心逻辑错误）"""
        logger.info("\nWriting prediction results back to original Excel sheet...")

        # 1. 准备最佳模型的预测结果（训练集+验证集）
        try:
            top_indices = [self.feature_columns.index(f) for f in self.top_features if
                           f in self.feature_columns]
        except ValueError as e:
            logger.error(f"特征名称匹配失败：{e}，跳过预测结果写入")
            return

        # 训练集预测
        X_train_top = self.X_train_scaled[:, top_indices]
        train_pred = self.best_model.predict(X_train_top)
        train_pred_proba = self.best_model.predict_proba(X_train_top)

        # 验证集预测
        X_val_top = self.X_val_scaled[:, top_indices]
        val_pred = self.best_model.predict(X_val_top)
        val_pred_proba = self.best_model.predict_proba(X_val_top)

        # 2. 映射预测标签到类别名称（early/late）
        train_pred_names = [self.class_name_mapping[p] for p in train_pred]
        val_pred_names = [self.class_name_mapping[p] for p in val_pred]

        # 3. 给训练/验证集添加预测列（保留核心ID列）
        # 训练集添加预测结果
        train_data_with_pred = self.train_data.copy()
        train_data_with_pred['ML_Predicted_Label'] = train_pred
        train_data_with_pred['ML_Predicted_Class'] = train_pred_names
        # 添加各类别概率列（二分类：early/late）
        for i, cls_name in enumerate(self.class_names):
            train_data_with_pred[f'ML_Prob_{cls_name}'] = train_pred_proba[:, i]

        # 验证集添加预测结果
        val_data_with_pred = self.val_data.copy()
        val_data_with_pred['ML_Predicted_Label'] = val_pred
        val_data_with_pred['ML_Predicted_Class'] = val_pred_names
        for i, cls_name in enumerate(self.class_names):
            val_data_with_pred[f'ML_Prob_{cls_name}'] = val_pred_proba[:, i]

        # 4. 合并训练/验证集并匹配原始数据顺序（按patient_id）
        combined_pred = pd.concat([train_data_with_pred, val_data_with_pred], ignore_index=True)
        # 只保留需要merge的列，避免重复列
        merge_cols = ['patient_id', 'ML_Predicted_Label', 'ML_Predicted_Class'] + \
                     [f'ML_Prob_{cls_name}' for cls_name in self.class_names]
        raw_data_with_pred = self.raw_data.merge(
            combined_pred[merge_cols],
            on='patient_id',
            how='left'  # 保留原始数据所有样本，无预测结果的填充NaN
        )

        # 5. 保存新Excel文件到结果目录
        output_excel_path = os.path.join(
            self.save_root, self.results_dir,
            f'prediction_results_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}.xlsx'
        )

        # ========== 核心修复1：正确转换原始标签为二分类标签（用于准确率计算） ==========
        def convert_original_to_binary(label):
            """将原始group_label（1/2/3/4）转换为二分类标签（0=early,1=late）"""
            if label in [1, 2]:
                return 0
            elif label in [3, 4]:
                return 1
            else:
                return np.nan

        # 新增二分类真实标签列
        raw_data_with_pred['True_Binary_Label'] = raw_data_with_pred['group_label'].apply(convert_original_to_binary)
        # 核心修复2：正确映射真实类别名称
        raw_data_with_pred['True_Binary_Class'] = raw_data_with_pred['True_Binary_Label'].map(
            self.class_name_mapping).fillna('Unknown')

        # 6. 写入原sheet（修复硬编码的sheet名称）+ 新增汇总sheet
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            # 原sheet保留所有列 + 预测列
            raw_data_with_pred.to_excel(writer, sheet_name='analysis_2_184', index=False)  # 修复为原始sheet名称

            # 新增预测汇总sheet（更易查看）
            prediction_summary = pd.DataFrame({
                'Patient_ID': raw_data_with_pred['patient_id'],
                'Dataset_Type': raw_data_with_pred['dataset_type'],
                'Original_True_Label': raw_data_with_pred['group_label'],
                'True_Binary_Label': raw_data_with_pred['True_Binary_Label'],
                'True_Binary_Class': raw_data_with_pred['True_Binary_Class'],
                'ML_Predicted_Label': raw_data_with_pred['ML_Predicted_Label'],
                'ML_Predicted_Class': raw_data_with_pred['ML_Predicted_Class'],
                **{f'ML_Prob_{cls_name}': raw_data_with_pred[f'ML_Prob_{cls_name}']
                   for cls_name in self.class_names}
            })

            # ========== 核心修复3：正确计算预测准确率 ==========
            # 仅计算有预测结果且真实标签有效的样本
            valid_mask = (
                raw_data_with_pred['True_Binary_Label'].notna() &
                raw_data_with_pred['ML_Predicted_Label'].notna()
            )
            # 计算正确预测的样本
            correct_pred = pd.Series(False, index=raw_data_with_pred.index)
            correct_pred[valid_mask] = (
                raw_data_with_pred.loc[valid_mask, 'True_Binary_Label'] ==
                raw_data_with_pred.loc[valid_mask, 'ML_Predicted_Label']
            )
            prediction_summary['Is_Correct'] = correct_pred
            prediction_summary.to_excel(writer, sheet_name='Prediction_Summary', index=False)

        # ========== 修复准确率统计逻辑 ==========
        total_valid = valid_mask.sum()
        total_correct = correct_pred.sum()
        accuracy = total_correct / total_valid if total_valid > 0 else 0.0

        logger.info(f"✅ 预测结果已保存至: {output_excel_path}")
        logger.info(f"📊 有效预测样本数: {total_valid} (总样本数: {len(raw_data_with_pred)})")
        logger.info(f"📈 预测准确率（仅有效样本）: {accuracy:.4f} ({total_correct}/{total_valid})")
        logger.info(f"❌ 预测错误样本数: {total_valid - total_correct}")
        logger.info(f"🔍 无预测结果样本数: {len(raw_data_with_pred) - total_valid}")

    # 新增：流程总结（最终步骤）
    def run_full_pipeline(self):
        """Run the full medical data analysis pipeline (binary classification)"""
        logger.info("=" * 80)
        logger.info("Starting Full Autoimmune Liver Disease Binary Classification Pipeline")
        logger.info("=" * 80)

        # 按步骤执行完整流程
        self.load_data() \
            .preprocess_data() \
            .filter_lipid_features() \
            .reduce_dimension() \
            .select_top_features() \
            .grid_search_with_cv() \
            .evaluate_models()
            # .plot_mfuzz_metabolic_clustering(differential_metabolites=self.top_features)

        # ========== 新增：调用Supplementary Figure 7-10绘制函数 ==========
        # self.plot_suppl_fig7_model_feature_auc()
        self.plot_suppl_fig8_core_features_boxplot()
        self.plot_suppl_fig9_core_features_correlation()
        self.plot_suppl_fig10_calibration_curve()

        # ========== 新增调用：写入预测结果到Excel ==========
        self.write_predictions_to_excel()

        logger.info("=" * 80)
        logger.info("Full Pipeline Completed Successfully!")
        logger.info(f"All results saved to: {self.save_root}")
        logger.info("=" * 80)


logger = None

def main():
    global logger
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 配置日志
    logger = setup_logging(
        args,
        dimension_reduction_method=args.dimension_reduction_method,
        feature_selection_method=args.feature_selection_method
    )

    # 3. 初始化分析器并运行完整流程
    analyzer = MedicalDataAnalyzer(args)
    analyzer.run_full_pipeline()

if __name__ == "__main__":
    main()