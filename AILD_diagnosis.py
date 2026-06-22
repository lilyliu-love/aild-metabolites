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
import random  # Limit brute force combinations
import torch  # GPU detection
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

warnings.filterwarnings('ignore', category=FutureWarning)  # 屏蔽IterativeImputer警告
warnings.filterwarnings('ignore')


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
    parser = argparse.ArgumentParser(description='Autoimmune Liver Disease Diagnostic Model Pipeline')

    # 基本配置
    parser.add_argument('--excel_path', type=str, default="./metabolism_analysis_updated_IgMold.xlsx",   #./metabolism_analysis_updated_IgMold.xlsx
                        help='Excel文件路径')
    parser.add_argument('--n_selected', type=int, default=11,
                        help='最终选择的特征数量')
    parser.add_argument('--is_filter', type=bool, default=True,
                        help='是否进行特征过滤')

    parser.add_argument('--save_root', type=str, default="analysis 2 11 features(0427-2)")
    parser.add_argument('--figure_dir', type=str, default="figure_3",
                        help='是否进行特征过滤')
    parser.add_argument('--results_dir', type=str, default="results_3",
                        help='是否进行特征过滤')
    parser.add_argument('--model_dir', type=str, default="saved_best_model",
                        help='是否进行特征过滤')
    parser.add_argument('--filter_dir', type=str, default="filtered_lipid_data",
                        help='保存目录')

    # 降维配置
    parser.add_argument('--dimension_reduction_method', type=str, default='none',
                        choices=['none', 'pca', 'selectkbest'],
                        help='降维方法')
    parser.add_argument('--pca_variance_ratio', type=float, default=0.8,
                        help='PCA保留方差比例')
    parser.add_argument('--selectkbest_k', type=int, default=100,
                        help='SelectKBest选择的特征数量')

    # 特征选择配置
    parser.add_argument('--feature_selection_method', type=str, default='shap',
                        choices=['specified', 'brute_force', 'shap', 'lasso', 'sis'],
                        help='特征选择方法')
    parser.add_argument('--specified_features', type=str, nargs='+',
                        # default=['ALP(35-100)', 'AMA-M2', 'ALT',
                        #          'PE(16:0/16:0)', 'PE(16:0/18:1)',
                        #          'Glycerophosphocholine', 'Glycoursodeoxycholic acid (GUDCA)'],
                        # default=['PC(18:0/19:0)', 'LPC(16:1)','Glycoursodeoxycholic acid (GUDCA)'],
                        # default=['ALP(35-100)', 'GGT(4-50)', 'ANA', 'IgM', "AMA-M2", "IgG", "AST", "ALT"],
                        default=['ALP(35-100)', 'GGT(4-50)', 'ANA', 'IgM', 'PC(18:0/19:0)', 'LPC(16:1)','Glycoursodeoxycholic acid (GUDCA)'],
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
    parser.add_argument('--save_filtered_data', type=bool, default=False,
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
    'IgG', 'IgM'  #'DBIL','ALB', 'GLO', 'TBA'
]

BASIC_FEATURES = [] #['Sex', 'Age', 'BMI']
CLINICAL_FEATURES = ANTIBODY_FEATURES + LIVER_FUNCTION_FEATURES + BASIC_FEATURES
LIPID_FEATURES_RANGE = (0, 755)  # 脂质代谢物列范围
BILE_ACID_FEATURES_RANGE = (755, 866)  # 胆汁酸代谢物列范围


# ====================== Logging Configuration ======================
def setup_logging(args, dimension_reduction_method, feature_selection_method):
    # 创建输出目录
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(os.path.join(args.save_root,args.figure_dir), exist_ok=True)
    os.makedirs(os.path.join(args.save_root,args.results_dir), exist_ok=True)
    os.makedirs(os.path.join(args.save_root,args.model_dir), exist_ok=True)
    os.makedirs(os.path.join(args.save_root,args.filter_dir), exist_ok=True)
    os.makedirs('results_3', exist_ok=True)
    log_filename = os.path.join(os.path.join(args.save_root, args.results_dir),
                                f'run_log_{dimension_reduction_method}_{feature_selection_method}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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
    'AMA-M2_1': '#1f77b4',    # AMA-M2阳性
    'AMA-M2_0': '#ff7f0e'    # AMA-M2阴性
}


# ====================== Delong Test Implementation (固定不变) =======================
class DelongTest:
    """Delong test for AUC significance comparison (binary classification)"""

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
        """Fixed net benefit calculation (per 100 samples)"""
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
        """Fixed DCA analysis (avoid division by zero)"""
        dca_results = []
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

            treat_none_nb = 0.0

            dca_results.append({
                'threshold': threshold,
                'model_net_benefit': model_nb,
                'treat_all_net_benefit': treat_all_nb,
                'treat_none_net_benefit': treat_none_nb
            })

        return pd.DataFrame(dca_results)


# ====================== AUC Confidence Interval Calculation (固定不变) =======================
def roc_auc_macro(y_true, y_score):
    """
    多分类 macro-average AUC（OvR）：每类单独计算 AUC 后取均值。
    对每种疾病的诊断能力给予相同权重，适合临床多分类诊断场景。
    等价于 roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')。
    """
    return roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')


def calculate_auc_ci(y_true, y_score, n_bootstrap=1000, ci=0.95):
    np.random.seed(42)
    n_samples = len(y_true)

    if len(y_score.shape) == 2:
        aucs_list = []
        for i in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_score_boot = y_score[indices]
            try:
                aucs_list.append(roc_auc_macro(y_true_boot, y_score_boot))
            except Exception:
                aucs_list.append(0.5)
        aucs = np.array(aucs_list)
    else:
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


def calculate_clinical_auc_ci(y_true, clinical_proba, n_bootstrap=1000, ci=0.95):
    """计算临床指南诊断结果的AUC 95%置信区间（适配one-hot概率）"""
    np.random.seed(42)
    n_samples = len(y_true)
    aucs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        clinical_proba_boot = clinical_proba[indices]

        try:
            auc = roc_auc_macro(y_true_boot, clinical_proba_boot)
            aucs.append(auc)
        except:
            aucs.append(0.5)

    aucs = np.array(aucs)
    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    return lower, upper


# ====================== Core Analysis Class =======================
class MedicalDataAnalyzer:
    def __init__(self, args):
        """初始化分析器，从args获取所有配置"""
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
        # 基础模型配置
        self.brute_force_base_model = XGBClassifier(
            random_state=42, eval_metric='mlogloss',
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
        self.n_classes = 4
        self.label_offset = 1
        self.class_name_mapping = {0: 'AIH', 1: 'PBC', 2: 'OS', 3: 'CTR'}
        self.class_names = ['AIH', 'PBC', 'OS', 'CTR']
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_train_reduced = None
        self.X_val_reduced = None
        self.X_train_imputed_df = None  # 插补后、标准化前的原始尺度数据（用于可视化）
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

    def load_data(self):
        logger.info("Step 1/7: Loading data...")
        self.raw_data = pd.read_excel(self.excel_path, sheet_name="analysis_2_584")
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

        self.y_train = (self.train_data['group_label'].astype(np.int32) - self.label_offset).values
        self.y_val = (self.val_data['group_label'].astype(np.int32) - self.label_offset).values

        assert set(self.y_train) == {0, 1, 2, 3}, f"Invalid training labels after conversion: {set(self.y_train)}"

        logger.info(f"Training samples: {len(self.train_data)}, Validation samples: {len(self.val_data)}")
        train_dist = np.bincount(self.y_train)
        val_dist = np.bincount(self.y_val)
        logger.info("Label distribution - Train:")
        for i, name in self.class_name_mapping.items():
            logger.info(f"  {name}: {train_dist[i]}")
        logger.info("Label distribution - Validation:")
        for i, name in self.class_name_mapping.items():
            logger.info(f"  {name}: {val_dist[i]}")
        return self

    def _perform_pairwise_mannwhitney(self, feature_data, class_labels, class_names):
        """
         pairwise Mann-Whitney U检验（非参数检验）+ Benjamini-Hochberg校正
        返回每个类别与对照组（CTR，标签3）的显著性结果
        """
        ctrl_mask = class_labels == 3  # CTR作为对照组
        if not np.any(ctrl_mask):
            return {}

        ctrl_data = feature_data[ctrl_mask]
        sig_results = {}

        for cls_idx, cls_name in enumerate(class_names[:-1]):  # 排除CTR自身
            cls_mask = class_labels == cls_idx
            if not np.any(cls_mask):
                sig_results[cls_name] = 'ns'
                continue

            cls_data = feature_data[cls_mask]
            # Mann-Whitney U检验
            stat, p_val = stats.mannwhitneyu(cls_data, ctrl_data, alternative='two-sided')
            sig_results[cls_name] = p_val

        # Benjamini-Hochberg校正
        p_values = list(sig_results.values())
        if len(p_values) > 0:
            from statsmodels.stats.multitest import multipletests
            _, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
            for i, cls_name in enumerate(class_names[:-1]):
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
        plt.savefig(os.path.join(self.save_root,self.figure_dir,f'imputation_comparison_{self.feature_type_filter}.png'),dpi=600, bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir, f'imputation_comparison_{self.feature_type_filter}.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
            )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir, f'imputation_comparison_{self.feature_type_filter}.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

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
            self.liver_group_median = df_imputed.groupby(self.train_data['group_label'])[liver_cols].median()
        for col in liver_cols:
            for group in self.liver_group_median.index:
                mask = (self.train_data['group_label'] == group) if is_train else (
                    self.val_data['group_label'] == group)
                df_imputed.loc[mask, col] = df_imputed.loc[mask, col].fillna(self.liver_group_median.loc[group, col])
        logger.info(f"Imputed liver function features ({len(liver_cols)}): group median")

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

        # 保存插补后、标准化前的数据，供可视化使用（原始生物学尺度）
        self.X_train_imputed_df = X_train_imputed.copy()

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

    # def filter_lipid_features(self):
    #     """Lipid + bile acid feature filtering (balanced selection)"""
    #     logger.info("\nPre-filtering lipid + bile acid features (balanced selection)")
    #     if self.X_train_reduced is None:
    #         self.X_train_reduced = self.X_train_scaled.copy()
    #     if self.X_val_reduced is None:
    #         self.X_val_reduced = self.X_val_scaled.copy()
    #
    #     if self.args.is_filter:
    #         self.reduced_feature_names = self.feature_columns
    #
    #         # 1. Split lipid/bile acid/clinical features
    #         lipid_cols = [f for f in self.lipid_features if f in self.reduced_feature_names]
    #         bile_acid_cols = [f for f in self.bile_acid_features if f in self.reduced_feature_names]
    #         clinic_cols = [f for f in self.clinical_features if f in self.reduced_feature_names]
    #
    #         # 2. Lipid feature filtering
    #         valid_lipid_cols = [f for f in lipid_cols if f in self.reduced_feature_names]
    #         lipid_indices = [self.reduced_feature_names.index(f) for f in valid_lipid_cols]
    #         X_train_lipid = self.X_train_reduced[:, lipid_indices]
    #         lipid_names = valid_lipid_cols
    #         n_samples, n_lipid_features = X_train_lipid.shape
    #         logger.info(f"Initial lipid features: {n_lipid_features}, training samples: {n_samples}")
    #
    #         lipid_count_tracker = {
    #             'Initial': len(lipid_names),
    #             'Variance Filter': 0,
    #             'ANOVA Filter': 0,
    #             'Correlation Reduction': 0,
    #             'SHAP Filter': 0
    #         }
    #
    #         assert len(np.unique(self.y_train)) == self.n_classes, \
    #             f"Invalid training label count: {len(np.unique(self.y_train))} (expected 4)"
    #
    #         # Step 1: Variance filter (top 30%)
    #         lipid_var = np.var(X_train_lipid, axis=0)
    #         var_threshold = np.percentile(lipid_var, 70)
    #         high_var_idx = np.where(lipid_var >= var_threshold)[0]
    #         X_train_lipid = X_train_lipid[:, high_var_idx]
    #         lipid_names = [lipid_names[i] for i in high_var_idx]
    #         lipid_count_tracker['Variance Filter'] = len(lipid_names)
    #         logger.info(f"Variance filtered lipid features: {len(lipid_names)}")
    #
    #         # Step 2: ANOVA test (p<0.05)
    #         from sklearn.feature_selection import f_classif
    #         f_stat, p_val = f_classif(X_train_lipid, self.y_train)
    #         sig_idx = np.where(p_val < 0.05)[0]
    #         X_train_lipid = X_train_lipid[:, sig_idx]
    #         lipid_names = [lipid_names[i] for i in sig_idx]
    #         lipid_count_tracker['ANOVA Filter'] = len(lipid_names)
    #         logger.info(f"ANOVA filtered lipid features (p<0.05): {len(lipid_names)}")
    #
    #         # Step 3: Correlation reduction (r<0.8)
    #         if len(lipid_names) > 1:
    #             corr_matrix = pd.DataFrame(X_train_lipid, columns=lipid_names).corr()
    #             drop_cols = []
    #             for i in range(len(corr_matrix.columns)):
    #                 for j in range(i + 1, len(corr_matrix.columns)):
    #                     col_i = corr_matrix.columns[i]
    #                     col_j = corr_matrix.columns[j]
    #                     if abs(corr_matrix.loc[col_i, col_j]) > 0.8 and col_j not in drop_cols:
    #                         drop_cols.append(col_j)
    #             lipid_names = [f for f in lipid_names if f not in drop_cols]
    #             lipid_indices_filtered = [self.reduced_feature_names.index(f) for f in lipid_names]
    #             X_train_lipid = self.X_train_reduced[:, lipid_indices_filtered]
    #             lipid_count_tracker['Correlation Reduction'] = len(lipid_names)
    #             logger.info(f"Correlation reduced lipid features (r<0.8): {len(lipid_names)}")
    #         else:
    #             lipid_count_tracker['Correlation Reduction'] = len(lipid_names)
    #             logger.warning("Lipid feature count ≤1, skip correlation reduction")
    #
    #         # Step 4: SHAP filter (top 30)
    #         shap_importance = None
    #         if len(lipid_names) > 30:
    #             logger.info(f"SHAP feature selection for {len(lipid_names)} lipid features")
    #             lr = LogisticRegression(
    #                 random_state=42, max_iter=1000, multi_class='ovr',
    #                 class_weight='balanced', n_jobs=-1
    #             )
    #             lr.fit(X_train_lipid, self.y_train)
    #
    #             explainer = shap.LinearExplainer(lr, X_train_lipid)
    #             shap_values = explainer.shap_values(X_train_lipid)
    #
    #             if isinstance(shap_values, list):
    #                 logger.info(f"SHAP shape: list ({len(shap_values)} classes)")
    #                 assert len(shap_values) == self.n_classes, \
    #                     f"SHAP list length ({len(shap_values)}) != class count ({self.n_classes})"
    #                 shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    #             elif isinstance(shap_values, np.ndarray):
    #                 if shap_values.ndim == 3:
    #                     logger.info(f"SHAP shape: 3D array ({shap_values.shape})")
    #                     if shap_values.shape[0] == self.n_classes:
    #                         shap_values = shap_values.transpose(1, 2, 0)
    #                     elif shap_values.shape[2] != self.n_classes:
    #                         raise ValueError(f"SHAP 3D array class dimension mismatch (expected {self.n_classes})")
    #                     shap_importance = np.mean(np.abs(shap_values), axis=(0, 2))
    #                 elif shap_values.ndim == 2:
    #                     logger.warning("SHAP 2D array (uncommon for multi-class)")
    #                     shap_importance = np.mean(np.abs(shap_values), axis=0)
    #                 else:
    #                     raise ValueError(f"Unsupported SHAP dimension: {shap_values.ndim} (expected 2/3)")
    #             else:
    #                 raise TypeError(f"Unsupported SHAP type: {type(shap_values)} (expected list/ndarray)")
    #
    #             assert len(shap_importance) == len(lipid_names), \
    #                 f"SHAP importance length ({len(shap_importance)}) != lipid feature count ({len(lipid_names)})"
    #
    #             shap_series = pd.Series(shap_importance, index=lipid_names).sort_values(ascending=False)
    #             lipid_names = shap_series.head(30).index.tolist()
    #             lipid_indices_filtered = [self.reduced_feature_names.index(f) for f in lipid_names]
    #             X_train_lipid = self.X_train_reduced[:, lipid_indices_filtered]
    #             lipid_count_tracker['SHAP Filter'] = len(lipid_names)
    #             logger.info(f"SHAP filtered lipid features (top 30): {len(lipid_names)}")
    #         else:
    #             lipid_count_tracker['SHAP Filter'] = len(lipid_names)
    #             logger.warning("Lipid feature count ≤30, skip SHAP filter")
    #
    #         # 3. Bile acid feature filtering
    #         valid_bile_cols = [f for f in bile_acid_cols if f in self.reduced_feature_names]
    #         bile_indices = [self.reduced_feature_names.index(f) for f in valid_bile_cols]
    #         X_train_bile = self.X_train_reduced[:, bile_indices]
    #
    #         bile_count_tracker = {
    #             'Initial': len(valid_bile_cols),
    #             'Variance Filter': 0,
    #             'ANOVA Filter': 0,
    #             'Correlation Reduction': 0,
    #             'Top 20 Selection': 0
    #         }
    #
    #         # Variance filter (top 50%)
    #         bile_var = np.var(X_train_bile, axis=0)
    #         var_threshold = np.percentile(bile_var, 50)
    #         high_var_idx = np.where(bile_var >= var_threshold)[0]
    #         X_train_bile = X_train_bile[:, high_var_idx]
    #         bile_names = [valid_bile_cols[i] for i in high_var_idx]
    #         bile_count_tracker['Variance Filter'] = len(bile_names)
    #         logger.info(f"Variance filtered bile acid features: {len(bile_names)}")
    #
    #         # ANOVA test (p<0.1)
    #         f_stat, p_val = f_classif(X_train_bile, self.y_train)
    #         sig_idx = np.where(p_val < 0.1)[0]
    #         bile_names = [bile_names[i] for i in sig_idx]
    #         X_train_bile = X_train_bile[:, sig_idx]
    #         bile_count_tracker['ANOVA Filter'] = len(bile_names)
    #         logger.info(f"ANOVA filtered bile acid features (p<0.1): {len(bile_names)}")
    #
    #         # Correlation reduction (r<0.8)
    #         if len(bile_names) > 1:
    #             corr_matrix = pd.DataFrame(X_train_bile, columns=bile_names).corr()
    #             drop_cols = []
    #             for i in range(len(corr_matrix.columns)):
    #                 for j in range(i + 1, len(corr_matrix.columns)):
    #                     col_i, col_j = corr_matrix.columns[i], corr_matrix.columns[j]
    #                     if abs(corr_matrix.loc[col_i, col_j]) > 0.8 and col_j not in drop_cols:
    #                         drop_cols.append(col_j)
    #             bile_names = [f for f in bile_names if f not in drop_cols]
    #             bile_count_tracker['Correlation Reduction'] = len(bile_names)
    #             logger.info(f"Correlation reduced bile acid features (r<0.8): {len(bile_names)}")
    #         else:
    #             bile_count_tracker['Correlation Reduction'] = len(bile_names)
    #             logger.warning("Bile acid feature count ≤1, skip correlation reduction")
    #
    #         # Keep top 20 bile acid features
    #         if len(bile_names) > 20:
    #             bile_var_series = pd.Series(bile_var, index=valid_bile_cols).sort_values(ascending=False)
    #             bile_names = bile_var_series[bile_var_series.index.isin(bile_names)].head(20).index.tolist()
    #             bile_count_tracker['Top 20 Selection'] = len(bile_names)
    #             logger.info(f"Final bile acid features (top 20): {len(bile_names)}")
    #         elif len(bile_names) < 10:
    #             bile_count_tracker['Top 20 Selection'] = len(bile_names)
    #             logger.warning("Bile acid features <10, supplementing high variance features")
    #             bile_var_series = pd.Series(bile_var, index=valid_bile_cols).sort_values(ascending=False)
    #             supplement_bile = bile_var_series[~bile_var_series.index.isin(bile_names)].head(
    #                 10 - len(bile_names)).index.tolist()
    #             bile_names = bile_names + supplement_bile
    #             bile_count_tracker['Top 20 Selection'] = len(bile_names)
    #
    #         # 4. Combine features
    #         final_features = clinic_cols + lipid_names + bile_names
    #         self.reduced_feature_names = final_features
    #         final_indices = [self.feature_columns.index(f) for f in final_features]
    #         self.X_train_reduced = self.X_train_scaled[:, final_indices]
    #         self.X_val_reduced = self.X_val_scaled[:, final_indices]
    #
    #         # 绘图
    #         self._plot_feature_count_change(lipid_count_tracker, bile_count_tracker)
    #
    #         if shap_importance is not None:
    #             self._plot_top30_lipid_shap(shap_series)
    #
    #         self._plot_core_markers_collinearity_heatmap(final_features)
    #
    #         # 保存过滤后的数据
    #         if self.save_filtered_data:
    #             os.makedirs(self.save_dir, exist_ok=True)
    #
    #             try:
    #                 train_path = os.path.join(self.save_dir, f"X_train_reduced.{self.save_format}")
    #                 val_path = os.path.join(self.save_dir, f"X_val_reduced.{self.save_format}")
    #                 feat_name_path = os.path.join(self.save_dir, "reduced_feature_names.txt")
    #
    #                 with open(feat_name_path, "w", encoding="utf-8") as f:
    #                     f.write("\n".join(self.reduced_feature_names))
    #
    #                 if self.save_format == "npy":
    #                     np.save(train_path, self.X_train_reduced)
    #                     np.save(val_path, self.X_val_reduced)
    #                 elif self.save_format == "csv":
    #                     pd.DataFrame(self.X_train_reduced, columns=self.reduced_feature_names).to_csv(train_path,
    #                                                                                                   index=False)
    #                     pd.DataFrame(self.X_val_reduced, columns=self.reduced_feature_names).to_csv(val_path,
    #                                                                                                 index=False)
    #                 else:
    #                     raise ValueError(f"Unsupported save format: {self.save_format} (only npy/csv)")
    #
    #                 logger.info(f"Filtered data saved successfully:")
    #                 logger.info(f"  - Training data: {train_path}")
    #                 logger.info(f"  - Validation data: {val_path}")
    #                 logger.info(f"  - Feature names: {feat_name_path}")
    #             except Exception as e:
    #                 logger.error(f"Failed to save filtered data: {str(e)}", exc_info=True)
    #
    #         logger.info(
    #             f"Filtered features: lipid {len(lipid_names)} + bile acid {len(bile_names)} + clinical {len(clinic_cols)}")
    #     return self
    def _kruskal_wallis_multifeature(self, X, y):
        """
        辅助函数：批量计算多特征的Kruskal-Wallis检验p值
        输入：X (n_samples, n_features)，y (n_samples,) 分组标签
        输出：每个特征的p值数组 (n_features,)
        """
        from scipy.stats import kruskal
        n_features = X.shape[1]
        p_vals = np.zeros(n_features)

        # 遍历每个特征进行Kruskal-Wallis检验
        for i in range(n_features):
            feature_vals = X[:, i]
            # 按分组拆分数据
            groups_data = []
            for group_label in np.unique(y):
                group_feature_vals = feature_vals[y == group_label]
                groups_data.append(group_feature_vals)

            # 执行检验，仅保留p值
            try:
                _, p_val = kruskal(*groups_data)
                p_vals[i] = p_val
            except Exception as e:
                logger.warning(f"Kruskal-Wallis failed for feature {i}: {str(e)}, setting p=1.0")
                p_vals[i] = 1.0

        return p_vals

    def filter_lipid_features(self):
        """Lipid + bile acid feature filtering (balanced selection) - 修改后版本"""
        logger.info("\nPre-filtering lipid + bile acid features (balanced selection)")
        if self.X_train_reduced is None:
            self.X_train_reduced = self.X_train_scaled.copy()
        if self.X_val_reduced is None:
            self.X_val_reduced = self.X_val_scaled.copy()

        if self.args.is_filter:
            self.reduced_feature_names = self.feature_columns.copy()

            # 1. Split lipid/bile acid/clinical features
            lipid_cols = [f for f in self.lipid_features if f in self.reduced_feature_names]
            bile_acid_cols = [f for f in self.bile_acid_features if f in self.reduced_feature_names]
            clinic_cols = [f for f in self.clinical_features if f in self.reduced_feature_names]

            # 2. Lipid feature filtering
            valid_lipid_cols = [f for f in lipid_cols if f in self.reduced_feature_names]
            lipid_indices = [self.reduced_feature_names.index(f) for f in valid_lipid_cols]
            X_train_lipid = self.X_train_reduced[:, lipid_indices]
            lipid_names = valid_lipid_cols.copy()
            n_lipid_features = X_train_lipid.shape[1]
            n_samples = X_train_lipid.shape[0]
            logger.info(f"Initial lipid features: {n_lipid_features}, training samples: {n_samples}")

            lipid_count_tracker = {
                'Initial': len(lipid_names),
                'Variance Filter': 0,
                'ANOVA Filter': 0,  # 保留原键名，避免影响后续可视化
                'Correlation Reduction': 0,
                'SHAP Filter': 0
            }

            assert len(np.unique(self.y_train)) == self.n_classes, \
                f"Invalid training label count: {len(np.unique(self.y_train))} (expected {self.n_classes})"

            # Step 1: Variance filter (top 30%) - 保持不变
            if len(lipid_names) > 0:
                lipid_var = np.var(X_train_lipid, axis=0)
                var_threshold = np.percentile(lipid_var, 70)
                high_var_idx = np.where(lipid_var >= var_threshold)[0]
                X_train_lipid = X_train_lipid[:, high_var_idx]
                lipid_names = [lipid_names[i] for i in high_var_idx]
                lipid_count_tracker['Variance Filter'] = len(lipid_names)
                logger.info(f"Variance filtered lipid features: {len(lipid_names)}")
            else:
                lipid_count_tracker['Variance Filter'] = 0
                logger.warning("No lipid features available (feature_type_filter may exclude lipids), skip lipid filtering")

            # Step 2: Kruskal-Wallis test (replace ANOVA, p<0.05) - 核心修改1
            if len(lipid_names) > 0:
                # 调用辅助函数计算Kruskal-Wallis p值
                p_val = self._kruskal_wallis_multifeature(X_train_lipid, self.y_train)
                sig_idx = np.where(p_val < 0.05)[0]

                # 更新特征数据和名称
                X_train_lipid = X_train_lipid[:, sig_idx] if len(sig_idx) > 0 else np.array([]).reshape(n_samples, 0)
                lipid_names = [lipid_names[i] for i in sig_idx] if len(sig_idx) > 0 else []
                lipid_count_tracker['ANOVA Filter'] = len(lipid_names)
                logger.info(f"Kruskal-Wallis filtered lipid features (p<0.05): {len(lipid_names)}")
            else:
                lipid_count_tracker['ANOVA Filter'] = 0
                logger.warning("No lipid features left after variance filter, skip Kruskal-Wallis")

            # Step 3: Correlation reduction (r<0.8) - 修复语法错误
            if len(lipid_names) > 1:
                corr_matrix = pd.DataFrame(X_train_lipid, columns=lipid_names).corr()
                drop_cols = set()  # 改用set避免重复删除
                # 修复j的循环范围：i+1 避免对角线和重复比较
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        col_i = corr_matrix.columns[i]
                        col_j = corr_matrix.columns[j]
                        if abs(corr_matrix.loc[col_i, col_j]) > 0.8:
                            drop_cols.add(col_j)  # 保留前一个特征，删除后一个特征

                # 更新脂质名称列表
                lipid_names = [f for f in lipid_names if f not in drop_cols]
                # 重新获取过滤后的特征索引（基于原始reduced_feature_names）
                lipid_indices_filtered = [self.reduced_feature_names.index(f) for f in lipid_names]
                X_train_lipid = self.X_train_reduced[:, lipid_indices_filtered] if len(
                    lipid_indices_filtered) > 0 else np.array([]).reshape(n_samples, 0)
                lipid_count_tracker['Correlation Reduction'] = len(lipid_names)
                logger.info(f"Correlation reduced lipid features (r<0.8): {len(lipid_names)}")
            else:
                lipid_count_tracker['Correlation Reduction'] = len(lipid_names)
                logger.warning("Lipid feature count ≤1, skip correlation reduction")

            # Step 4: SHAP filter (top 30) - 保持不变
            shap_importance = None
            shap_series = None
            if len(lipid_names) > 30:
                logger.info(f"SHAP feature selection for {len(lipid_names)} lipid features")
                lr = LogisticRegression(
                    random_state=42, max_iter=1000, multi_class='ovr',
                    class_weight='balanced', n_jobs=-1
                )
                # 确保数据非空再拟合
                if X_train_lipid.shape[1] > 0 and len(lipid_names) > 0:
                    lr.fit(X_train_lipid, self.y_train)

                    explainer = shap.LinearExplainer(lr, X_train_lipid)
                    shap_values = explainer.shap_values(X_train_lipid)

                    if isinstance(shap_values, list):
                        logger.info(f"SHAP shape: list ({len(shap_values)} classes)")
                        assert len(shap_values) == self.n_classes, \
                            f"SHAP list length ({len(shap_values)}) != class count ({self.n_classes})"
                        shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                    elif isinstance(shap_values, np.ndarray):
                        if shap_values.ndim == 3:
                            logger.info(f"SHAP shape: 3D array ({shap_values.shape})")
                            if shap_values.shape[0] == self.n_classes:
                                shap_values = shap_values.transpose(1, 2, 0)
                            elif shap_values.shape[2] != self.n_classes:
                                raise ValueError(f"SHAP 3D array class dimension mismatch (expected {self.n_classes})")
                            shap_importance = np.mean(np.abs(shap_values), axis=(0, 2))
                        elif shap_values.ndim == 2:
                            logger.warning("SHAP 2D array (uncommon for multi-class)")
                            shap_importance = np.mean(np.abs(shap_values), axis=0)
                        else:
                            raise ValueError(f"Unsupported SHAP dimension: {shap_values.ndim} (expected 2/3)")
                    else:
                        raise TypeError(f"Unsupported SHAP type: {type(shap_values)} (expected list/ndarray)")

                    assert len(shap_importance) == len(lipid_names), \
                        f"SHAP importance length ({len(shap_importance)}) != lipid feature count ({len(lipid_names)})"

                    shap_series = pd.Series(shap_importance, index=lipid_names).sort_values(ascending=False)
                    lipid_names = shap_series.head(30).index.tolist()
                    lipid_indices_filtered = [self.reduced_feature_names.index(f) for f in lipid_names]
                    X_train_lipid = self.X_train_reduced[:, lipid_indices_filtered] if len(
                        lipid_indices_filtered) > 0 else np.array([]).reshape(n_samples, 0)
                    lipid_count_tracker['SHAP Filter'] = len(lipid_names)
                    logger.info(f"SHAP filtered lipid features (top 30): {len(lipid_names)}")
                else:
                    logger.warning("No valid lipid features for SHAP selection")
                    lipid_count_tracker['SHAP Filter'] = 0
            else:
                lipid_count_tracker['SHAP Filter'] = len(lipid_names)
                logger.warning("Lipid feature count ≤30, skip SHAP filter")

            # 3. Bile acid feature filtering - 补充缺失变量，修改ANOVA为Kruskal-Wallis
            valid_bile_cols = [f for f in bile_acid_cols if f in self.reduced_feature_names]  # 补充缺失变量
            bile_indices = [self.reduced_feature_names.index(f) for f in valid_bile_cols]
            X_train_bile = self.X_train_reduced[:, bile_indices] if len(bile_indices) > 0 else np.array([]).reshape(
                n_samples, 0)
            bile_names = valid_bile_cols.copy()

            bile_count_tracker = {
                'Initial': len(valid_bile_cols),
                'Variance Filter': 0,
                'ANOVA Filter': 0,  # 保留原键名
                'Correlation Reduction': 0,
                'Top 20 Selection': 0
            }

            # Variance filter (top 50%) - 保持不变
            if len(bile_names) > 0:
                bile_var = np.var(X_train_bile, axis=0)
                var_threshold = np.percentile(bile_var, 50)
                high_var_idx = np.where(bile_var >= var_threshold)[0]
                X_train_bile = X_train_bile[:, high_var_idx] if len(high_var_idx) > 0 else np.array([]).reshape(
                    n_samples, 0)
                bile_names = [bile_names[i] for i in high_var_idx] if len(high_var_idx) > 0 else []
                bile_count_tracker['Variance Filter'] = len(bile_names)
                logger.info(f"Variance filtered bile acid features: {len(bile_names)}")
            else:
                bile_count_tracker['Variance Filter'] = 0
                logger.warning("No bile acid features left, skip subsequent bile filters")

            # Kruskal-Wallis test (replace ANOVA, p<0.1) - 核心修改2
            if len(bile_names) > 0:
                # 调用辅助函数计算Kruskal-Wallis p值
                p_val = self._kruskal_wallis_multifeature(X_train_bile, self.y_train)
                sig_idx = np.where(p_val < 0.1)[0]

                # 更新特征数据和名称
                X_train_bile = X_train_bile[:, sig_idx] if len(sig_idx) > 0 else np.array([]).reshape(n_samples, 0)
                bile_names = [bile_names[i] for i in sig_idx] if len(sig_idx) > 0 else []
                bile_count_tracker['ANOVA Filter'] = len(bile_names)
                logger.info(f"Kruskal-Wallis filtered bile acid features (p<0.1): {len(bile_names)}")
            else:
                bile_count_tracker['ANOVA Filter'] = 0
                logger.warning("No bile acid features left after variance filter, skip Kruskal-Wallis")

            # Correlation reduction (r<0.8) - 修复语法错误
            if len(bile_names) > 1:
                corr_matrix = pd.DataFrame(X_train_bile, columns=bile_names).corr()
                drop_cols = set()
                # 修复j的循环范围：i+1 避免重复比较
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        col_i, col_j = corr_matrix.columns[i], corr_matrix.columns[j]
                        if abs(corr_matrix.loc[col_i, col_j]) > 0.8:
                            drop_cols.add(col_j)

                bile_names = [f for f in bile_names if f not in drop_cols]
                bile_count_tracker['Correlation Reduction'] = len(bile_names)
                logger.info(f"Correlation reduced bile acid features (r<0.8): {len(bile_names)}")
            else:
                bile_count_tracker['Correlation Reduction'] = len(bile_names)
                logger.warning("Bile acid feature count ≤1, skip correlation reduction")

            # Keep top 20 bile acid features - 保持不变
            if len(bile_names) > 20:
                # 重新计算筛选后的胆汁酸方差（避免使用原始方差）
                # current_bile_var = np.var(pd.DataFrame(X_train_bile, columns=bile_names), axis=0)
                bile_var_series = pd.Series(bile_var, index=valid_bile_cols).sort_values(ascending=False)
                bile_names = bile_var_series[bile_var_series.index.isin(bile_names)].head(20).index.tolist()
                # bile_names = bile_var_series.head(20).index.tolist()
                bile_count_tracker['Top 20 Selection'] = len(bile_names)
                logger.info(f"Final bile acid features (top 20): {len(bile_names)}")
            elif len(bile_names) < 10 and len(valid_bile_cols) > len(bile_names):
                bile_count_tracker['Top 20 Selection'] = len(bile_names)
                logger.warning("Bile acid features <10, supplementing high variance features")
                # 补充高方差但未被选中的胆汁酸特征
                all_bile_var = np.var(self.X_train_reduced[:, bile_indices], axis=0) if len(
                    bile_indices) > 0 else np.array([])
                bile_var_series = pd.Series(all_bile_var, index=valid_bile_cols).sort_values(ascending=False)
                supplement_bile = bile_var_series[~bile_var_series.index.isin(bile_names)].head(
                    10 - len(bile_names)).index.tolist()
                bile_names = bile_names + supplement_bile
                bile_count_tracker['Top 20 Selection'] = len(bile_names)
            else:
                bile_count_tracker['Top 20 Selection'] = len(bile_names)

            # 4. Combine features - 保持不变
            final_features = clinic_cols + lipid_names + bile_names
            # 去重（避免特征重复）
            final_features = list(dict.fromkeys(final_features))
            self.reduced_feature_names = final_features

            # 重新获取最终特征索引（基于原始feature_columns）
            final_indices = [self.feature_columns.index(f) for f in final_features if f in self.feature_columns]
            self.X_train_reduced = self.X_train_scaled[:, final_indices] if len(final_indices) > 0 else np.array(
                []).reshape(self.X_train_scaled.shape[0], 0)
            self.X_val_reduced = self.X_val_scaled[:, final_indices] if len(final_indices) > 0 else np.array(
                []).reshape(self.X_val_scaled.shape[0], 0)

            # 绘图 - 保持不变
            self._plot_feature_count_change(lipid_count_tracker, bile_count_tracker)

            if shap_importance is not None and shap_series is not None:
                self._plot_top30_lipid_shap(shap_series)

            self._plot_core_markers_collinearity_heatmap(final_features)

            # 保存过滤后的数据 - 保持不变
            if self.save_filtered_data and len(final_features) > 0:
                os.makedirs(self.save_dir, exist_ok=True)

                try:
                    train_path = os.path.join(self.save_dir, f"X_train_reduced.{self.save_format}")
                    val_path = os.path.join(self.save_dir, f"X_val_reduced.{self.save_format}")
                    feat_name_path = os.path.join(self.save_dir, "reduced_feature_names.txt")

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
            elif self.save_filtered_data and len(final_features) == 0:
                logger.error("No final features left, skip saving filtered data")

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

        # 如果脂质和胆汁酸特征都为0（如 clinical_only 模式），跳过绘图
        if max(lipid_vals + bile_vals) == 0:
            logger.warning("All lipid/bile acid feature counts are 0, skip feature count change plot")
            return

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

        y_max = max(lipid_vals + bile_vals)
        y_offset = max(1, y_max * 0.03)  # 相对偏移，避免 y 轴极值撑大画布
        for i, (l_val, b_val) in enumerate(zip(lipid_vals, bile_vals)):
            ax.text(i, l_val + y_offset, str(l_val), ha='center', va='bottom', fontsize=7, color=COLOR_PALETTE['lipid'])
            ax.text(i, max(0, b_val - y_offset), str(b_val), ha='center', va='top', fontsize=7, color=COLOR_PALETTE['bile_acid'])

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root,self.figure_dir,f'feature_count_change_{self.feature_type_filter}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir, f'feature_count_change_{self.feature_type_filter}.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
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
        plt.savefig(os.path.join(self.save_root,self.figure_dir,
            f'ftop30_lipid_shap_importance_{self.feature_type_filter}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'ftop30_lipid_shap_importance_{self.feature_type_filter}.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
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
        # out_path = os.path.join(self.save_root,self.figure_dir,f'core_markers_collinearity_heatmap_{self.feature_type_filter}.png')
        plt.savefig(os.path.join(self.save_root,self.figure_dir,f'core_markers_collinearity_heatmap_{self.feature_type_filter}.png'), dpi=600, bbox_inches='tight')
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'core_markers_collinearity_heatmap_{self.feature_type_filter}.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
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
                'Explained_Variance_Ratio': self.dimension_reducer.explained_variance_ratio_,
                'Cumulative_Variance_Ratio': np.cumsum(self.dimension_reducer.explained_variance_ratio_)
            })
            pca_results.to_csv(os.path.join(self.save_root, self.results_dir,f'pca_dimension_reduction_results.csv'), index=False)

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
            selectkbest_scores.to_csv(os.path.join(self.save_root, self.results_dir,f'selectkbest_dimension_reduction_results.csv'), index=False)

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
        """SHAP feature selection (multi-class support)"""
        n_samples, n_reduced_features = self.X_train_reduced.shape
        base_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        base_model.fit(self.X_train_reduced, self.y_train)
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(self.X_train_reduced, check_additivity=False)

        if isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                if shap_values.shape[0] == self.n_classes:
                    shap_values = [shap_values[i] for i in range(self.n_classes)]
                elif shap_values.shape[2] == self.n_classes:
                    shap_values = np.transpose(shap_values, (2, 0, 1))
                    shap_values = [shap_values[i] for i in range(self.n_classes)]
                else:
                    raise ValueError(f"Unsupported SHAP shape: {shap_values.shape}")
            else:
                raise ValueError(f"SHAP array dimension: {shap_values.ndim} (expected 3)")

        for i in range(self.n_classes):
            assert shap_values[i].shape == (n_samples, n_reduced_features), \
                f"SHAP shape mismatch for class {self.class_name_mapping[i]}: {shap_values[i].shape}"

        shap_importance = [
            np.mean([np.abs(shap_values[class_idx][:, i]).mean() for class_idx in range(self.n_classes)])
            for i in range(n_reduced_features)
        ]
        shap_series = pd.Series(shap_importance, index=self.reduced_feature_names)
        self.top_features = shap_series.sort_values(ascending=False).head(self.n_selected).index.tolist()

    def _select_features_with_lasso(self):
        """Lasso feature selection (GPU precision fix)"""
        n_samples, n_reduced_features = self.X_train_reduced.shape
        lasso_importance = np.zeros(n_reduced_features)

        logger.info(f"Running LassoCV (OvR) with GPU precision fix...")
        for class_idx in range(self.n_classes):
            y_binary = (self.y_train == class_idx).astype(int)
            X_train_float64 = self.X_train_reduced.astype(np.float64)
            lasso = LassoCV(
                cv=5, random_state=42, max_iter=10000,
                n_jobs=1, precompute=False, tol=1e-4
            )
            lasso.fit(X_train_float64, y_binary)
            class_importance = np.abs(lasso.coef_)
            lasso_importance += class_importance
            logger.info(
                f"Class {self.class_name_mapping[class_idx]}: Best alpha = {lasso.alpha_:.6f}, non-zero features = {np.sum(class_importance > 0)}")

        lasso_importance_avg = lasso_importance / self.n_classes
        lasso_series = pd.Series(lasso_importance_avg, index=self.reduced_feature_names)

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
            'Lasso_Importance': lasso_importance_avg,
            'Selected': [f in self.top_features for f in self.reduced_feature_names]
        }).sort_values('Lasso_Importance', ascending=False)
        lasso_results.to_csv(os.path.join(self.save_root, self.results_dir,f'lasso_feature_importance_{self.dimension_reduction_method}.csv'), index=False)

    def _select_features_with_brute_force(self):
        """Brute force feature selection (batch processing for memory efficiency)"""
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

        logger.info(f"Brute force screening (batch processing)...")
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
                    y_val_proba = self.brute_force_base_model.predict_proba(X_val_comb)
                    val_auc = roc_auc_macro(self.y_val, y_val_proba)

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
                y_val_proba = self.brute_force_base_model.predict_proba(X_val_comb)
                val_auc = roc_auc_score(self.y_val, y_val_proba, multi_class='ovr')

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
            f'brute_force_results_{self.dimension_reduction_method}_top{self.n_selected}.csv'),
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
            f'brute_force_feature_frequency_{self.dimension_reduction_method}_top{self.n_selected}.csv'),
            index=False)

        self.top_features = best_combination
        logger.info(f"Best combination (AUC: {best_auc:.4f}): {self.top_features}")
        logger.info(f"Top 5 combinations by AUC:")
        for i in range(min(5, len(self.brute_force_results))):
            row = self.brute_force_results.iloc[i]
            logger.info(f"  Rank {i+1}: AUC={row['val_auc']:.4f}, features={row['feature_names'][:50]}...")

    def _select_features_with_sis(self):
        """SIS feature selection (weighted score for multi-class)"""
        n_samples, n_reduced_features = self.X_train_reduced.shape
        logger.info(f"SIS feature selection: {n_reduced_features} reduced features")
        logger.info(f"SIS config: score func={self.sis_score_func.__name__}, k method={self.sis_k_method}")

        raw_scores, p_values = self.sis_score_func(self.X_train_reduced, self.y_train)
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)

        weighted_scores = np.zeros(n_reduced_features)
        for cls_idx, weight in enumerate(class_weights):
            y_binary = (self.y_train == cls_idx).astype(int)
            cls_score, _ = self.sis_score_func(self.X_train_reduced, y_binary)
            weighted_scores += cls_score * weight
        weighted_scores /= sum(class_weights)

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
            f'sis_feature_selection_results_{self.dimension_reduction_method}_top{self.n_selected}.csv'),
            index=False, encoding='utf-8-sig'
        )
        logger.info(f"SIS results saved to results_3/sis_feature_selection_results_*.csv")

    def _perform_kruskal_wallis_test(self, feature_data, class_labels, class_names):
        """Kruskal-Wallis H test (non-parametric, multi-class)"""
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

    def define_models(self):
        """Model definition (GPU optimized)"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        models_config = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [81, 100, 150, 200],
                    'max_depth': [5, 6, 7],
                    'min_samples_split': [4, 5, 8],
                    'min_samples_leaf': [1, 2, 3],
                    'class_weight': ['balanced', {0: 4.0, 1: 1.0, 2: 3.0, 3: 1.0}]
                },
                'cv': cv
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=2000, n_jobs=-1),
                'param_grid': {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 20.0, 30.0],
                    'penalty': ['l2'],
                    'class_weight': ['balanced', None],
                    'l1_ratio': [None]},
                'cv': cv
            },
            'SupportVectorMachine': {
                'model': SVC(probability=True, random_state=42),
                'param_grid': {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'class_weight': ['balanced', None]},
                'cv': cv
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=42, eval_metric='mlogloss'),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5],
                    'learning_rate': [0.005, 0.01, 0.1]
                },
                'cv': cv
            }
        }

        if self.use_gpu:
            logger.info("Optimizing XGBoost for GPU")
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
        """Grid search with cross validation (GPU optimized)"""
        logger.info("\nStep 5/7: Grid search with 5-fold CV...")
        models = self.define_models()
        top_indices = [self.reduced_feature_names.index(f) for f in self.top_features]
        X_train_top = self.X_train_reduced[:, top_indices]
        assert X_train_top.shape[1] == self.n_selected, \
            f"Train top features shape error: {X_train_top.shape[1]} (expected {self.n_selected})"

        n_jobs = 1 if self.use_gpu else -1
        logger.info(f"Grid search n_jobs: {n_jobs} (GPU: {self.use_gpu})")

        for name, config in models.items():
            logger.info(f"\nTuning {name}...")

            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['param_grid'],
                cv=config['cv'],
                scoring='roc_auc_ovr',
                n_jobs=n_jobs,
                verbose=0,
                return_train_score=False
            )

            if self.use_gpu and name == 'XGBoost':
                grid_search.estimator.set_params(batch_size=self.gpu_batch_size)

            grid_search.fit(X_train_top, self.y_train)
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

    def evaluate_models(self):
        """Evaluate models on validation set (AUC, CI, Delong test, clinical metrics)"""
        logger.info("\nStep 6/7: Evaluating models...")
        target_names = self.class_names
        top_indices = [self.reduced_feature_names.index(f) for f in self.top_features]
        X_val_top = self.X_val_reduced[:, top_indices]
        assert X_val_top.shape[1] == self.n_selected, \
            f"Validation top features shape error: {X_val_top.shape[1]} (expected {self.n_selected})"
        X_train_top = self.X_train_reduced[:, top_indices]

        val_aucs = {}
        val_probas = {}

        for name, (model, best_params) in self.models.items():
            if self.use_gpu and name == 'XGBoost':
                y_val_proba = model.predict_proba(X_val_top)
                y_train_proba = model.predict_proba(X_train_top)
            else:
                y_val_proba = model.predict_proba(X_val_top)
                y_train_proba = model.predict_proba(X_train_top)

            y_val_pred = model.predict(X_val_top)
            y_train_pred = model.predict(X_train_top)

            val_auc = roc_auc_macro(self.y_val, y_val_proba)
            train_auc = roc_auc_macro(self.y_train, y_train_proba)

            val_ci_lower, val_ci_upper = calculate_auc_ci(self.y_val, y_val_proba, n_bootstrap=1000)
            train_ci_lower, train_ci_upper = calculate_auc_ci(self.y_train, y_train_proba, n_bootstrap=1000)

            self.model_results[name] = {
                'train_auc': train_auc, 'val_auc': val_auc,
                'train_ci_lower': train_ci_lower, 'train_ci_upper': train_ci_upper,
                'val_ci_lower': val_ci_lower, 'val_ci_upper': val_ci_upper,
                'y_train_proba': y_train_proba, 'y_val_proba': y_val_proba,
                'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred,
                'cv_mean_auc': self.cv_results[name]['cv_mean'],
                'cv_std_auc': self.cv_results[name]['cv_std'],
                'cv_ci_lower': self.cv_results[name]['cv_ci_lower'],
                'cv_ci_upper': self.cv_results[name]['cv_ci_upper'],
                'cv_fold_auc': self.cv_results[name]['cv_fold_auc'],
                'best_params': best_params,
                'top_features': self.top_features,
                'feature_type_filter': self.feature_type_filter
            }

            val_aucs[name] = val_auc
            val_probas[name] = y_val_proba

            logger.info(f"\n{name} Validation Report (Feature Type: {self.feature_type_filter}):")
            logger.info(f"Train AUC: {train_auc:.4f} (95% CI: {train_ci_lower:.4f}-{train_ci_upper:.4f})")
            logger.info(f"Validation AUC: {val_auc:.4f} (95% CI: {val_ci_lower:.4f}-{val_ci_upper:.4f})")
            logger.info(
                f"5-fold CV Mean AUC: {self.cv_results[name]['cv_mean']:.4f} (95% CI: {self.cv_results[name]['cv_ci_lower']:.4f}-{self.cv_results[name]['cv_ci_upper']:.4f})")
            class_report = classification_report(self.y_val, y_val_pred, target_names=target_names)
            for line in class_report.split('\n'):
                if line.strip():
                    logger.info(line.strip())

        logger.info("\nRunning Delong test for pairwise AUC comparison...")
        model_names = list(val_aucs.keys())
        self.delong_results = {}
        self.delong_target_class = 0
        y_true_binary = (self.y_val == self.delong_target_class).astype(int)
        target_class_name = self.class_name_mapping[self.delong_target_class]
        logger.info(
            f"Delong test based on class: {target_class_name} (binary: {self.delong_target_class} vs others)")

        for (model1, model2) in combinations(model_names, 2):
            y_score1 = val_probas[model1][:, self.delong_target_class]
            y_score2 = val_probas[model2][:, self.delong_target_class]

            assert y_score1.ndim == 1 and y_score2.ndim == 1, "Prediction probabilities must be 1D arrays"
            assert len(y_score1) == len(y_true_binary), "Probability and label lengths must match"

            try:
                p_value, auc1, auc2 = self.delong_test.compare(y_true_binary, y_score1, y_score2)
                self.delong_results[(model1, model2)] = {
                    'p_value': p_value,
                    'auc1': auc1,
                    'auc2': auc2,
                    'significant': p_value < 0.05
                }
                logger.info(f"{model1} vs {model2}: p-value = {p_value:.4f} (AUC1: {auc1:.4f}, AUC2: {auc2:.4f})")
            except Exception as e:
                logger.warning(f"Delong test failed for {model1} vs {model2}: {str(e)}")
                self.delong_results[(model1, model2)] = {
                    'p_value': np.nan,
                    'auc1': np.nan,
                    'auc2': np.nan,
                    'significant': False
                }

        results_df = pd.DataFrame({
            'Model': self.model_results.keys(),
            'Dimension_Reduction_Method': self.dimension_reduction_method.upper(),
            'Feature_Selection_Method': self.feature_selection_method.upper(),
            'Feature_Type_Filter': self.feature_type_filter,
            'Selected_Features_Count': self.n_selected,
            'Train_AUC': [v['train_auc'] for v in self.model_results.values()],
            'Train_CI_Lower': [v['train_ci_lower'] for v in self.model_results.values()],
            'Train_CI_Upper': [v['train_ci_upper'] for v in self.model_results.values()],
            'Validation_AUC': [v['val_auc'] for v in self.model_results.values()],
            'Validation_CI_Lower': [v['val_ci_lower'] for v in self.model_results.values()],
            'Validation_CI_Upper': [v['val_ci_upper'] for v in self.model_results.values()],
            'CV_Mean_AUC': [v['cv_mean_auc'] for v in self.model_results.values()],
            'CV_CI_Lower': [v['cv_ci_lower'] for v in self.model_results.values()],
            'CV_CI_Upper': [v['cv_ci_upper'] for v in self.model_results.values()],
            'CV_Std_AUC': [v['cv_std_auc'] for v in self.model_results.values()],
            'Training_Device': ['GPU' if self.use_gpu and name == 'XGBoost' else 'CPU' for name in
                                self.model_results.keys()]
        })
        results_df.to_csv(os.path.join(self.save_root, self.results_dir,
            f'model_comparison_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.csv'),
            index=False)
        logger.info(
            f"Model comparison results saved (feature type: {self.feature_type_filter})")

        delong_df = pd.DataFrame([
            {'Model1': m1, 'Model2': m2, 'Dimension_Reduction_Method': self.dimension_reduction_method.upper(),
             'Feature_Selection_Method': self.feature_selection_method.upper(),
             'Feature_Type_Filter': self.feature_type_filter,
             'Target_Class': target_class_name, 'P_Value': res['p_value'],
             'AUC1': res['auc1'], 'AUC2': res['auc2'], 'Significant': res['significant']}
            for (m1, m2), res in self.delong_results.items()
        ])
        delong_df.to_csv(os.path.join(self.save_root, self.results_dir,
            f'delong_test_results_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.csv'),
            index=False)

        selected_features_df = pd.DataFrame({
            'Feature_Rank': range(1, len(self.top_features) + 1),
            'Feature_Name': self.top_features,
            'Dimension_Reduction_Method': self.dimension_reduction_method.upper(),
            'Feature_Selection_Method': self.feature_selection_method.upper(),
            'Feature_Type_Filter': self.feature_type_filter,
            'Selected_Features_Count': self.n_selected
        })
        selected_features_df.to_csv(os.path.join(self.save_root, self.results_dir,
            f'selected_features_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.csv'),
            index=False)

        self._select_best_model()
        self._save_best_model()

        logger.info("\nModel evaluation completed (all results include feature type filter)")
        return self

    def _select_best_model(self):
        """Select best model based on validation AUC"""
        logger.info("\nSelecting best model based on validation AUC...")
        model_auc_pairs = [(name, metrics['val_auc']) for name, metrics in self.model_results.items()]
        model_auc_pairs.sort(key=lambda x: x[1], reverse=True)

        self.best_model_name = model_auc_pairs[0][0]
        self.best_model = self.models[self.best_model_name][0]
        self.best_model_params = self.models[self.best_model_name][1]
        self.best_model_metrics = self.model_results[self.best_model_name]

        logger.info(f"Best model selected: {self.best_model_name}")
        logger.info(f"Best model validation AUC: {self.best_model_metrics['val_auc']:.4f}")
        logger.info(f"Best model parameters: {self.best_model_params}")
        logger.info(f"Feature type filter used: {self.feature_type_filter}")

    def _save_best_model(self):
        """Save best model, preprocessors, and metadata"""
        logger.info("\nSaving best model and components...")
        save_dir = os.path.join(os.path.join(self.save_root, self.model_dir,
                                f"{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_top{self.n_selected}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu"))
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, 'best_model.joblib')
        joblib.dump(self.best_model, model_path)

        preprocessor_path = os.path.join(save_dir, 'preprocessors_and_reducer.joblib')
        joblib.dump({
            'imputer': self.imputer,
            'scaler': self.scaler,
            'dimension_reducer': self.dimension_reducer,
            'reduced_feature_names': self.reduced_feature_names,
            'sex_mode': getattr(self, 'sex_mode', None),
            'basic_median': getattr(self, 'basic_median', None),
            'liver_group_median': getattr(self, 'liver_group_median', None),
            'other_median': getattr(self, 'other_median', None),
            'non_threshold_mean': getattr(self, 'non_threshold_mean', None),
            'non_threshold_std': getattr(self, 'non_threshold_std', None)
        }, preprocessor_path)

        metadata = {
            'best_model_name': self.best_model_name,
            'dimension_reduction_method': self.dimension_reduction_method,
            'feature_selection_method': self.feature_selection_method,
            'feature_type_filter': self.feature_type_filter,
            'selected_features': self.top_features,
            'reduced_feature_names': self.reduced_feature_names,
            'all_raw_feature_columns': self.feature_columns,
            'class_name_mapping': self.class_name_mapping,
            'class_names': self.class_names,
            'n_classes': self.n_classes,
            'label_offset': self.label_offset,
            'best_model_params': self.best_model_params,
            'performance_metrics': {
                'train_auc': self.best_model_metrics['train_auc'],
                'val_auc': self.best_model_metrics['val_auc'],
                'val_auc_ci': [self.best_model_metrics['val_ci_lower'], self.best_model_metrics['val_ci_upper']],
                'cv_mean_auc': self.best_model_metrics['cv_mean_auc'],
                'cv_auc_ci': [self.best_model_metrics['cv_ci_lower'], self.best_model_metrics['cv_ci_upper']]
            },
            'clinical_preprocessing_config': {
                'antibody_imputation': 'fillna(0) (not tested = negative)',
                'liver_function_imputation': 'disease group median',
                'basic_features_imputation': 'Sex=mode, Age/BMI=global median',
                'standardization_method': 'clinical threshold-based (upper/lower limit) + Z-score for metabolites'
            },
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pca_variance_ratio': self.pca_variance_ratio if self.dimension_reduction_method == 'pca' else None,
            'selectkbest_k': self.selectkbest_k if self.dimension_reduction_method == 'selectkbest' else None,
            'brute_force_params': {
                'max_combinations': self.brute_force_max_combinations,
                'base_model': str(self.brute_force_base_model),
                'best_combination_auc': self.brute_force_results.iloc[0][
                    'val_auc'] if self.brute_force_results is not None else None
            },
            'lime_config': {
                'n_samples': self.lime_n_samples,
                'n_features': self.lime_n_features,
                'random_state': self.lime_random_state
            },
            'gpu_info': {
                'use_gpu': self.use_gpu,
                'gpu_available': check_gpu_availability(self.args.use_gpu),
                'gpu_batch_size': self.gpu_batch_size,
                'trained_on_gpu': self.use_gpu and self.best_model_name == 'XGBoost'
            },
            'command_line_args': vars(self.args)
        }

        metadata_path = os.path.join(save_dir, 'model_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        report_path = os.path.join(save_dir, 'best_model_performance.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"Best Model Report (Clinical Logic + Feature Type Filter + LIME)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Training Date: {metadata['training_date']}\n")
            f.write(f"Feature Type Filter: {self.feature_type_filter}\n")
            f.write(f"Clinical Preprocessing: {metadata['clinical_preprocessing_config']}\n")
            f.write(f"LIME Config: {metadata['lime_config']}\n")
            f.write(f"GPU Acceleration: {'Enabled' if self.use_gpu else 'Disabled'}\n")
            if self.use_gpu:
                f.write(f"GPU Batch Size: {self.gpu_batch_size}\n")
                f.write(f"Trained on GPU: {metadata['gpu_info']['trained_on_gpu']}\n")
            f.write(f"Dimension Reduction: {self.dimension_reduction_method.upper()}\n")
            f.write(f"Feature Selection: {self.feature_selection_method.upper()}\n")
            if self.feature_selection_method == 'brute_force':
                f.write(f"Brute-force Config: Max {self.brute_force_max_combinations} combinations\n")
                f.write(f"Brute-force Best AUC: {metadata['brute_force_params']['best_combination_auc']:.4f}\n")
            f.write(f"Model Name: {self.best_model_name}\n")
            f.write(f"Selected Features Count: {len(self.top_features)}\n")
            f.write(f"\nBest Parameters:\n")
            for k, v in self.best_model_params.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nPerformance Metrics:\n")
            f.write(
                f"  Train AUC: {self.best_model_metrics['train_auc']:.4f} (95% CI: {self.best_model_metrics['train_ci_lower']:.4f}-{self.best_model_metrics['train_ci_upper']:.4f})\n")
            f.write(
                f"  Validation AUC: {self.best_model_metrics['val_auc']:.4f} (95% CI: {self.best_model_metrics['val_ci_lower']:.4f}-{self.best_model_metrics['val_ci_upper']:.4f})\n")
            f.write(
                f"  5-fold CV Mean AUC: {self.best_model_metrics['cv_mean_auc']:.4f} (95% CI: {self.best_model_metrics['cv_ci_lower']:.4f}-{self.best_model_metrics['cv_ci_upper']:.4f})\n")
            f.write(f"\nSelected Features (Top 10):\n")
            for i, feat in enumerate(self.top_features[:10]):
                f.write(f"  {i+1}. {feat}\n")
            if len(self.top_features) > 10:
                f.write(f"  ... and {len(self.top_features)-10} more features\n")

        logger.info(f"Best model saved to: {save_dir}")

    def plot_selected_features_correlation(self):
        """Plot correlation heatmap of selected features (journal-grade)"""
        logger.info("Generating selected features correlation heatmap...")

        top_indices = [self.reduced_feature_names.index(f) for f in self.top_features]
        X_train_top = self.X_train_reduced[:, top_indices]
        corr_matrix = pd.DataFrame(X_train_top, columns=self.top_features).corr()

        n_features = len(self.top_features)
        fig_size = (min(12, n_features * 0.6), min(10, n_features * 0.5)) if n_features > 0 else self.figure_params[
            'medium']
        fig, ax = plt.subplots(figsize=fig_size)

        im = ax.imshow(corr_matrix, cmap='YlGnBu', vmin=-1, vmax=1, interpolation='nearest')

        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                corr_value = corr_matrix.iloc[i, j]
                text_color = 'white' if abs(corr_value) > 0.6 else 'black'
                ax.text(j, i, f'{corr_value:.2f}', ha='center', va='center',
                        color=text_color, fontsize=6, fontweight='bold')

        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(corr_matrix.columns, fontsize=7)

        ax.set_title(
            f'Correlation Heatmap of Selected Features)',
            pad=20, fontsize=10
        )

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Pearson Correlation Coefficient', fontsize=9, labelpad=10)
        cbar.ax.tick_params(labelsize=8)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root,self.figure_dir,
            f'selected_features_correlation_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'selected_features_correlation_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'selected_features_correlation_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

        logger.info(f"Correlation heatmap saved (filter: {self.feature_type_filter})")

    def _get_representative_samples(self, X, y, y_proba, class_idx, n_samples=2):
        """
        选择每个类别的代表性样本
        """
        true_class_mask = (y == class_idx)
        if not np.any(true_class_mask):
            logger.warning(f"No true samples for class {self.class_name_mapping[class_idx]}, using all samples")
            true_class_mask = np.ones_like(y, dtype=bool)

        class_proba = y_proba[true_class_mask, class_idx]
        sample_indices = np.where(true_class_mask)[0]

        sorted_idx = np.argsort(class_proba)[::-1]
        selected_sample_indices = sample_indices[sorted_idx[:n_samples]]

        logger.info(
            f"Selected representative samples for {self.class_name_mapping[class_idx]}: {selected_sample_indices}")
        return selected_sample_indices

    def plot_lime_explanations(self):
        """
        LIME局部可解释性分析
        """
        logger.info(f"\nGenerating LIME explanations (clinical-focused)...")
        if self.best_model is None:
            logger.warning("No best model found, skip LIME analysis")
            return

        top_indices = [self.reduced_feature_names.index(f) for f in self.top_features]
        X_val_top = self.X_val_reduced[:, top_indices]
        y_val_proba = self.best_model.predict_proba(X_val_top)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train_reduced[:, top_indices],
            feature_names=self.top_features,
            class_names=self.class_names,
            mode='classification',
            random_state=self.lime_random_state,
            discretize_continuous=False,
            verbose=False
        )

        lime_dir = os.path.join(self.save_root,self.figure_dir,f'lime_explanations_{self.best_model_name}_{self.feature_type_filter}_{self.feature_selection_method}')
        os.makedirs(lime_dir, exist_ok=True)
        lime_results = []

        for class_idx in range(self.n_classes):
            class_name = self.class_name_mapping[class_idx]
            logger.info(f"Processing LIME for class: {class_name}")

            sample_indices = self._get_representative_samples(
                X=X_val_top,
                y=self.y_val,
                y_proba=y_val_proba,
                class_idx=class_idx,
                n_samples=2
            )

            for sample_idx in sample_indices:
                sample_data = X_val_top[sample_idx:sample_idx + 1]
                true_label = self.y_val[sample_idx]
                true_label_name = self.class_name_mapping[true_label]
                pred_proba = y_val_proba[sample_idx, class_idx]

                explanation = explainer.explain_instance(
                    data_row=sample_data[0],
                    predict_fn=self.best_model.predict_proba,
                    num_samples=self.lime_n_samples,
                    num_features=self.lime_n_features,
                    top_labels=self.n_classes
                )

                class_explanation = explanation.as_list(label=class_idx)
                feature_names = [feat for feat, _ in class_explanation]
                contributions = [contrib for _, contrib in class_explanation]

                lime_df = pd.DataFrame({
                    'Feature_Name': feature_names,
                    'Contribution': contributions,
                    'Class': class_name,
                    'Sample_Index': sample_idx,
                    'True_Label': true_label_name,
                    'Predicted_Probability': pred_proba,
                    'Feature_Type': [
                        'Clinical' if feat in self.clinical_features else
                        'Lipid' if feat in self.lipid_features else
                        'Bile_Acid' for feat in feature_names
                    ],
                    'Clinical_Threshold': [
                        CLINICAL_THRESHOLDS.get(feat, 'N/A') for feat in feature_names
                    ]
                })
                lime_results.append(lime_df)

                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#E64B35' if c < 0 else '#4DBBD5' for c in contributions]
                bars = ax.barh(range(len(feature_names)), contributions, color=colors, alpha=0.8, edgecolor='black',
                               linewidth=0.6)

                feature_labels = []
                for feat in feature_names:
                    feat_type = 'Clinical' if feat in self.clinical_features else 'Lipid' if feat in self.lipid_features else 'Bile_Acid'
                    threshold = CLINICAL_THRESHOLDS.get(feat, 'N/A')
                    label = f"{feat}\n({feat_type}, Threshold: {threshold})"
                    feature_labels.append(label)

                ax.set_yticks(range(len(feature_names)))
                ax.set_yticklabels(feature_labels, fontsize=7)
                ax.set_xlabel('LIME Feature Contribution', fontsize=10)
                ax.set_title(
                    f'LIME Explanation - Class: {class_name}\nSample Index: {sample_idx} | True Label: {true_label_name} | Pred Prob: {pred_proba:.3f}',
                    pad=15, fontsize=11
                )

                ax.axvline(x=0, color='black', linestyle='-', linewidth=1.0, alpha=0.7)
                ax.grid(axis='x', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_linewidth(1.0)
                ax.spines['left'].set_linewidth(1.0)

                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#4DBBD5', alpha=0.8, label='Promote Diagnosis'),
                    Patch(facecolor='#E64B35', alpha=0.8, label='Inhibit Diagnosis')
                ]
                ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

                plt.tight_layout()
                plt.savefig(
                    os.path.join(lime_dir, f'lime_{class_name}_sample{sample_idx}_{self.feature_type_filter}.png'),
                    dpi=600, bbox_inches='tight'
                )
                plt.savefig(
                    os.path.join(lime_dir, f'lime_{class_name}_sample{sample_idx}_{self.feature_type_filter}.pdf'),
                    dpi=600,
                    format='pdf',
                    bbox_inches='tight'
                )
                plt.savefig(
                    os.path.join(lime_dir, f'lime_{class_name}_sample{sample_idx}_{self.feature_type_filter}.svg'),
                    dpi=600,
                    format='svg',
                    bbox_inches='tight'
                )
                plt.close()

        final_lime_df = pd.concat(lime_results, ignore_index=True)
        final_lime_df.to_csv(os.path.join(self.save_root, self.results_dir,
            f'lime_feature_contributions_{self.best_model_name}_{self.feature_type_filter}.csv'),
            index=False, encoding='utf-8-sig'
        )

        self._plot_lime_summary(final_lime_df)

        logger.info(
            f"LIME explanations saved to: {lime_dir} and results_3/lime_feature_contributions_{self.feature_selection_method}*.csv")

    def _plot_lime_summary(self, lime_df):
        """LIME汇总可视化：每个类别的平均特征贡献"""
        logger.info("Generating LIME summary plot...")

        lime_summary = lime_df.groupby(['Class', 'Feature_Name'])['Contribution'].mean().reset_index()
        lime_summary = lime_summary.sort_values(['Class', 'Contribution'], ascending=[True, False])

        top_features_per_class = []
        for cls in self.class_names:
            cls_data = lime_summary[lime_summary['Class'] == cls].head(3)
            top_features_per_class.append(cls_data)
        top_lime_summary = pd.concat(top_features_per_class, ignore_index=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(top_lime_summary))
        colors = [
            COLOR_PALETTE[
                f'class{list(self.class_name_mapping.keys())[list(self.class_name_mapping.values()).index(cls)]+1}']
            for cls in top_lime_summary['Class']
        ]

        bars = ax.bar(x, top_lime_summary['Contribution'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['Feature_Name']}\n({row['Class']})" for _, row in top_lime_summary.iterrows()],
                           rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Average LIME Contribution', fontsize=10)
        ax.set_title(
            f'LIME Summary - Top 3 Contributing Features per Class',
            pad=20, fontsize=11
        )

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.7, label='No Contribution')
        ax.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bar, contrib in zip(bars, top_lime_summary['Contribution']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + (0.01 if height > 0 else -0.03),
                    f'{contrib:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=7)

        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root,self.figure_dir,
            f'lime_summary_{self.best_model_name}_{self.feature_type_filter}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'lime_summary_{self.best_model_name}_{self.feature_type_filter}.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'lime_summary_{self.best_model_name}_{self.feature_type_filter}.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

        logger.info("LIME summary plot saved")

    def generate_visualizations(self):
        """Generate all journal-grade visualizations"""
        logger.info(
            f"\nStep 7/7: Generating visualizations (Filter: {self.feature_type_filter}, GPU: {self.use_gpu}, LIME: Enabled)...")

        self.plot_model_comparison()
        self.plot_cv_auc_comparison()
        self.plot_per_model_metrics()

        if self.feature_selection_method == 'lasso':
            self.plot_lasso_feature_importance()
        elif self.feature_selection_method == 'brute_force':
            self.plot_brute_force_results()
        elif self.feature_selection_method == 'sis':
            self.plot_sis_feature_importance()

        self.plot_dimension_reduction_visualization()

        self.plot_selected_features_correlation()
        self.plot_feature_target_violin_plots()
        self.plot_dca_curves()

        self.plot_lime_explanations()

        # ===== 新增代码 =====
        # ML vs 临床指南性能对比
        self.plot_clinical_vs_ml_performance()
        # PBC患者AMA-M2分层分析
        self.plot_pbc_ama_m2_stratified_analysis()
        # ===== 新增 =====
        # AIH典型/非典型分层分析
        self.plot_aih_stratified_analysis()
        # OS典型/非典型分层分析
        self.plot_os_stratified_analysis()

        # 新增：调用校准曲线函数
        self.plot_calibration_curves()
        # ====================
        # ========== 新增调用：写入预测结果到Excel ==========
        self.write_predictions_to_excel()

        # 最终模型LR回归系数图（若最优模型为LR则输出）
        self.plot_lr_coefficients()

        logger.info(f"All visualizations saved to 'figure_3' (filter: {self.feature_type_filter}, including LIME)")

    def plot_brute_force_results(self):
        """Visualize brute force results (AUC distribution + top combinations)"""
        if self.brute_force_results is None:
            logger.warning("No brute force results to visualize")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.hist(self.brute_force_results['val_auc'], bins=30, color=COLOR_PALETTE['cv'],
                 alpha=0.8, edgecolor='black', linewidth=0.6)
        ax1.axvline(self.brute_force_results['val_auc'].max(), color='darkred', linestyle='--',
                    linewidth=1.5, label=f'Best AUC: {self.brute_force_results["val_auc"].max():.4f}')
        ax1.axvline(self.brute_force_results['val_auc'].mean(), color='navy', linestyle='-',
                    linewidth=1.5, label=f'Mean AUC: {self.brute_force_results["val_auc"].mean():.4f}')
        ax1.set_xlabel('Validation AUC (OvR)', fontsize=9)
        ax1.set_ylabel('Number of Feature Combinations', fontsize=9)
        ax1.set_title(
            f'Brute-force AUC Distribution\n Top {self.n_selected} Features)',
            fontsize=10
        )
        ax1.legend(fontsize=8)
        ax1.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        top10_results = self.brute_force_results.head(10).copy()
        top10_results['combination_label'] = [f'Comb_{i+1}' for i in range(10)]

        bars = ax2.barh(range(len(top10_results)), top10_results['val_auc'],
                        color=COLOR_PALETTE['val'], alpha=0.8, edgecolor='black', linewidth=0.6)
        ax2.set_yticks(range(len(top10_results)))
        ax2.set_yticklabels(top10_results['combination_label'], fontsize=8)
        ax2.set_xlabel('Validation AUC (OvR)', fontsize=9)
        ax2.set_title('Top 10 Combinations by AUC', fontsize=10)
        ax2.grid(axis='x', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        for i, (bar, auc) in enumerate(zip(bars, top10_results['val_auc'])):
            ax2.text(auc + 0.005, bar.get_y() + bar.get_height() / 2, f'{auc:.3f}',
                     ha='left', va='center', fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root,self.figure_dir,
            f'brute_force_results_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_top{self.n_selected}.png'),
            dpi=600
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'brute_force_results_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_top{self.n_selected}.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'brute_force_results_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_top{self.n_selected}.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4))
        best_features = self.top_features
        feature_lines = []
        for i in range(0, len(best_features), 3):
            line = ' | '.join(best_features[i:i + 3])
            feature_lines.append(line)
        feature_text = '\n'.join(feature_lines)

        display_text = (f"Best Feature Combination (Filter: {self.feature_type_filter})\n"
                        f"Validation AUC: {self.brute_force_results['val_auc'].max():.4f}\n"
                        f"Base Model: {self.brute_force_base_model.__class__.__name__}\n"
                        f"Features:\n{feature_text}")

        ax.text(0.05, 0.95, display_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                                                   facecolor=COLOR_PALETTE['cv'], alpha=0.3, edgecolor='gray'))
        ax.axis('off')
        plt.title('Brute-force Best Combination Details', fontsize=11, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root,self.figure_dir,
            f'brute_force_best_combination_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_top{self.n_selected}.png'),
            dpi=600
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'brute_force_best_combination_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_top{self.n_selected}.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'brute_force_best_combination_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_top{self.n_selected}.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

    def plot_model_comparison(self):
        """Model AUC comparison plot (train/validation/CV with CI)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        model_names = list(self.model_results.keys())
        x = np.arange(len(model_names))
        width = 0.25

        train_aucs = [self.model_results[m]['train_auc'] for m in model_names]
        val_aucs = [self.model_results[m]['val_auc'] for m in model_names]
        cv_aucs = [self.model_results[m]['cv_mean_auc'] for m in model_names]

        train_ci_lower = [self.model_results[m]['train_ci_lower'] for m in model_names]
        train_ci_upper = [self.model_results[m]['train_ci_upper'] for m in model_names]
        val_ci_lower = [self.model_results[m]['val_ci_lower'] for m in model_names]
        val_ci_upper = [self.model_results[m]['val_ci_upper'] for m in model_names]
        cv_ci_lower = [self.model_results[m]['cv_ci_lower'] for m in model_names]
        cv_ci_upper = [self.model_results[m]['cv_ci_upper'] for m in model_names]

        bars1 = ax.bar(x - width, train_aucs, width, label='Train AUC',
                       color=COLOR_PALETTE['train'], alpha=0.8, edgecolor='black', linewidth=0.6)
        bars2 = ax.bar(x, val_aucs, width, label='Validation AUC',
                       color=COLOR_PALETTE['val'], alpha=0.8, edgecolor='black', linewidth=0.6)
        bars3 = ax.bar(x + width, cv_aucs, width, label='CV Mean AUC',
                       color=COLOR_PALETTE['cv'], alpha=0.8, edgecolor='black', linewidth=0.6)

        ax.errorbar(x - width, train_aucs,
                    yerr=[np.subtract(train_aucs, train_ci_lower), np.subtract(train_ci_upper, train_aucs)],
                    fmt='none', c='black', capsize=4, capthick=1.0, elinewidth=1.0)
        ax.errorbar(x, val_aucs,
                    yerr=[np.subtract(val_aucs, val_ci_lower), np.subtract(val_ci_upper, val_aucs)],
                    fmt='none', c='black', capsize=4, capthick=1.0, elinewidth=1.0)
        ax.errorbar(x + width, cv_aucs,
                    yerr=[np.subtract(cv_aucs, cv_ci_lower), np.subtract(cv_ci_upper, cv_aucs)],
                    fmt='none', c='black', capsize=4, capthick=1.0, elinewidth=1.0)

        ax.set_xlabel('Machine Learning Models', fontsize=10)
        ax.set_ylabel('AUC (One-vs-Rest)', fontsize=10)
        ax.set_title(
            f'Model Performance Comparison',
            pad=20, fontsize=11
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0.5, 1.05)
        ax.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root,self.figure_dir,
            f'model_comparison_auc_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.png'),
            dpi=600
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'model_comparison_auc_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'model_comparison_auc_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

    def plot_cv_auc_comparison(self):
        """5-fold CV AUC distribution box plot"""
        fig, ax = plt.subplots(figsize=(8, 5))
        model_names = list(self.cv_results.keys())
        cv_fold_aucs = [self.cv_results[m]['cv_fold_auc'] for m in model_names]

        box_plot = ax.boxplot(cv_fold_aucs, labels=model_names, patch_artist=True,
                              boxprops=dict(alpha=0.7), medianprops=dict(color='darkred', linewidth=1.5),
                              whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))

        colors = [COLOR_PALETTE['train'], COLOR_PALETTE['val'], COLOR_PALETTE['cv'], COLOR_PALETTE['class1']]
        for patch, color in zip(box_plot['boxes'], colors[:len(model_names)]):
            patch.set_facecolor(color)

        ax.set_xlabel('Machine Learning Models', fontsize=10)
        ax.set_ylabel('AUC (One-vs-Rest)', fontsize=10)
        ax.set_title(
            f'5-fold CV AUC Distribution',
            pad=15, fontsize=11
        )
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for i, (model, aucs) in enumerate(zip(model_names, cv_fold_aucs)):
            mean_auc = np.mean(aucs)
            ax.scatter(i + 1, mean_auc, color='black', s=40, zorder=5, marker='*', label='Mean' if i == 0 else "")
            ax.text(i + 1, mean_auc + 0.01, f'{mean_auc:.3f}', ha='center', va='bottom', fontsize=7)

        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root,self.figure_dir,
            f'cv_auc_distribution_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.png'),
            dpi=600
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'cv_auc_distribution_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'cv_auc_distribution_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

    def plot_per_model_metrics(self):
        """Per-model metrics (ROC/PR/confusion matrix/SHAP)"""
        colors = [COLOR_PALETTE['class1'], COLOR_PALETTE['class2'],
                  COLOR_PALETTE['class3'], COLOR_PALETTE['class4']]
        target_names = self.class_names
        top_indices = [self.reduced_feature_names.index(f) for f in self.top_features]
        X_val_top = self.X_val_reduced[:, top_indices]

        for name, metrics in self.model_results.items():
            logger.info(f"Generating metrics for {name} (filter: {self.feature_type_filter})...")

            # ---- ROC 曲线（多分类 one-vs-rest，CI阴影 + 规范坐标轴） ----
            fig, ax = plt.subplots(figsize=(5, 5))
            for i in range(self.n_classes):
                class_name = self.class_name_mapping[i]
                y_bin = (self.y_val == i).astype(int)
                proba_i = metrics['y_val_proba'][:, i]
                fpr, tpr, _ = roc_curve(y_bin, proba_i)
                auc = roc_auc_score(y_bin, proba_i)
                # 95% CI（bootstrap）
                np.random.seed(42)
                auc_boots = []
                for _ in range(500):
                    idx = np.random.choice(len(y_bin), len(y_bin), replace=True)
                    if len(np.unique(y_bin[idx])) < 2:
                        continue
                    auc_boots.append(roc_auc_score(y_bin[idx], proba_i[idx]))
                ci_lo = np.percentile(auc_boots, 2.5) if auc_boots else auc
                ci_hi = np.percentile(auc_boots, 97.5) if auc_boots else auc
                color = colors[i]
                label_str = f"{class_name}\nAUC={auc:.3f} ({ci_lo:.3f}–{ci_hi:.3f})"
                ax.plot(fpr, tpr, color=color, lw=1.8, label=label_str, alpha=0.9, zorder=3)
                # # CI 阴影
                # fpr_grid = np.linspace(0, 1, 200)
                # tpr_boots = []
                # for _ in range(200):
                #     idx = np.random.choice(len(y_bin), len(y_bin), replace=True)
                #     if len(np.unique(y_bin[idx])) < 2:
                #         continue
                #     f_b, t_b, _ = roc_curve(y_bin[idx], proba_i[idx])
                #     tpr_boots.append(np.interp(fpr_grid, f_b, t_b))
                # if tpr_boots:
                #     tpr_arr = np.array(tpr_boots)
                #     ax.fill_between(fpr_grid,
                #                     np.percentile(tpr_arr, 2.5, axis=0),
                #                     np.percentile(tpr_arr, 97.5, axis=0),
                #                     alpha=0.10, color=color, zorder=1)

            ax.plot([0, 1], [0, 1], linestyle=':', lw=1.0, color='#AAAAAA',
                    label='Random Guess', alpha=0.8, zorder=0)
            ax.set_xlabel('1 - Specificity (FPR)', fontsize=9, labelpad=8)
            ax.set_ylabel('Sensitivity (TPR)', fontsize=9, labelpad=8)
            ax.set_title(f'ROC Curves — {name}\n(Validation Set, One-vs-Rest)',
                         fontsize=9, pad=10, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.01])
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='both', color=COLOR_PALETTE['grid'], linestyle='-',
                    linewidth=0.3, alpha=0.6)
            ax.set_axisbelow(True)
            ax.legend(loc='lower right', frameon=True, framealpha=0.92,
                      fontsize=6.5, labelspacing=0.4, handlelength=1.5,
                      borderpad=0.6, edgecolor='#CCCCCC')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_root,self.figure_dir,
                f'froc_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_nature_style.png'),
                dpi=600)
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'froc_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_nature_style.pdf'),
                dpi=600,
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'froc_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_nature_style.svg'),
                dpi=600,
                format='svg',
                bbox_inches='tight'
            )
            plt.close()

            fig, ax = plt.subplots(figsize=(6, 5))
            for i in range(self.n_classes):
                class_name = self.class_name_mapping[i]
                precision, recall, _ = precision_recall_curve(self.y_val == i, metrics['y_val_proba'][:, i])
                ap = average_precision_score(self.y_val == i, metrics['y_val_proba'][:, i])
                ax.plot(recall, precision, color=colors[i], lw=1.5, label=f'{class_name} (AP={ap:.3f})', alpha=0.9)

            ax.set_xlabel('Recall', fontsize=10)
            ax.set_ylabel('Precision', fontsize=10)
            ax.set_title(
                f'Precision-Recall Curves - {name}',
                fontsize=11, pad=15
            )
            ax.legend(loc='upper right', frameon=True, fontsize=8, framealpha=0.9)
            ax.grid(axis='both', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_root,self.figure_dir,
                f'pr_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.png'),
                dpi=600
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'pr_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.pdf'),
                dpi=600,
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'pr_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.svg'),
                dpi=600,
                format='svg',
                bbox_inches='tight'
            )
            plt.close()

            cm = confusion_matrix(self.y_val, metrics['y_val_pred'])
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap='YlGnBu', vmin=0, vmax=1)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                    text = ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                                   ha="center", va="center", color=text_color,
                                   fontsize=9, fontweight='bold')

            ax.set_xlabel('Predicted Label', fontsize=11, labelpad=15)
            ax.set_ylabel('True Label', fontsize=11, labelpad=15)
            ax.set_title(
                f'Confusion Matrix - {name}',
                fontsize=12, pad=20
            )
            ax.set_xticks(np.arange(self.n_classes))
            ax.set_yticks(np.arange(self.n_classes))
            ax.set_xticklabels(target_names, rotation=0, ha='center', fontsize=10)
            ax.set_yticklabels(target_names, fontsize=10)

            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color('black')

            cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.12)
            cbar.set_label('Normalized Score', fontsize=10, labelpad=10)
            cbar.ax.tick_params(labelsize=9)

            plt.subplots_adjust(top=0.85, bottom=0.12, left=0.12, right=0.88)
            plt.savefig(os.path.join(self.save_root,self.figure_dir,
                f'fcm_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.png'),
                dpi=600
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'fcm_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.pdf'),
                dpi=600,
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'fcm_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.svg'),
                dpi=600,
                format='svg',
                bbox_inches='tight'
            )
            plt.close()

            if name in ['RandomForest', 'XGBoost', 'LogisticRegression']:
                try:
                    # 1. 根据模型类型选择 SHAP 解释器
                    if name == 'LogisticRegression':
                        # 逻辑回归使用 LinearExplainer
                        explainer = shap.LinearExplainer(
                            self.models[name][0],
                            self.X_train_reduced[:, top_indices],  # 训练集top特征
                            feature_dependence="independent"
                        )
                        shap_values = explainer.shap_values(self.X_val_reduced[:, top_indices])
                    else:
                        # 树模型使用 TreeExplainer
                        explainer = shap.TreeExplainer(self.models[name][0])
                        shap_values = explainer.shap_values(self.X_val_reduced[:, top_indices], check_additivity=False)

                    # 2. 统一处理 SHAP 值格式（适配多分类）
                    if isinstance(shap_values, list):
                        # 多分类场景：计算各类别平均绝对值
                        shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                    elif isinstance(shap_values, np.ndarray):
                        if shap_values.ndim == 3:
                            shap_importance = np.mean(np.abs(shap_values), axis=(0, 1))
                        else:
                            shap_importance = np.mean(np.abs(shap_values), axis=0)
                    else:
                        raise ValueError(f"Unsupported SHAP shape for {name}: {np.shape(shap_values)}")

                    assert len(shap_importance) == self.n_selected, \
                        f"SHAP importance length mismatch: {len(shap_importance)} vs {self.n_selected}"

                    # 3. 绘制 SHAP 特征重要性条形图（单图）
                    shap_series = pd.Series(shap_importance, index=self.top_features).sort_values(ascending=True)
                    fig_height = max(5, 0.4 * len(shap_series))
                    fig, ax = plt.subplots(figsize=(6, fig_height))

                    bars = ax.barh(range(len(shap_series)), shap_series.values,
                                   color=COLOR_PALETTE['shap'], alpha=0.8, edgecolor='black', linewidth=0.6)

                    ax.set_yticks(range(len(shap_series)))
                    ax.set_yticklabels(shap_series.index, fontsize=8)
                    ax.set_xlabel('SHAP Importance (Average Absolute Value)', fontsize=10)
                    ax.set_title(
                        f'SHAP Feature Importance - {name}',
                        fontsize=11, pad=15
                    )
                    ax.grid(axis='x', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
                    ax.set_axisbelow(True)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xlim(0, max(shap_series.values) * 1.1)

                    plt.tight_layout()
                    plt.savefig(os.path.join(self.save_root, self.figure_dir,
                                             f'shap_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.png'),
                                dpi=600
                                )
                    plt.savefig(
                        os.path.join(self.save_root, self.figure_dir,
                                     f'shap_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.pdf'),
                        dpi=600,
                        format='pdf',
                        bbox_inches='tight'
                    )
                    plt.savefig(
                        os.path.join(self.save_root, self.figure_dir,
                                     f'shap_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.svg'),
                        dpi=600,
                        format='svg',
                        bbox_inches='tight'
                    )
                    plt.close()

                    # 4. 补充 SHAP 森林图（Summary Plot，多分类/二分类适配）
                    plt.figure(figsize=(8, 5))
                    if name == 'LogisticRegression':
                        # 逻辑回归多分类：取目标类别（如 PBC=1）或平均
                        if isinstance(shap_values, list):
                            # 多分类时选择 PBC 类别（可根据需求调整）
                            shap.summary_plot(
                                shap_values[1],  # 1=PBC 类别
                                features=self.X_val_reduced[:, top_indices],
                                feature_names=self.top_features,
                                plot_type="dot",  # 森林图（dot/violin 可选）
                                show=False,
                                color=COLOR_PALETTE['shap']
                            )
                        else:
                            shap.summary_plot(
                                shap_values,
                                features=self.X_val_reduced[:, top_indices],
                                feature_names=self.top_features,
                                show=False
                            )
                    else:
                        # 树模型直接绘制
                        shap.summary_plot(
                            shap_values,
                            features=self.X_val_reduced[:, top_indices],
                            feature_names=self.top_features,
                            show=False
                        )
                    plt.title(f'SHAP Summary Plot (Forest Plot) - {name}', fontsize=11, pad=15)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.save_root, self.figure_dir,
                                             f'shap_forest_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.png'),
                                dpi=600, bbox_inches='tight'
                                )
                    plt.savefig(
                        os.path.join(self.save_root, self.figure_dir,
                                     f'shap_forest_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.pdf'),
                        dpi=600,
                        format='pdf',
                        bbox_inches='tight'
                    )
                    plt.savefig(
                        os.path.join(self.save_root, self.figure_dir,
                                     f'shap_forest_{name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.svg'),
                        dpi=600,
                        format='svg',
                        bbox_inches='tight'
                    )
                    plt.close()

                    # 5. SHAP Dependence plots (top-2 features by importance)
                    X_top_feat = self.X_val_reduced[:, top_indices]
                    # For multi-class list, use mean absolute SHAP across classes
                    if isinstance(shap_values, list):
                        shap_for_dep = np.mean(np.abs(shap_values), axis=0)  # (n_samples, n_features)
                        shap_vals_dep = shap_values  # keep list for dependence
                    else:
                        shap_for_dep = shap_values if shap_values.ndim == 2 else np.mean(np.abs(shap_values), axis=0)
                        shap_vals_dep = shap_values

                    top2_idx = np.argsort(shap_importance)[::-1][:2]
                    for feat_idx in top2_idx:
                        feat_name = self.top_features[feat_idx]
                        # Use class-0 SHAP values for dependence (or 2D array directly)
                        sv_dep = shap_vals_dep[0] if isinstance(shap_vals_dep, list) else (
                            shap_vals_dep[:, :, 0] if shap_vals_dep.ndim == 3 else shap_vals_dep
                        )
                        fig_dep, ax_dep = plt.subplots(figsize=(5, 4))
                        shap.dependence_plot(
                            feat_idx,
                            sv_dep,
                            X_top_feat,
                            feature_names=self.top_features,
                            ax=ax_dep,
                            show=False,
                        )
                        ax_dep.set_title(f"SHAP Dependence: {feat_name} ({name})", fontsize=9, pad=8)
                        plt.tight_layout()
                        safe_feat = feat_name.replace("/", "_").replace("(", "").replace(")", "").replace(" ", "_")
                        dep_base = f'shap_dep_{safe_feat}_{name}_{self.feature_selection_method}_{self.feature_type_filter}'
                        for ext, fmt in [('.png', None), ('.pdf', 'pdf'), ('.svg', 'svg')]:
                            kw = dict(dpi=600, bbox_inches='tight')
                            if fmt:
                                kw['format'] = fmt
                            plt.savefig(os.path.join(self.save_root, self.figure_dir, dep_base + ext), **kw)
                        plt.close()

                    # 6. SHAP Interaction plot (top feature vs top-2 interactor) — tree models only
                    if name in ['RandomForest', 'XGBoost']:
                        try:
                            interaction_values = shap.TreeExplainer(self.models[name][0]).shap_interaction_values(
                                X_top_feat
                            )
                            # interaction_values: (n_samples, n_features, n_features) or list
                            if isinstance(interaction_values, list):
                                interaction_values = np.mean(np.abs(interaction_values), axis=0)
                            elif interaction_values.ndim == 4:
                                # (n_classes, n_samples, n_feat, n_feat) → mean over classes
                                interaction_values = np.mean(np.abs(interaction_values), axis=0)
                            # mean over samples → (n_feat, n_feat)
                            mean_inter = np.mean(np.abs(interaction_values), axis=0)
                            np.fill_diagonal(mean_inter, 0)
                            fig_inter, ax_inter = plt.subplots(figsize=(5, 4))
                            im = ax_inter.imshow(mean_inter, cmap='YlOrRd', aspect='auto')
                            ax_inter.set_xticks(range(len(self.top_features)))
                            ax_inter.set_yticks(range(len(self.top_features)))
                            ax_inter.set_xticklabels(self.top_features, rotation=45, ha='right', fontsize=6)
                            ax_inter.set_yticklabels(self.top_features, fontsize=6)
                            plt.colorbar(im, ax=ax_inter, label='Mean |SHAP interaction|', fraction=0.03, pad=0.04)
                            ax_inter.set_title(f'SHAP Interaction Heatmap ({name})', fontsize=9, pad=8, fontweight='bold')
                            plt.tight_layout()
                            inter_base = f'shap_interaction_{name}_{self.feature_selection_method}_{self.feature_type_filter}'
                            for ext, fmt in [('.png', None), ('.pdf', 'pdf'), ('.svg', 'svg')]:
                                kw = dict(dpi=600, bbox_inches='tight')
                                if fmt:
                                    kw['format'] = fmt
                                plt.savefig(os.path.join(self.save_root, self.figure_dir, inter_base + ext), **kw)
                            plt.close()
                            logger.info(f"SHAP interaction heatmap saved for {name}")
                        except Exception as e_inter:
                            logger.warning(f"SHAP interaction plot failed for {name}: {e_inter}")

                    logger.info(f"SHAP plots (bar + forest + dependence) saved for {name} (filter: {self.feature_type_filter})")
                except Exception as e:
                    logger.warning(f"SHAP plot generation failed for {name}: {str(e)}")


    def plot_lr_coefficients(self):
        """Output LogisticRegression coefficients for the best model (if LR is best)."""
        if self.best_model_name != 'LogisticRegression':
            logger.info(f"Best model is {self.best_model_name}, skipping LR coefficient plot.")
            return

        lr_model = self.best_model
        coef = lr_model.coef_  # shape: (n_classes, n_features) for multi-class, (1, n_features) for binary
        feature_names = self.top_features
        n_classes = coef.shape[0]

        # Build DataFrame: rows=features, columns=classes
        class_labels = [self.class_name_mapping.get(i, f'Class_{i}') for i in range(n_classes)]
        coef_df = pd.DataFrame(coef.T, index=feature_names, columns=class_labels)
        coef_df.index.name = 'Feature'

        # Save to CSV
        coef_csv = os.path.join(
            self.save_root, self.figure_dir,
            f'lr_coefficients_{self.feature_selection_method}_{self.feature_type_filter}.csv'
        )
        coef_df.to_csv(coef_csv)
        logger.info(f"LR coefficients saved to {coef_csv}")

        # Plot: grouped horizontal bar chart (one group per feature, bars = classes)
        n_feat = len(feature_names)
        group_h = 0.72          # total height allocated per feature group
        bar_h = group_h / n_classes * 0.82   # individual bar height with small gap
        fig_h = max(5.0, n_feat * (group_h + 0.15) + 1.5)
        fig_w = 9.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        x = np.arange(n_feat, dtype=float)
        class_colors = [COLOR_PALETTE.get(f'class{i+1}', f'C{i}') for i in range(n_classes)]

        for i, (cls_label, color) in enumerate(zip(class_labels, class_colors)):
            offset = (i - (n_classes - 1) / 2.0) * (group_h / n_classes)
            vals = coef_df[cls_label].values
            bars = ax.barh(
                x + offset, vals,
                height=bar_h, color=color, alpha=0.88,
                edgecolor='white', linewidth=0.6, label=cls_label
            )
            # 在每个 bar 末端标注数值
            for bar, v in zip(bars, vals):
                if abs(v) > 1e-6:
                    xpos = v + (0.015 if v >= 0 else -0.015)
                    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                            f'{v:.2f}', va='center',
                            ha='left' if v >= 0 else 'right',
                            fontsize=7.5, color='#333333')

        ax.axvline(0, color='#333333', linewidth=1.2, linestyle='--', zorder=3)
        ax.set_yticks(x)
        ax.set_yticklabels(feature_names, fontsize=10)
        ax.set_xlabel('Coefficient value', fontsize=11, fontweight='bold')
        ax.set_title(
            f'Logistic Regression Coefficients\n({self.feature_selection_method.upper()} | {self.feature_type_filter})',
            fontsize=11, pad=10, fontweight='bold'
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.grid(axis='x', linestyle='--', linewidth=0.6, alpha=0.4, zorder=0)
        ax.set_ylim(-0.6, n_feat - 0.4)
        legend = ax.legend(
            title='Class', title_fontsize=10,
            frameon=True, framealpha=0.9, edgecolor='#cccccc',
            fontsize=9, loc='lower right'
        )
        plt.tight_layout(pad=1.5)

        base = f'lr_coefficients_{self.feature_selection_method}_{self.feature_type_filter}'
        for ext, fmt in [('.png', None), ('.pdf', 'pdf'), ('.svg', 'svg')]:
            kw = dict(dpi=600, bbox_inches='tight')
            if fmt:
                kw['format'] = fmt
            plt.savefig(os.path.join(self.save_root, self.figure_dir, base + ext), **kw)
        plt.close()
        logger.info("LR coefficient plot saved.")

        # ── Heatmap version ──────────────────────────────────────────────
        # coef_df: rows=features, cols=classes; values are raw coefficients
        mat = coef_df.values          # shape (n_feat, n_classes)
        abs_max = np.abs(mat).max()
        if abs_max == 0:
            abs_max = 1.0

        cell_w = 1.4                  # width per class column (inches)
        cell_h = max(0.45, 5.0 / max(n_feat, 1))  # height per feature row
        fig_w_hm = cell_w * n_classes + 3.0
        fig_h_hm = max(3.5, cell_h * n_feat + 1.5)

        fig2, ax2 = plt.subplots(figsize=(fig_w_hm, fig_h_hm))

        im = ax2.imshow(
            mat, aspect='auto', cmap='RdBu_r',
            vmin=-abs_max, vmax=abs_max,
            interpolation='nearest'
        )

        # Annotate each cell with the coefficient value
        for row_i in range(n_feat):
            for col_j in range(n_classes):
                v = mat[row_i, col_j]
                # choose text color for contrast
                txt_color = 'white' if abs(v) > abs_max * 0.6 else '#222222'
                ax2.text(col_j, row_i, f'{v:.2f}',
                         ha='center', va='center',
                         fontsize=9, color=txt_color, fontweight='bold')

        # Axes decoration
        ax2.set_xticks(np.arange(n_classes))
        ax2.set_xticklabels(class_labels, fontsize=11, fontweight='bold')
        ax2.set_yticks(np.arange(n_feat))
        ax2.set_yticklabels(feature_names, fontsize=10)
        ax2.xaxis.set_ticks_position('bottom')

        # Thin grid lines between cells
        ax2.set_xticks(np.arange(n_classes + 1) - 0.5, minor=True)
        ax2.set_yticks(np.arange(n_feat + 1) - 0.5, minor=True)
        ax2.grid(which='minor', color='white', linewidth=1.5)
        ax2.tick_params(which='minor', length=0)

        # Colorbar
        cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
        cbar.set_label('Coefficient value', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        ax2.set_title(
            f'LR Coefficients Heatmap\n({self.feature_selection_method.upper()} | {self.feature_type_filter})',
            fontsize=11, pad=10, fontweight='bold'
        )
        plt.tight_layout(pad=1.5)

        base_hm = f'lr_coefficients_heatmap_{self.feature_selection_method}_{self.feature_type_filter}'
        for ext, fmt in [('.png', None), ('.pdf', 'pdf'), ('.svg', 'svg')]:
            kw = dict(dpi=600, bbox_inches='tight')
            if fmt:
                kw['format'] = fmt
            plt.savefig(os.path.join(self.save_root, self.figure_dir, base_hm + ext), **kw)
        plt.close()
        logger.info("LR coefficient heatmap saved.")

    def plot_sis_feature_importance(self):
        """SIS feature importance plot (weighted score for multi-class)"""
        sis_df = self.sis_scores.sort_values('Weighted_SIS_Score', ascending=False).head(20).copy()
        sis_df = sis_df.iloc[::-1].reset_index(drop=True)  # 反转：最重要在顶部

        # 归一化为百分比
        total = sis_df['Weighted_SIS_Score'].sum()
        sis_df['pct'] = sis_df['Weighted_SIS_Score'] / total * 100 if total > 0 else sis_df['Weighted_SIS_Score']

        # 按特征类型着色
        def get_feat_color(fname):
            if fname in (self.bile_acid_features or []):
                return COLOR_PALETTE['val']       # 紫色：胆汁酸
            elif fname in (self.lipid_features or []):
                return COLOR_PALETTE['train']     # 蓝色：脂质
            else:
                return COLOR_PALETTE['cv']        # 橙色：临床

        bar_colors = [
            COLOR_PALETTE['class1'] if row['Feature'] in self.top_features
            else get_feat_color(row['Feature'])
            for _, row in sis_df.iterrows()
        ]

        fig_height = max(3.0, len(sis_df) * 0.42)
        fig, ax = plt.subplots(figsize=(6.5, fig_height))
        bars = ax.barh(range(len(sis_df)), sis_df['pct'],
                       color=bar_colors, alpha=0.85,
                       edgecolor='white', linewidth=0.3, height=0.65, zorder=2)

        x_max_val = sis_df['pct'].max()
        for bar, pct in zip(bars, sis_df['pct']):
            ax.text(bar.get_width() + x_max_val * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f'{pct:.1f}%', ha='left', va='center',
                    fontsize=7, color='#333333', fontweight='bold', zorder=3)

        y_labels = [n if len(n) <= 22 else f"{n[:19]}..." for n in sis_df['Feature']]
        ax.set_yticks(range(len(sis_df)))
        ax.set_yticklabels(y_labels, fontsize=7.5)
        ax.set_xlabel(f'Relative SIS Score (%)', fontsize=9, labelpad=8)
        ax.set_title(f'SIS Feature Importance (Top 20)\n{self.feature_selection_method.upper()} | {self.feature_type_filter}',
                     fontsize=10, pad=10, fontweight='bold')
        ax.set_xlim(0, x_max_val * 1.18)
        ax.tick_params(axis='x', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.grid(axis='x', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3, alpha=0.7, zorder=1)
        ax.set_axisbelow(True)

        from matplotlib.patches import Patch
        legend_patches = [
            Patch(color=COLOR_PALETTE['class1'], alpha=0.85, label=f'Selected ({self.n_selected})'),
            Patch(color=COLOR_PALETTE['cv'],    alpha=0.85, label='Clinical'),
            Patch(color=COLOR_PALETTE['val'],   alpha=0.85, label='Bile Acid'),
            Patch(color=COLOR_PALETTE['train'], alpha=0.85, label='Lipid'),
        ]
        ax.legend(handles=legend_patches, fontsize=7, loc='lower right',
                  frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
                  handlelength=1.2, borderpad=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root, self.figure_dir,
            f'sis_feature_importance_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.png'),
            dpi=600, bbox_inches='tight')
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'sis_feature_importance_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.pdf'),
            dpi=600, format='pdf', bbox_inches='tight')
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'sis_feature_importance_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.svg'),
            dpi=600, format='svg', bbox_inches='tight')
        plt.close()

    def plot_lasso_feature_importance(self):
        """Lasso feature importance plot (GPU-optimized)"""
        n_reduced_features = len(self.reduced_feature_names)
        lasso_importance = np.zeros(n_reduced_features)

        for class_idx in range(self.n_classes):
            y_binary = (self.y_train == class_idx).astype(int)
            X_train_float64 = self.X_train_reduced.astype(np.float64)
            lasso = LassoCV(
                cv=5, random_state=42, max_iter=10000,
                n_jobs=1, precompute=False, tol=1e-4
            )
            lasso.fit(X_train_float64, y_binary)
            lasso_importance += np.abs(lasso.coef_)

        lasso_importance_avg = lasso_importance / self.n_classes
        lasso_df = pd.DataFrame({
            'Feature': self.reduced_feature_names,
            'Importance': lasso_importance_avg
        }).sort_values('Importance', ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(lasso_df)), lasso_df['Importance'],
                       color=COLOR_PALETTE['shap'], alpha=0.8, edgecolor='black', linewidth=0.6)

        for i, (idx, row) in enumerate(lasso_df.iterrows()):
            if row['Feature'] in self.top_features:
                bars[i].set_color(COLOR_PALETTE['class1'])
                bars[i].set_alpha(0.9)

        ax.set_yticks(range(len(lasso_df)))
        ax.set_yticklabels(lasso_df['Feature'], fontsize=8)
        ax.set_xlabel('Lasso Importance (Average Absolute Coefficient)', fontsize=10)
        ax.set_title(
            f'Top 20 Lasso Feature Importance\n',
            fontsize=11, pad=15
        )
        ax.grid(axis='x', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=COLOR_PALETTE['shap'], alpha=0.8, label='Top 20 Features'),
            Patch(facecolor=COLOR_PALETTE['class1'], alpha=0.9, label=f'Selected {self.n_selected} Features')
        ], fontsize=9, loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root,self.figure_dir,
            f'lasso_feature_importance_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.png'),
            dpi=600
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'lasso_feature_importance_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'lasso_feature_importance_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

    def plot_dimension_reduction_visualization(self):
        """Dimension reduction visualization (PCA/SelectKBest)"""
        if self.dimension_reduction_method == 'pca' and self.X_train_reduced.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            for class_idx, class_name in self.class_name_mapping.items():
                mask = self.y_train == class_idx
                ax.scatter(self.X_train_reduced[mask, 0], self.X_train_reduced[mask, 1],
                           label=f'{class_name} (Train)', alpha=0.7, s=40,
                           color=COLOR_PALETTE[f'class{class_idx+1}'], edgecolors='black', linewidth=0.5)

            for class_idx, class_name in self.class_name_mapping.items():
                mask = self.y_val == class_idx
                ax.scatter(self.X_val_reduced[mask, 0], self.X_val_reduced[mask, 1],
                           label=f'{class_name} (Val)', alpha=0.7, s=40, marker='s',
                           color=COLOR_PALETTE[f'class{class_idx+1}'], edgecolors='black', linewidth=0.5)

            ax.set_xlabel(f'PC1 ({self.dimension_reducer.explained_variance_ratio_[0]:.2%} variance)', fontsize=10)
            ax.set_ylabel(f'PC2 ({self.dimension_reducer.explained_variance_ratio_[1]:.2%} variance)', fontsize=10)
            ax.set_title(
                f'PCA Visualization (Top 2 Components)\n',
                pad=15, fontsize=11
            )
            ax.legend(fontsize=9, loc='best')
            ax.grid(color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_root,self.figure_dir,
                f'pca_visualization_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.png'),
                dpi=600
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'pca_visualization_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.pdf'),
                dpi=600,
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'pca_visualization_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.svg'),
                dpi=600,
                format='svg',
                bbox_inches='tight'
            )
            plt.close()

        elif self.dimension_reduction_method == 'selectkbest':
            selectkbest_scores = pd.DataFrame({
                'Feature': self.feature_columns,
                'F_Score': self.dimension_reducer.scores_,
                'P_Value': self.dimension_reducer.pvalues_
            }).sort_values('F_Score', ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(range(len(selectkbest_scores)), selectkbest_scores['F_Score'],
                           color=COLOR_PALETTE['val'], alpha=0.8, edgecolor='black', linewidth=0.6)
            ax.set_yticks(range(len(selectkbest_scores)))
            ax.set_yticklabels(selectkbest_scores['Feature'], fontsize=8)
            ax.set_xlabel('F-Score (ANOVA)', fontsize=10)
            ax.set_title(
                f'Top 10 SelectKBest Features',
                pad=15, fontsize=11
            )
            ax.grid(axis='x', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for i, (idx, row) in enumerate(selectkbest_scores.iterrows()):
                p_val = row['P_Value']
                p_label = 'p<0.001' if p_val < 0.001 else f'p={p_val:.3f}'
                ax.text(row['F_Score'] + 1, i, p_label, ha='left', va='center', fontsize=7)

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_root,self.figure_dir,
                f'selectkbest_top10_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.png'),
                dpi=600
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'selectkbest_top10_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.pdf'),
                dpi=600,
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'selectkbest_top10_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.svg'),
                dpi=600,
                format='svg',
                bbox_inches='tight'
            )
            plt.close()

    def plot_feature_target_violin_plots(self):
        """优化后的小提琴图（期刊级）：移除显著性横线，仅组上方标注符号，更精致紧凑"""
        logger.info("Generating journal-grade violin plots with statistical significance (no horizontal lines)...")

        violin_dir = os.path.join(self.save_root, self.figure_dir,
                                  f'violin_plots_{self.feature_type_filter}_{self.feature_selection_method}')
        os.makedirs(violin_dir, exist_ok=True)

        top_indices = [self.reduced_feature_names.index(f) for f in self.top_features]
        train_df = self.X_train_imputed_df[self.top_features].copy().reset_index(drop=True)
        train_df['class_label'] = self.y_train

        # 期刊级配色（更柔和专业）
        class_colors = {
            0: '#E64B35',  # AIH - 红色
            1: '#4DBBD5',  # PBC - 蓝色
            2: '#00A087',  # OS - 绿色
            3: '#F18F01'  # CTR - 橙色（对照组）
        }
        class_names = self.class_names
        n_classes = self.n_classes

        # 计算每个类别的样本量
        class_counts = pd.Series(self.y_train).value_counts().sort_index()
        sample_counts = [class_counts.get(i, 0) for i in range(n_classes)]

        for idx, feature in enumerate(tqdm(self.top_features, desc="Journal-grade violin plots")):
            feature_data = train_df[feature].values
            class_labels = train_df['class_label'].values

            # 1. 统计检验（Kruskal-Wallis整体检验 + pairwise Mann-Whitney与对照组比较）
            kw_result = self._perform_kruskal_wallis_test(feature_data, class_labels, class_names)
            pairwise_sig = self._perform_pairwise_mannwhitney(feature_data, class_labels, class_names)

            # 2. 准备数据（过滤空组）
            class_groups = []
            valid_cls_indices = []  # 记录非空组的索引
            for cls_idx in range(n_classes):
                group_data = feature_data[class_labels == cls_idx]
                if len(group_data) > 0:  # 只保留非空组
                    class_groups.append(group_data)
                    valid_cls_indices.append(cls_idx)
                else:
                    logger.warning(f"Feature {feature} - Class {class_names[cls_idx]} has no valid data, skipping")

            # 若所有组都为空，跳过该特征
            if len(class_groups) == 0:
                logger.warning(f"Feature {feature} has no valid data in any class, skipping violin plot")
                continue

            # 3. 创建图表（优化尺寸和布局，更紧凑）
            fig, ax = plt.subplots(figsize=self.figure_params['medium'])

            # 4. 绘制小提琴图（优化细节）
            violin_parts = ax.violinplot(
                class_groups, positions=valid_cls_indices,
                showmeans=True, showmedians=True, showextrema=True,
                widths=0.6,
                bw_method=0.3
            )

            # 5. 优化小提琴图颜色和线条（只处理有效组）
            for i, (cls_idx, (pc, count)) in enumerate(
                    zip(valid_cls_indices, zip(violin_parts['bodies'], sample_counts))):
                pc.set_facecolor(class_colors[cls_idx])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.8)
                pc.set_hatch('///' if cls_idx == 3 else '')  # 对照组添加纹理标记

            # 6. 优化统计标记（均值、中位数、极值）- 存在性检查
            if 'means' in violin_parts:
                vp = violin_parts['means']
                vp.set_marker('*')
                vp.set_markerfacecolor('red')
                vp.set_markersize(8)
                vp.set_markeredgecolor('black')
                vp.set_markeredgewidth(0.5)
            else:
                logger.warning(f"Feature {feature} - No 'means' marker generated (insufficient data)")

            if 'medians' in violin_parts:
                vp = violin_parts['medians']
                vp.set_color('#1E3A8A')
                vp.set_linewidth(2.0)
            else:
                logger.warning(f"Feature {feature} - No 'medians' marker generated (insufficient data)")

            # 极值线优化
            for partname in ('cbars', 'cmins', 'cmaxes'):
                if partname in violin_parts:
                    vp = violin_parts[partname]
                    vp.set_color('black')
                    vp.set_linewidth(0.8)

            # 7. 添加统计显著性标注（核心修改：移除横线，仅在对应组上方标注符号）
            # 计算y轴基准（所有组95百分位的最大值）
            y_max = np.max([np.percentile(group, 95) for group in class_groups if len(group) > 0])
            # 缩小偏移量，让标注更紧凑（从15%改为8%）
            y_offset = y_max * 0.08 if y_max != 0 else 0.2
            sig_y_base = y_max + y_offset  # 显著性符号的基准y坐标

            # 只标注有效组中的病例组（排除对照组）
            for cls_idx in valid_cls_indices:
                if cls_idx == 3:  # 跳过对照组
                    continue
                cls_name = class_names[cls_idx]
                sig_marker = pairwise_sig.get(cls_name, 'ns')
                if sig_marker != 'ns' and 3 in valid_cls_indices:  # 对照组存在才标注
                    # 核心修改：仅在对应组上方标注符号，无横线
                    ax.text(
                        cls_idx,  # 符号x坐标 = 对应病例组的x位置
                        sig_y_base,  # 符号y坐标 = 组95百分位上方
                        sig_marker,
                        ha='center', va='bottom',
                        fontsize=9,  # 缩小字体更精致
                        fontweight='bold',
                        color='black',
                        # 添加轻微背景色，增强可读性（可选）
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none')
                    )

            # 8. 优化标题和标签（更紧凑）
            if np.isnan(kw_result['H_statistic']):
                kw_text = f'Kruskal-Wallis: {kw_result["significance"]}'
            else:
                kw_text = f'Kruskal-Wallis H={kw_result["H_statistic"]:.1f}, p={kw_result["p_value"]:.3f}'

            # 特征名称换行处理（更紧凑）
            feature_name_display = feature.replace('/', '\n').replace('(', '\n(') if len(feature) > 18 else feature
            ax.set_title(
                f'{feature_name_display}\n{kw_text}',
                fontsize=8.5, pad=12, fontweight='bold'  # 缩小字体和内边距
            )

            # 9. 优化坐标轴（更紧凑）
            ax.set_xlabel('Disease Class', fontsize=9, fontweight='bold')
            ax.set_ylabel('Feature Value', fontsize=9, fontweight='bold')
            ax.set_xticks(valid_cls_indices)
            # 横坐标标签更紧凑（缩小字体）
            xtick_labels = [f'{class_names[cls_idx]}\n(n={sample_counts[cls_idx]})' for cls_idx in valid_cls_indices]
            ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=7.5, fontweight='bold')
            ax.tick_params(axis='y', labelsize=7.5)

            # 10. 优化网格和边框（更精致）
            ax.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.4, alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.0)
            ax.spines['left'].set_linewidth(1.0)
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')

            # 11. 调整y轴范围（更紧凑，适配新的标注位置）
            y_min = np.min([np.percentile(group, 5) for group in class_groups if len(group) > 0])
            y_total_range = y_max - y_min
            # 缩小顶部预留空间（从1.5倍偏移改为1.0倍）
            ax.set_ylim(y_min - y_total_range * 0.08, sig_y_base + y_offset * 1.0)

            # 12. 保存图表（优化）
            safe_feature = feature.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_').replace(':',
                                                                                                                 '_')
            safe_feature = safe_feature[:30]
            # save_path =

            plt.tight_layout(pad=0.8)  # 缩小整体内边距
            plt.savefig(
                os.path.join(violin_dir, f'violin_{idx+1:02d}_{safe_feature}_{self.feature_type_filter}.png'),
                dpi=800,
                bbox_inches='tight',
                pad_inches=0.08,  # 更紧凑的保存边距
                facecolor='white'
            )
            plt.savefig(
                os.path.join(violin_dir, f'violin_{idx+1:02d}_{safe_feature}_{self.feature_type_filter}.pdf'),
                dpi=600,
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(violin_dir, f'violin_{idx+1:02d}_{safe_feature}_{self.feature_type_filter}.svg'),
                dpi=600,
                format='svg',
                bbox_inches='tight'
            )
            plt.close()

        # 生成统计汇总表
        kw_summary = []
        for feature in self.top_features:
            feature_data = train_df[feature].values
            class_labels = train_df['class_label'].values
            kw_result = self._perform_kruskal_wallis_test(feature_data, class_labels, class_names)
            pairwise_sig = self._perform_pairwise_mannwhitney(feature_data, class_labels, class_names)

            row = {
                'Feature': feature,
                'H_statistic': kw_result['H_statistic'],
                'p_value': kw_result['p_value'],
                'Overall_Significance': kw_result['significance'],
                'AIH_vs_CTR': pairwise_sig.get('AIH', 'ns'),
                'PBC_vs_CTR': pairwise_sig.get('PBC', 'ns'),
                'OS_vs_CTR': pairwise_sig.get('OS', 'ns'),
                'Dimension_Reduction': self.dimension_reduction_method.upper(),
                'Feature_Selection': self.feature_selection_method.upper(),
                'Feature_Type_Filter': self.feature_type_filter
            }
            kw_summary.append(row)

        kw_summary_df = pd.DataFrame(kw_summary)
        kw_summary_df.to_csv(
            os.path.join(self.save_root, self.results_dir,
                         f'kruskal_wallis_pairwise_summary_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}.csv'),
            index=False, encoding='utf-8-sig'
        )

        logger.info(f"Journal-grade violin plots (compact style) saved to: {violin_dir}")
        logger.info(f"Statistical summary saved to results_3/")

    def plot_dca_curves(self):
        """Decision Curve Analysis (DCA) for clinical utility evaluation"""
        logger.info("Generating DCA curves (clinical utility)...")
        best_model = self.best_model
        top_indices = [self.reduced_feature_names.index(f) for f in self.top_features]
        X_val_top = self.X_val_reduced[:, top_indices]
        y_val_proba = best_model.predict_proba(X_val_top)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        thresholds = np.linspace(0.01, 0.99, 99)

        for class_idx, class_name in enumerate(self.class_names):
            ax = axes[class_idx]
            y_true_binary = (self.y_val == class_idx).astype(int)
            y_pred_prob = y_val_proba[:, class_idx]

            try:
                dca_results = DelongTest.custom_dca_analysis(y_true_binary, y_pred_prob, thresholds)

                ax.plot(dca_results['threshold'], dca_results['model_net_benefit'],
                        color=COLOR_PALETTE[f'class{class_idx+1}'], linewidth=1.5, label='Proposed Model',
                        alpha=0.9)
                ax.plot(dca_results['threshold'], dca_results['treat_all_net_benefit'],
                        color='gray', linestyle='--', linewidth=1.0, label='Treat All', alpha=0.7)
                ax.plot(dca_results['threshold'], dca_results['treat_none_net_benefit'],
                        color='black', linestyle='-', linewidth=1.0, label='Treat None', alpha=0.7)

                ax.set_ylim(-10, 40)
                ax.set_xlim(0, 1)
                ax.set_title(f'DCA Curve - {class_name} (OvR)', fontsize=10, pad=10)
                ax.set_xlabel('Threshold Probability', fontsize=9)
                ax.set_ylabel('Net Benefit (per 100 Patients)', fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(axis='both', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.legend(fontsize=8, loc='best')

            except Exception as e:
                logger.warning(f"DCA curve failed for {class_name}: {str(e)}")
                ax.text(0.5, 0.5, f'DCA Generation Failed\n{class_name}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)

        fig.suptitle(
            f'Decision Curve Analysis (DCA) - {self.best_model_name}\n(Feature Filter: {self.feature_type_filter}, Clinical Utility)',
            fontsize=12, y=0.98
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(self.save_root,self.figure_dir,
            f'dca_curves_{self.best_model_name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'dca_curves_{self.best_model_name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'dca_curves_{self.best_model_name}_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}_gpu.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

        logger.info("DCA curves saved (clinical utility evaluation)")

    def plot_calibration_curves(self):
        """生成最佳模型的校准曲线（Calibration Curve），保存为PNG/TIF/PDF格式"""
        logger.info("Generating calibration curves for best model...")

        from sklearn.calibration import calibration_curve

        # 准备数据
        top_indices = [self.reduced_feature_names.index(f) for f in self.top_features]
        X_val_top = self.X_val_reduced[:, top_indices]
        y_val = self.y_val
        n_classes = self.n_classes
        class_names = self.class_names

        # 子图：各类别校准曲线
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for cls_idx, cls_name in enumerate(class_names):
            ax = axes[cls_idx]
            y_true_binary = (y_val == cls_idx).astype(int)
            y_pred_prob = self.best_model.predict_proba(X_val_top)[:, cls_idx]

            # 计算校准曲线
            prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_prob, n_bins=10, strategy='quantile')

            # 绘制校准曲线
            ax.plot(prob_pred, prob_true, marker='o', linewidth=1.5, markersize=6,
                    color=COLOR_PALETTE[f'class{cls_idx+1}'], label=f'{cls_name}')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, alpha=0.7, label='Perfect Calibration')

            # 计算校准误差（ECE）
            n_actual_bins = len(prob_true)
            ece = np.mean(
                np.abs(prob_true - prob_pred) * np.histogram(y_pred_prob, bins=n_actual_bins, range=(0, 1))[0] / len(y_pred_prob))

            ax.set_xlabel('Predicted Probability', fontsize=9)
            ax.set_ylabel('True Probability (Fraction of Positives)', fontsize=9)
            ax.set_title(f'Calibration Curve - {cls_name}\nECE = {ece:.3f}', fontsize=10, pad=10)
            ax.legend(fontsize=8)
            ax.grid(axis='both', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        fig.suptitle(f'Calibration Curves - Best Model ({self.best_model_name})', fontsize=11, y=0.98)
        plt.tight_layout()

        # 多格式保存
        base_path = os.path.join(self.save_root, self.figure_dir,
                                 f'calibration_curves_{self.best_model_name}_{self.feature_type_filter}')
        plt.savefig(f'{base_path}.png', dpi=600, bbox_inches='tight')
        plt.savefig(
            f'{base_path}.pdf',
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            f'{base_path}.svg',
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

        # 整体校准曲线（多分类平均）
        fig, ax = plt.subplots(figsize=(8, 6))
        per_class_prob_true = []
        per_class_prob_pred = []
        for cls_idx in range(n_classes):
            y_true_binary = (y_val == cls_idx).astype(int)
            y_pred_prob = self.best_model.predict_proba(X_val_top)[:, cls_idx]
            prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_prob, n_bins=10, strategy='quantile')
            per_class_prob_true.append(prob_true)
            per_class_prob_pred.append(prob_pred)

        # 各类 bin 数可能不同，取最短长度对齐后平均用于绘图
        min_len = min(len(a) for a in per_class_prob_true)
        plot_prob_true = np.mean([a[:min_len] for a in per_class_prob_true], axis=0)
        plot_prob_pred = np.mean([a[:min_len] for a in per_class_prob_pred], axis=0)

        ax.plot(plot_prob_pred, plot_prob_true, marker='o', linewidth=1.5, markersize=6,
                color=COLOR_PALETTE['class1'], label='Average (All Classes)')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, alpha=0.7, label='Perfect Calibration')

        # 整体ECE：各类标准加权ECE的均值（与每类子图公式一致）
        per_class_ece = []
        for cls_idx, (pt, pp) in enumerate(zip(per_class_prob_true, per_class_prob_pred)):
            y_pred_prob_cls = self.best_model.predict_proba(X_val_top)[:, cls_idx]
            n_b = len(pt)
            w = np.histogram(y_pred_prob_cls, bins=n_b, range=(0, 1))[0] / len(y_pred_prob_cls)
            per_class_ece.append(np.sum(np.abs(pt - pp) * w))
        overall_ece = float(np.mean(per_class_ece))

        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('True Probability (Fraction of Positives)', fontsize=10)
        ax.set_title(
            f'Overall Calibration Curve - Best Model ({self.best_model_name})\nOverall ECE = {overall_ece:.3f}',
            fontsize=11, pad=15)
        ax.legend(fontsize=9)
        ax.grid(axis='both', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # 多格式保存
        overall_base_path = os.path.join(self.save_root, self.figure_dir,
                                         f'overall_calibration_curve_{self.best_model_name}_{self.feature_type_filter}')
        plt.savefig(f'{overall_base_path}.png', dpi=600, bbox_inches='tight')
        plt.savefig(
            f'{overall_base_path}.pdf',
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            f'{overall_base_path}.svg',
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

        logger.info("Calibration curves saved as PNG/TIF/PDF")

    # def _clinical_guideline_diagnosis(self, X_val_df):
    #     """
    #     基于最新临床指南的PBC/AIH/OS诊断规则（适配缺少部分指标的场景）
    #     缺失指标：影像学排除大胆管梗阻、肝组织学表现、SMA滴度、排除病毒性肝炎
    #     仅保留指标：'AMA-M2', 'AMA', 'ANA', 'Anti-Sp100', 'Anti-Gp210',
    #                 'Anti-LKM-1', 'Anti-SLA/LP', 'ALT', 'AST', 'ALP(35-100)',
    #                 'GGT(4-50)', 'TBIL','IgG', 'IgM'
    #     """
    #     n_samples = len(X_val_df)
    #     # ==================== PBC诊断（适配缺失指标：无影像学/组织学证据） ====================
    #     # 条件1：仅保留胆汁淤积生化证据（无影像学排除大胆管梗阻指标，故仅判断ALP/GGT升高）
    #     alp = X_val_df['ALP(35-100)'].values if 'ALP(35-100)' in X_val_df.columns else np.zeros(n_samples)
    #     ggt = X_val_df['GGT(4-50)'].values if 'GGT(4-50)' in X_val_df.columns else np.zeros(n_samples)
    #     pbc_cond1 = (alp > 100) | (ggt > 50)  # 仅胆汁淤积生化证据
    #
    #     # 条件2：PBC特异性抗体阳性（保留，指标存在）
    #     ama = X_val_df.get('AMA', np.zeros(n_samples)).values
    #     ama_m2 = X_val_df.get('AMA-M2', np.zeros(n_samples)).values
    #     anti_sp100 = X_val_df.get('Anti-Sp100', np.zeros(n_samples)).values
    #     anti_gp210 = X_val_df.get('Anti-Gp210', np.zeros(n_samples)).values
    #     pbc_cond2 = (ama == 1) | (ama_m2 == 1) | (anti_sp100 == 1) | (anti_gp210 == 1)
    #
    #     # 条件3：组织学证据（无该指标，设为全False）
    #     pbc_cond3 = np.zeros(n_samples, dtype=bool)  # 无肝组织学表现数据
    #
    #     # 适配缺失指标：原3条满足2条 → 现仅2个有效条件，满足≥1条则诊断PBC（标签1）
    #     pbc_cond_count = np.sum([pbc_cond1, pbc_cond2, pbc_cond3], axis=0)
    #     pbc_clinical = np.where(pbc_cond_count >= 1, 1, 0)  # 调整诊断阈值
    #
    #     # ==================== AIH诊断（适配缺失指标：无SMA滴度/肝组织学/排除病毒性肝炎） ====================
    #     aih_score = np.zeros(n_samples)
    #     # 1. 仅保留ANA滴度（无SMA滴度指标，故仅判断ANA）
    #     ana_titer = X_val_df.get('ANA', np.zeros(n_samples)).values
    #     aih_score += np.where(ana_titer == 1, 2, 0)
    #
    #     # 2. LKM-1滴度≥1:40得2分（保留，指标存在）
    #     lkm1_titer = X_val_df.get('Anti-LKM-1', np.zeros(n_samples)).values
    #     aih_score += np.where(lkm1_titer == 1, 2, 0)
    #
    #     # 3. SLA（Anti-SLA/LP）阳性得2分（保留，指标存在）
    #     sla_pos = X_val_df.get('Anti-SLA/LP', np.zeros(n_samples)).values
    #     aih_score += np.where(sla_pos == 1, 2, 0)
    #
    #     # 4. IgG升高（>1.1倍上限得2分，>上限得1分；参考值16/17.6为示例，可根据实际调整）
    #     igg = X_val_df.get('IgG', np.zeros(n_samples)).values
    #     aih_score += np.where(igg > 17.6, 2, np.where(igg > 16, 1, 0))
    #
    #     # 移除：肝组织学评分（无该指标）
    #     # 移除：排除病毒性肝炎评分（无该指标）
    #
    #     # 适配缺失指标：原评分阈值≥6/≥7 → 现评分项减少，调整为≥4分可能AIH，≥5分确诊AIH（标签0）
    #     aih_clinical = np.full(n_samples, np.nan)
    #     aih_clinical = np.where(aih_score >= 4, 0, aih_clinical)  # ≥4分：AIH可能
    #     aih_clinical = np.where(aih_score >= 5, 0, aih_clinical)  # ≥5分：AIH确诊
    #
    #     # ==================== OS诊断（同时满足PBC+AIH） ====================
    #     os_clinical = np.where((pbc_clinical == 1) & (~np.isnan(aih_clinical)), 2, 0)
    #
    #     # ==================== 整合诊断结果（优先级：OS > PBC > AIH > CTR） ====================
    #     clinical_pred = np.full(n_samples, 3)  # 默认CTR（标签3）
    #     clinical_pred[~np.isnan(aih_clinical)] = 0  # 填充AIH
    #     clinical_pred[pbc_clinical == 1] = 1  # 填充PBC（覆盖AIH）
    #     clinical_pred[os_clinical == 2] = 2  # 填充OS（覆盖PBC/AIH）
    #
    #     return clinical_pred, {
    #         'PBC': pbc_clinical,
    #         'AIH': aih_clinical,
    #         'OS': os_clinical
    #     }

    def _clinical_guideline_diagnosis(self, X_val_df):
        """
        基于最新临床指南的PBC/AIH/OS诊断规则（适配缺少部分指标的场景）
        缺失指标：影像学排除大胆管梗阻、肝组织学表现、SMA滴度、排除病毒性肝炎
        仅保留指标：'AMA-M2', 'AMA', 'ANA', 'Anti-Sp100', 'Anti-Gp210',
                    'Anti-LKM-1', 'Anti-SLA/LP', 'ALT', 'AST', 'ALP(35-100)',
                    'GGT(4-50)', 'TBIL','IgG', 'IgM'
        """
        n_samples = len(X_val_df)

        # ==================== 辅助函数：安全获取列值 ====================
        def _get_col(df, col_name):
            """安全获取DataFrame列，返回numpy array"""
            if col_name in df.columns:
                return np.asarray(df[col_name])
            else:
                return np.zeros(n_samples)

        # =================================================================

        # ==================== PBC诊断（适配缺失指标：无影像学/组织学证据） ====================
        # 条件1：仅保留胆汁淤积生化证据（无影像学排除大胆管梗阻指标，故仅判断ALP/GGT升高）
        alp = _get_col(X_val_df, 'ALP(35-100)')
        ggt = _get_col(X_val_df, 'GGT(4-50)')
        pbc_cond1 = (alp > 100) | (ggt > 50)  # 仅胆汁淤积生化证据

        # 条件2：PBC特异性抗体阳性（保留，指标存在）
        ama = _get_col(X_val_df, 'AMA')
        ama_m2 = _get_col(X_val_df, 'AMA-M2')
        anti_sp100 = _get_col(X_val_df, 'Anti-Sp100')
        anti_gp210 = _get_col(X_val_df, 'Anti-Gp210')
        pbc_cond2 = (ama == 1) | (ama_m2 == 1) | (anti_sp100 == 1) | (anti_gp210 == 1)

        # 条件3：组织学证据（无该指标，设为全False）
        pbc_cond3 = np.zeros(n_samples, dtype=bool)  # 无肝组织学表现数据

        # 适配缺失指标：原3条满足2条 → 现仅2个有效条件，满足≥1条则诊断PBC（标签1）
        pbc_cond_count = np.sum([pbc_cond1, pbc_cond2, pbc_cond3], axis=0)
        pbc_clinical = np.where(pbc_cond_count >= 1, 1, 0)  # 调整诊断阈值

        # ==================== AIH诊断（适配缺失指标：无SMA滴度/肝组织学/排除病毒性肝炎） ====================
        aih_score = np.zeros(n_samples)
        # 1. 仅保留ANA滴度（无SMA滴度指标，故仅判断ANA）
        ana_titer = _get_col(X_val_df, 'ANA')
        aih_score += np.where(ana_titer == 1, 2, 0)

        # 2. LKM-1滴度≥1:40得2分（保留，指标存在）
        lkm1_titer = _get_col(X_val_df, 'Anti-LKM-1')
        aih_score += np.where(lkm1_titer == 1, 2, 0)

        # 3. SLA（Anti-SLA/LP）阳性得2分（保留，指标存在）
        sla_pos = _get_col(X_val_df, 'Anti-SLA/LP')
        aih_score += np.where(sla_pos == 1, 2, 0)

        # 4. IgG升高（>1.1倍上限得2分，>上限得1分；参考值16/17.6为示例，可根据实际调整）
        igg = _get_col(X_val_df, 'IgG')
        aih_score += np.where(igg > 17.6, 2, np.where(igg > 16, 1, 0))

        # 移除：肝组织学评分（无该指标）
        # 移除：排除病毒性肝炎评分（无该指标）

        # 适配缺失指标：原评分阈值≥6/≥7 → 现评分项减少，调整为≥4分可能AIH，≥5分确诊AIH（标签0）
        aih_clinical = np.full(n_samples, np.nan)
        aih_clinical = np.where(aih_score >= 4, 0, aih_clinical)  # ≥4分：AIH可能
        aih_clinical = np.where(aih_score >= 5, 0, aih_clinical)  # ≥5分：AIH确诊

        # ==================== OS诊断（同时满足PBC+AIH） ====================
        os_clinical = np.where((pbc_clinical == 1) & (~np.isnan(aih_clinical)), 2, 0)

        # ==================== 整合诊断结果（优先级：OS > PBC > AIH > CTR） ====================
        clinical_pred = np.full(n_samples, 3)  # 默认CTR（标签3）
        clinical_pred[~np.isnan(aih_clinical)] = 0  # 填充AIH
        clinical_pred[pbc_clinical == 1] = 1  # 填充PBC（覆盖AIH）
        clinical_pred[os_clinical == 2] = 2  # 填充OS（覆盖PBC/AIH）

        return clinical_pred, {
            'PBC': pbc_clinical,
            'AIH': aih_clinical,
            'OS': os_clinical
        }

    def _calc_clinical_ml_metrics(self, y_true, clinical_pred, ml_pred, ml_proba):
        """计算临床指南和ML模型的性能指标（含灵敏度、特异度、F1、Delong test，新增临床AUC的95%CI）"""
        from sklearn.metrics import recall_score, precision_score, f1_score
        metrics = {}
        class_names = self.class_names
        n_classes = self.n_classes

        # 临床概率转换为one-hot形式
        clinical_proba = np.zeros_like(ml_proba)
        for i in range(len(clinical_pred)):
            cls_idx = int(clinical_pred[i])
            if 0 <= cls_idx < n_classes:
                clinical_proba[i, cls_idx] = 1.0

        # 整体多分类指标
        overall_ml_auc = roc_auc_macro(y_true, ml_proba)
        overall_clinical_auc = roc_auc_macro(y_true, clinical_proba)

        # 新增：计算临床指南AUC的95%置信区间
        clinical_auc_ci_lower, clinical_auc_ci_upper = calculate_clinical_auc_ci(y_true, clinical_proba)
        ml_auc_ci_lower, ml_auc_ci_upper = calculate_auc_ci(y_true, ml_proba)

        metrics['Overall'] = {
            'ML_AUC': overall_ml_auc,
            'Clinical_AUC': overall_clinical_auc,
            'ML_AUC_CI': (ml_auc_ci_lower, ml_auc_ci_upper),  # ML的CI
            'Clinical_AUC_CI': (clinical_auc_ci_lower, clinical_auc_ci_upper),  # 临床的CI
            'ML_Accuracy': np.mean(ml_pred == y_true),
            'Clinical_Accuracy': np.mean(clinical_pred == y_true),
            'ML_F1': f1_score(y_true, ml_pred, average='weighted'),
            'Clinical_F1': f1_score(y_true, clinical_pred, average='weighted'),
            'AUC_P_Value': np.nan
        }

        # 各类别二分类指标（含Delong test）
        for cls_idx, cls_name in enumerate(class_names):
            y_true_binary = (y_true == cls_idx).astype(int)
            ml_pred_binary = (ml_pred == cls_idx).astype(int)
            clinical_pred_binary = (clinical_pred == cls_idx).astype(int)
            ml_proba_bin = ml_proba[:, cls_idx]
            clinical_proba_bin = clinical_proba[:, cls_idx]

            # 基础指标
            ml_sens = recall_score(y_true_binary, ml_pred_binary, zero_division=0)  # 灵敏度
            clinical_sens = recall_score(y_true_binary, clinical_pred_binary, zero_division=0)
            ml_spec = recall_score(y_true_binary, ml_pred_binary, pos_label=0, zero_division=0)  # 特异度
            clinical_spec = recall_score(y_true_binary, clinical_pred_binary, pos_label=0, zero_division=0)

            # Delong test（AUC差异）
            auc_p = np.nan
            ml_cls_auc = np.nan
            clinical_cls_auc = np.nan
            ml_cls_ci = (np.nan, np.nan)
            clinical_cls_ci = (np.nan, np.nan)

            if len(np.unique(y_true_binary)) > 1:
                try:
                    # 计算各类别ML和临床的AUC
                    ml_cls_auc = roc_auc_score(y_true_binary, ml_proba_bin)
                    clinical_cls_auc = roc_auc_score(y_true_binary, clinical_proba_bin)
                    # 计算各类别AUC的CI
                    ml_cls_ci = calculate_auc_ci(y_true_binary, ml_proba_bin)
                    clinical_cls_ci = calculate_auc_ci(y_true_binary, clinical_proba_bin)
                    # Delong test
                    auc_p, _, _ = self.delong_test.compare(y_true_binary, ml_proba_bin, clinical_proba_bin)
                except:
                    auc_p = np.nan

            metrics[cls_name] = {
                'ML_Accuracy': np.mean(ml_pred_binary == y_true_binary),
                'Clinical_Accuracy': np.mean(clinical_pred_binary == y_true_binary),
                'ML_Sensitivity': ml_sens,
                'Clinical_Sensitivity': clinical_sens,
                'ML_Specificity': ml_spec,
                'Clinical_Specificity': clinical_spec,
                'ML_F1': f1_score(y_true_binary, ml_pred_binary, zero_division=0),
                'Clinical_F1': f1_score(y_true_binary, clinical_pred_binary, zero_division=0),
                'ML_AUC': ml_cls_auc,
                'Clinical_AUC': clinical_cls_auc,
                'ML_AUC_CI': ml_cls_ci,  # 新增：ML类别AUC的CI
                'Clinical_AUC_CI': clinical_cls_ci,  # 新增：临床类别AUC的CI
                'AUC_P_Value': auc_p  # Delong test的p值
            }

        # 新增：整体Delong test（ML vs 临床）
        try:
            # 转换为二分类适配Delong test
            ml_max_proba = np.max(ml_proba, axis=1)
            clinical_max_proba = np.max(clinical_proba, axis=1)
            y_true_binary_all = np.ones_like(y_true)
            overall_p_value, _, _ = self.delong_test.compare(y_true_binary_all, ml_max_proba, clinical_max_proba)
            metrics['Overall']['AUC_P_Value'] = overall_p_value
        except Exception as e:
            logger.warning(f"Failed to calculate overall Delong test: {e}")
            metrics['Overall']['AUC_P_Value'] = np.nan

        return metrics, clinical_proba

    def plot_clinical_vs_ml_performance(self):
        """生成ML模型 vs 临床指南诊断性能对比图"""
        logger.info("Generating ML vs Clinical Guideline performance comparison plot...")

        # 准备数据
        top_indices = [self.reduced_feature_names.index(f) for f in self.top_features]
        X_val_top = self.X_val_reduced[:, top_indices]
        ml_pred = self.best_model.predict(X_val_top)
        ml_proba = self.best_model.predict_proba(X_val_top)

        # 原始验证集特征数据（用于临床诊断）
        X_val_raw = self.val_data[self.feature_columns]
        clinical_pred, clinical_binary = self._clinical_guideline_diagnosis(X_val_raw)
        metrics, clinical_proba = self._calc_clinical_ml_metrics(self.y_val, clinical_pred, ml_pred, ml_proba)

        # ==================== 第1个图：整体性能对比（AUC+准确率）====================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # AUC对比柱状图
        models = ['Clinical Guideline', f'ML ({self.best_model_name})']
        aucs = [metrics['Overall']['Clinical_AUC'], metrics['Overall']['ML_AUC']]
        accs = [metrics['Overall']['Clinical_Accuracy'], metrics['Overall']['ML_Accuracy']]

        bars1 = ax1.bar(models, aucs, width=0.6,
                        color=[COLOR_PALETTE['clinical'], COLOR_PALETTE['val']],
                        alpha=0.8, edgecolor='black', linewidth=0.6)
        ax1.set_ylabel('AUC (One-vs-Rest)', fontsize=9)
        ax1.set_title('Overall Diagnostic Performance (AUC)', fontsize=10, pad=10)
        ax1.set_ylim(0.5, 1.0)
        ax1.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        for bar, auc in zip(bars1, aucs):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{auc:.3f}', ha='center', va='bottom', fontsize=8)

        # 各类别准确率对比
        cls_names = ['AIH', 'PBC', 'OS', 'CTR']
        ml_accs = [metrics[cls]['ML_Accuracy'] for cls in cls_names]
        clinical_accs = [metrics[cls]['Clinical_Accuracy'] for cls in cls_names]

        x = np.arange(len(cls_names))
        width = 0.35
        bars2 = ax2.bar(x - width / 2, clinical_accs, width, label='Clinical Guideline',
                        color=COLOR_PALETTE['clinical'], alpha=0.8, edgecolor='black', linewidth=0.6)
        bars3 = ax2.bar(x + width / 2, ml_accs, width, label=f'ML ({self.best_model_name})',
                        color=COLOR_PALETTE['val'], alpha=0.8, edgecolor='black', linewidth=0.6)

        ax2.set_xlabel('Disease Class', fontsize=9)
        ax2.set_ylabel('Accuracy', fontsize=9)
        ax2.set_title('Class-specific Diagnostic Accuracy', fontsize=10, pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(cls_names, fontsize=8)
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        for bars in [bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=7)

        fig.suptitle(
            f'ML Model vs Clinical Guideline Diagnostic Performance (Validation Set)\nFeature Filter: {self.feature_type_filter}',
            fontsize=11, y=0.98)
        plt.tight_layout()

        # 保存第1个图（PNG/SVG/PDF）
        base_path1 = os.path.join(self.save_root, self.figure_dir,
                                  f'clinical_vs_ml_performance_{self.best_model_name}_{self.feature_type_filter}')
        fig.savefig(f'{base_path1}.png', dpi=600, bbox_inches='tight')
        fig.savefig(f'{base_path1}.svg', dpi=600, format='svg', bbox_inches='tight')
        fig.savefig(f'{base_path1}.pdf', dpi=600, format='pdf', bbox_inches='tight')
        plt.close(fig)  # 显式关闭

        # ==================== 第2个图：各类别ROC对比 ====================
        fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
        target_cls = [0, 1, 2]  # AIH, PBC, OS
        cls_labels = ['AIH', 'PBC', 'OS']
        colors = [COLOR_PALETTE['class1'], COLOR_PALETTE['class2'], COLOR_PALETTE['class3']]

        for idx, (cls_idx, cls_name) in enumerate(zip(target_cls, cls_labels)):
            ax = axes[idx]
            # ML模型ROC
            fpr_ml, tpr_ml, _ = roc_curve(self.y_val == cls_idx, ml_proba[:, cls_idx])
            auc_ml = roc_auc_score(self.y_val == cls_idx, ml_proba[:, cls_idx])
            # 临床指南ROC
            fpr_clin, tpr_clin, _ = roc_curve(self.y_val == cls_idx, clinical_proba[:, cls_idx])
            auc_clin = roc_auc_score(self.y_val == cls_idx, clinical_proba[:, cls_idx])

            ax.plot(fpr_ml, tpr_ml, color=colors[idx], lw=1.2,
                    label=f'ML (AUC={auc_ml:.3f})', alpha=0.9)
            ax.plot(fpr_clin, tpr_clin, color='gray', lw=1.2, linestyle='--',
                    label=f'Clinical (AUC={auc_clin:.3f})', alpha=0.9)
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.7)

            ax.set_xlabel('False Positive Rate', fontsize=8)
            ax.set_ylabel('True Positive Rate', fontsize=8)
            ax.set_title(f'ROC Curve - {cls_name}', fontsize=9, pad=8)
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.01])
            ax.legend(fontsize=7, loc='lower right')
            ax.grid(axis='both', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig2.suptitle(f'ROC Comparison: ML vs Clinical Guideline (Validation Set)',
                      fontsize=11, y=0.98)
        plt.tight_layout()

        # 保存第2个图（PNG/SVG/PDF）
        base_path2 = os.path.join(self.save_root, self.figure_dir,
                                  f'clinical_vs_ml_roc_{self.best_model_name}_{self.feature_type_filter}')
        fig2.savefig(f'{base_path2}.png', dpi=600, bbox_inches='tight')
        fig2.savefig(f'{base_path2}.svg', dpi=600, format='svg', bbox_inches='tight')
        fig2.savefig(f'{base_path2}.pdf', dpi=600, format='pdf', bbox_inches='tight')
        plt.close(fig2)  # 显式关闭

        # ==================== 保存性能指标CSV ====================
        metrics_df = pd.DataFrame({
            'Class': ['AIH', 'PBC', 'OS', 'CTR', 'Overall'],
            'Clinical_Accuracy': [metrics[cls]['Clinical_Accuracy'] for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [
                metrics['Overall']['Clinical_Accuracy']],
            'ML_Accuracy': [metrics[cls]['ML_Accuracy'] for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [
                metrics['Overall']['ML_Accuracy']],
            'Clinical_Sensitivity': [metrics[cls]['Clinical_Sensitivity'] for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [
                np.nan],
            'ML_Sensitivity': [metrics[cls]['ML_Sensitivity'] for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [np.nan],
            'Clinical_Specificity': [metrics[cls]['Clinical_Specificity'] for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [
                np.nan],
            'ML_Specificity': [metrics[cls]['ML_Specificity'] for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [np.nan],
            'Clinical_F1': [metrics[cls]['Clinical_F1'] for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [
                metrics['Overall']['Clinical_F1']],
            'ML_F1': [metrics[cls]['ML_F1'] for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [metrics['Overall']['ML_F1']],
            'Clinical_AUC': [metrics[cls]['Clinical_AUC'] for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [
                metrics['Overall']['Clinical_AUC']],
            'ML_AUC': [metrics[cls]['ML_AUC'] for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [metrics['Overall']['ML_AUC']],
            'Clinical_AUC_95CI': [f"{metrics[cls]['Clinical_AUC_CI'][0]:.3f}-{metrics[cls]['Clinical_AUC_CI'][1]:.3f}"
                                  for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [
                                     f"{metrics['Overall']['Clinical_AUC_CI'][0]:.3f}-{metrics['Overall']['Clinical_AUC_CI'][1]:.3f}"],
            'ML_AUC_95CI': [f"{metrics[cls]['ML_AUC_CI'][0]:.3f}-{metrics[cls]['ML_AUC_CI'][1]:.3f}"
                            for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [
                               f"{metrics['Overall']['ML_AUC_CI'][0]:.3f}-{metrics['Overall']['ML_AUC_CI'][1]:.3f}"],
            'AUC_P_Value(ML_vs_Clinical)': [metrics[cls]['AUC_P_Value'] for cls in ['AIH', 'PBC', 'OS', 'CTR']] + [
                metrics['Overall']['AUC_P_Value']]
        })
        metrics_df.to_csv(os.path.join(self.save_root, self.results_dir,
                                       f'clinical_vs_ml_metrics_{self.best_model_name}_{self.feature_type_filter}.csv'),
                          index=False, encoding='utf-8-sig')

    def plot_pbc_ama_m2_stratified_analysis(self):
        """PBC患者按AMA-M2分层分析（验证集），生成诊断结果散点图（纵轴为PCA主成分）"""
        logger.info("Generating PBC patients stratified analysis by AMA-M2 status (PCA axis)...")

        # 筛选验证集中的PBC患者
        pbc_mask = self.y_val == 1  # PBC标签为1
        pbc_count = np.sum(pbc_mask)
        logger.info(f"Number of PBC patients in validation set: {pbc_count}")
        if pbc_count == 0:
            logger.warning("No PBC patients in validation set, skip AMA-M2 stratified analysis")
            return

        # 提取PBC患者的标准化特征（用于PCA）
        X_val_pbc_scaled = self.X_val_scaled[pbc_mask, :]  # 分步索引：先筛行，避免广播问题

        # 3. 检查并提取AMA-M2特征（增加特征存在性强校验）
        if 'AMA-M2' not in self.feature_columns:
            logger.error("'AMA-M2' feature not found in feature columns, cannot stratify!")
            # 绘制“无AMA-M2特征”的提示图
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Error: 'AMA-M2' feature is not included in the analysis",
                    ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f'PBC Diagnosis Prediction vs PC1\n(No AMA-M2 Feature Available)', fontsize=10, pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_root, self.figure_dir,
                                     f'pbc_diagnosis_scatter_PCA_{self.feature_type_filter}.png'),
                        dpi=600, bbox_inches='tight')
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                                     f'pbc_diagnosis_scatter_PCA_{self.feature_type_filter}.pdf'),
                dpi=600,
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                                     f'pbc_diagnosis_scatter_PCA_{self.feature_type_filter}.svg'),
                dpi=600,
                format='svg',
                bbox_inches='tight'
            )
            plt.close()
            return

        # 3. 提取PBC患者的AMA-M2特征（重点：先排查数据分布）
        ama_m2_col = self.feature_columns.index('AMA-M2')
        # 提取原始AMA-M2值（未标准化的原始值更易判断正负）
        ama_m2_raw = self.val_data['AMA-M2'].values[pbc_mask]  # 改用原始数据，避免标准化后值被缩放
        ama_m2_scaled = self.X_val_scaled[pbc_mask, ama_m2_col]  # 标准化后的值（备用）

        # ========== 关键修复：打印AMA-M2数据分布，排查取值问题 ==========
        logger.info(f"AMA-M2原始值（PBC患者）- 唯一值: {np.unique(ama_m2_raw)}")
        logger.info(f"AMA-M2原始值缺失值数量: {np.sum(np.isnan(ama_m2_raw))}")
        logger.info(f"AMA-M2原始值非数值数量: {np.sum(~np.isnumeric(ama_m2_raw)) if ama_m2_raw.dtype == 'object' else 0}")
        logger.info(f"AMA-M2标准化值（PBC患者）- 范围: [{np.min(ama_m2_scaled)}, {np.max(ama_m2_scaled)}]")

        # 4. 修复AMA-M2正负判断逻辑（兼容多种取值情况）
        # 步骤1：处理缺失值
        ama_m2_clean = np.copy(ama_m2_raw)
        nan_mask = np.isnan(ama_m2_clean)
        logger.info(f"AMA-M2缺失值（PBC患者）: {np.sum(nan_mask)} 例")

        # 步骤2：灵活判断正负（优先用原始值，支持连续值/分类值）
        # 规则：① 分类值：1=阳性，0=阴性；② 连续值：>0=阳性，≤0=阴性；③ 缺失值单独标记
        if np.issubdtype(ama_m2_clean.dtype, np.number):
            # 数值型：按阈值判断（适配连续值/二值编码）
            ama_pos_mask = (ama_m2_clean > 0) & (~nan_mask)
            ama_neg_mask = (ama_m2_clean <= 0) & (~nan_mask)
        else:
            # 字符串型（如"Positive"/"Negative"）：按文本判断
            ama_pos_mask = (ama_m2_clean == 'Positive') | (ama_m2_clean == '阳性') | (ama_m2_clean == '1')
            ama_neg_mask = (ama_m2_clean == 'Negative') | (ama_m2_clean == '阴性') | (ama_m2_clean == '0')

        # 统计各类样本数
        pos_count = np.sum(ama_pos_mask)
        neg_count = np.sum(ama_neg_mask)
        nan_count = np.sum(nan_mask)
        other_count = pbc_count - pos_count - neg_count - nan_count  # 其他异常值
        logger.info(f"AMA-M2分组统计（PBC患者）：")
        logger.info(f"  阳性: {pos_count} | 阴性: {neg_count} | 缺失值: {nan_count} | 其他值: {other_count}")

        # ========== 1. PCA降维（计算主成分及方差解释率） ==========
        pca = PCA(n_components=2)
        X_pbc_pca = pca.fit_transform(X_val_pbc_scaled)
        pca_var_ratio = pca.explained_variance_ratio_
        pc1_var = round(pca_var_ratio[0] * 100, 2)

        # ========== 2. 计算PBC预测概率（横轴） ==========
        # 检查top_features和top_indices有效性
        if not hasattr(self, 'top_features') or len(self.top_features) == 0:
            logger.warning("No top features selected, use all reduced features")
            top_indices = np.arange(self.X_val_reduced.shape[1])
        else:
            # 确保top_indices在列范围内
            top_indices = [self.reduced_feature_names.index(f) for f in self.top_features
                           if f in self.reduced_feature_names]
            if len(top_indices) == 0:
                logger.warning("No valid top features found, use all reduced features")
                top_indices = np.arange(self.X_val_reduced.shape[1])

        # 分步索引：先筛行，再筛列（核心修复：避免广播不匹配）
        X_val_reduced_pbc = self.X_val_reduced[pbc_mask, :]  # 先筛选PBC患者的行
        if len(X_val_reduced_pbc) == 0:
            logger.warning("No PBC patients in validation set after filtering, skip prediction")
            return
        X_val_top_pbc = X_val_reduced_pbc[:, top_indices]  # 再筛选top特征列

        # 计算PBC预测概率
        pbc_proba = self.best_model.predict_proba(X_val_top_pbc)[:, 1]  # 模型对PBC的预测概率

        # ========== 3. 诊断结果散点图（纵轴=PC1，标注方差解释率） ==========
        fig, ax = plt.subplots(figsize=(8, 6))

        # 纵轴：PCA主成分1（PC1），标注PC1 (10.83%)
        pc1_vals = X_pbc_pca[:, 0]

        # 绘制散点（原有逻辑保留，但补充无样本提示）
        has_points = False
        if np.any(ama_pos_mask):
            ax.scatter(
                pbc_proba[ama_pos_mask], pc1_vals[ama_pos_mask],
                color=COLOR_PALETTE['AMA-M2_1'], label='AMA-M2 Positive',
                alpha=0.8, s=60, edgecolors='black', linewidth=0.5
            )
            has_points = True
        if np.any(ama_neg_mask):
            ax.scatter(
                pbc_proba[ama_neg_mask], pc1_vals[ama_neg_mask],
                color=COLOR_PALETTE['AMA-M2_0'], label='AMA-M2 Negative',
                alpha=0.8, s=60, edgecolors='black', linewidth=0.5, marker='s'
            )
            has_points = True

        # 若无任何散点，添加提示文本
        if not has_points:
            ax.text(
                0.5, 0.5, "No valid samples in AMA-M2 Positive/Negative groups",
                ha='center', va='center', fontsize=10, color='gray', style='italic'
            )

        # 添加诊断阈值参考线（预测概率≥0.5为PBC）
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7,
                   label='PBC Diagnosis Threshold (0.5)')

        # 图表样式（重点：纵轴标注PC1 + 方差解释率）
        ax.set_xlabel('Predicted Probability for PBC (ML Model)', fontsize=9)
        ax.set_ylabel(f'PC1 ({pc1_var}%)', fontsize=9)  # 核心：标注PC1 (10.83%)
        ax.set_title(
            f'PBC Diagnosis Prediction vs PC1 ({pc1_var}%)\nStratified by AMA-M2 Status (Validation Set)',
            fontsize=10, pad=15
        )
        ax.legend(fontsize=8, loc='best')
        ax.grid(axis='both', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 样本量+PCA方差解释率标注
        pos_count = np.sum(ama_pos_mask)
        neg_count = np.sum(ama_neg_mask)
        ax.text(
            0.05, 0.95,
            f'AMA-M2 Positive: {pos_count}\nAMA-M2 Negative: {neg_count}\nPC1 Variance: {pc1_var}%',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

        plt.tight_layout()
        # 保存图表（文件名体现PCA）
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'pbc_diagnosis_scatter_PCA_{self.feature_type_filter}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'pbc_diagnosis_scatter_PCA_{self.feature_type_filter}.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'pbc_diagnosis_scatter_PCA_{self.feature_type_filter}.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

        # ========== 4. 保留原箱线图+分层诊断性能逻辑（无需修改） ==========
        # （若需要箱线图也用PCA维度，可自行替换，此处保留原临床特征逻辑）
        key_features = ['ALP(35-100)', 'PE(16:0/18:1)', 'Glycoursodeoxycholic acid (GUDCA)', 'IgM']
        key_features = [f for f in key_features if f in self.feature_columns]
        if len(key_features) >= 2:
            n_feats = len(key_features)
            fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 5))
            if n_feats == 1:
                axes = [axes]

            for idx, feat in enumerate(key_features):
                ax = axes[idx]
                feat_idx = self.feature_columns.index(feat)
                # 分步索引避免广播问题
                feat_vals_pos = X_val_pbc_scaled[ama_pos_mask, feat_idx] if np.any(ama_pos_mask) else np.array([])
                feat_vals_neg = X_val_pbc_scaled[ama_neg_mask, feat_idx] if np.any(ama_neg_mask) else np.array([])

                data = [feat_vals_pos, feat_vals_neg]
                box_plot = ax.boxplot(data, labels=['AMA-M2 (+)', 'AMA-M2 (-)'], patch_artist=True,
                                      boxprops=dict(alpha=0.7), medianprops=dict(color='darkred', linewidth=1.5))
                box_plot['boxes'][0].set_facecolor(COLOR_PALETTE['class2'])
                if len(box_plot['boxes']) > 1:
                    box_plot['boxes'][1].set_facecolor(COLOR_PALETTE['class3'])

                for i, d in enumerate(data):
                    if len(d) > 0:
                        x = np.random.normal(i + 1, 0.08, size=len(d))
                        ax.scatter(x, d, color='black', alpha=0.6, s=15, zorder=3)

                ax.set_xlabel('AMA-M2 Status', fontsize=8)
                ax.set_ylabel(f'{feat} (Standardized)', fontsize=8)
                ax.set_title(feat, fontsize=9, pad=8)
                ax.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(labelsize=7)

            fig.suptitle(f'PBC Patients Key Features: AMA-M2 Stratified Analysis (Validation Set)',
                         fontsize=11, y=0.98)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'pbc_ama_m2_boxplot_{self.feature_type_filter}.png'),
                dpi=600, bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'pbc_ama_m2_boxplot_{self.feature_type_filter}.pdf'),
                dpi=600,
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'pbc_ama_m2_boxplot_{self.feature_type_filter}.svg'),
                dpi=600,
                format='svg',
                bbox_inches='tight'
            )
            plt.close()

        # 分层诊断性能计算
        pos_auc = np.nan
        neg_auc = np.nan
        if np.any(ama_pos_mask):
            try:
                pos_auc = roc_auc_score(np.ones_like(pbc_proba[ama_pos_mask]), pbc_proba[ama_pos_mask])
            except Exception as e:
                logger.warning(f"Failed to calculate AUC for AMA-M2 Positive group: {e}")
                pos_auc = np.nan
        if np.any(ama_neg_mask):
            try:
                neg_auc = roc_auc_score(np.ones_like(pbc_proba[ama_neg_mask]), pbc_proba[ama_neg_mask])
            except Exception as e:
                logger.warning(f"Failed to calculate AUC for AMA-M2 Negative group: {e}")
                neg_auc = np.nan

        # 新增：获取PBC患者的临床指南预测结果
        X_val_raw_pbc = self.val_data[self.feature_columns][pbc_mask]
        clinical_pred_pbc, _ = self._clinical_guideline_diagnosis(X_val_raw_pbc)
        # 临床对PBC的预测概率（one-hot取PBC类）
        clinical_proba_pbc = np.zeros((len(clinical_pred_pbc), self.n_classes))
        for i in range(len(clinical_pred_pbc)):
            cls_idx = int(clinical_pred_pbc[i])
            if 0 <= cls_idx < self.n_classes:
                clinical_proba_pbc[i, cls_idx] = 1.0
        clinical_proba_pbc_pbc = clinical_proba_pbc[:, 1]  # 取PBC类的概率

        # 新增：分组指标计算函数
        def calculate_group_metrics(y_true, ml_pred, clinical_pred, ml_proba, clinical_proba):
            y_true_bin = (y_true == 1).astype(int)
            ml_pred_bin = (ml_pred == 1).astype(int)
            clinical_pred_bin = (clinical_pred == 1).astype(int)

            # 计算指标
            ml_sens = recall_score(y_true_bin, ml_pred_bin, zero_division=0)
            clinical_sens = recall_score(y_true_bin, clinical_pred_bin, zero_division=0)
            ml_spec = recall_score(y_true_bin, ml_pred_bin, pos_label=0, zero_division=0)
            clinical_spec = recall_score(y_true_bin, clinical_pred_bin, pos_label=0, zero_division=0)

            # Delong test
            auc_p = np.nan
            if len(np.unique(y_true_bin)) > 1:
                try:
                    auc_p, _, _ = self.delong_test.compare(y_true_bin, ml_proba, clinical_proba)
                except:
                    auc_p = np.nan

            return {
                'ML_Accuracy': np.mean(ml_pred_bin == y_true_bin),
                'Clinical_Accuracy': np.mean(clinical_pred_bin == y_true_bin),
                'ML_Sensitivity': ml_sens,
                'Clinical_Sensitivity': clinical_sens,
                'ML_Specificity': ml_spec,
                'Clinical_Specificity': clinical_spec,
                'ML_F1': f1_score(y_true_bin, ml_pred_bin, zero_division=0),
                'Clinical_F1': f1_score(y_true_bin, clinical_pred_bin, zero_division=0),
                'ML_AUC': roc_auc_score(y_true_bin, ml_proba) if len(np.unique(y_true_bin)) > 1 else np.nan,
                'Clinical_AUC': roc_auc_score(y_true_bin, clinical_proba) if len(
                    np.unique(y_true_bin)) > 1 else np.nan,
                'AUC_P_Value': auc_p
            }

        # 计算ML预测标签
        ml_pred_pbc = self.best_model.predict(X_val_top_pbc)
        y_true_pbc = np.ones_like(ml_pred_pbc)  # PBC患者标签为1

        # 分组计算指标
        pos_metrics = {}
        if np.any(ama_pos_mask):
            pos_metrics = calculate_group_metrics(
                y_true_pbc[ama_pos_mask],
                ml_pred_pbc[ama_pos_mask],
                clinical_pred_pbc[ama_pos_mask],
                pbc_proba[ama_pos_mask],
                clinical_proba_pbc_pbc[ama_pos_mask]
            )
        else:
            pos_metrics = {k: np.nan for k in
                           ['ML_Accuracy', 'Clinical_Accuracy', 'ML_Sensitivity', 'Clinical_Sensitivity',
                            'ML_Specificity', 'Clinical_Specificity', 'ML_F1', 'Clinical_F1', 'ML_AUC',
                            'Clinical_AUC', 'AUC_P_Value']}

        neg_metrics = {}
        if np.any(ama_neg_mask):
            neg_metrics = calculate_group_metrics(
                y_true_pbc[ama_neg_mask],
                ml_pred_pbc[ama_neg_mask],
                clinical_pred_pbc[ama_neg_mask],
                pbc_proba[ama_neg_mask],
                clinical_proba_pbc_pbc[ama_neg_mask]
            )
        else:
            neg_metrics = {k: np.nan for k in pos_metrics.keys()}

        stratified_results = pd.DataFrame({
            'AMA_M2_Status': ['Positive', 'Negative'],
            'Sample_Count': [pos_count, neg_count],
            'ML_Accuracy': [pos_metrics['ML_Accuracy'], neg_metrics['ML_Accuracy']],
            'Clinical_Accuracy': [pos_metrics['Clinical_Accuracy'], neg_metrics['Clinical_Accuracy']],
            'ML_Sensitivity': [pos_metrics['ML_Sensitivity'], neg_metrics['ML_Sensitivity']],
            'Clinical_Sensitivity': [pos_metrics['Clinical_Sensitivity'], neg_metrics['Clinical_Sensitivity']],
            'ML_Specificity': [pos_metrics['ML_Specificity'], neg_metrics['ML_Specificity']],
            'Clinical_Specificity': [pos_metrics['Clinical_Specificity'], neg_metrics['Clinical_Specificity']],
            'ML_F1': [pos_metrics['ML_F1'], neg_metrics['ML_F1']],
            'Clinical_F1': [pos_metrics['Clinical_F1'], neg_metrics['Clinical_F1']],
            'ML_AUC': [pos_metrics['ML_AUC'], neg_metrics['ML_AUC']],
            'Clinical_AUC': [pos_metrics['Clinical_AUC'], neg_metrics['Clinical_AUC']],
            'AUC_P_Value': [pos_metrics['AUC_P_Value'], neg_metrics['AUC_P_Value']],
            'PC1_Mean': [np.mean(pc1_vals[ama_pos_mask]) if np.any(ama_pos_mask) else np.nan,
                         np.mean(pc1_vals[ama_neg_mask]) if np.any(ama_neg_mask) else np.nan]
        })
        stratified_results.to_csv(
            os.path.join(self.save_root, self.results_dir,
                         f'pbc_ama_m2_stratified_results_PCA_{self.feature_type_filter}.csv'),
            index=False, encoding='utf-8-sig'
        )

        logger.info(f"PBC diagnosis scatter plot (PCA axis: PC1 ({pc1_var}%)) saved successfully!")

    def plot_aih_stratified_analysis(self):
        """AIH患者按典型/非典型分层分析（验证集）：
        典型AIH: 核心特征→ANA/LKM-1任一自身抗体阳性
        非典型AIH: 核心特征→ANA/LKM-1自身抗体均阴性
        （核心依据：典型AIH以ANA/LKM-1等自身抗体阳性为核心诊断特征）
        """
        logger.info("Generating AIH patients stratified analysis (typical vs atypical)...")

        # 筛选验证集中的AIH患者（标签0）
        aih_mask = self.y_val == 0
        aih_count = np.sum(aih_mask)
        if aih_count == 0:
            logger.warning("No AIH patients in validation set, skip AIH stratified analysis")
            return

        # 提取AIH患者核心特征
        X_val_aih_scaled = self.X_val_scaled[aih_mask, :]
        X_val_raw_aih = self.val_data[self.feature_columns][aih_mask]

        # 1. 典型AIH判断逻辑（核心：仅ANA/LKM-1任一自身抗体阳性）
        # 1.1 自身抗体判断（仅ANA/LKM-1，无SMA特征）
        # 抗体映射：1=阳性，0=阴性；任一阳性即判定为典型AIH
        auto_ab_cols = {
            'ANA': self.feature_columns.index('ANA') if 'ANA' in self.feature_columns else -1,
            'Anti-LKM-1': self.feature_columns.index('Anti-LKM-1') if 'Anti-LKM-1' in self.feature_columns else -1
        }
        # 初始化：所有患者抗体阴性
        aih_ab_positive = np.zeros(aih_count, dtype=bool)
        # 遍历抗体列，只要任一抗体阳性（1），则判定为典型AIH核心特征满足
        for ab_name, ab_idx in auto_ab_cols.items():
            if ab_idx != -1:  # 仅处理数据中存在的抗体
                ab_raw = self.val_data[ab_name].values[aih_mask]  # 原始抗体值（0=阴，1=阳）
                # 处理抗体值缺失（nan）：缺失值不判定为阳性
                ab_raw_no_nan = np.nan_to_num(ab_raw, nan=0)
                aih_ab_positive = aih_ab_positive | (ab_raw_no_nan == 1)
                logger.info(f"AIH患者{ab_name}阳性数：{np.sum(ab_raw_no_nan == 1)}/{aih_count}")

        # 1.2 典型/非典型分组（仅基于ANA/LKM-1）
        # 典型AIH：ANA/LKM-1任一阳性
        typical_aih_mask = aih_ab_positive
        # 非典型AIH：ANA/LKM-1均阴性
        atypical_aih_mask = ~aih_ab_positive
        # 处理抗体值缺失（nan）：缺失值排除在分组外
        ab_nan_mask = np.zeros(aih_count, dtype=bool)
        for ab_name, ab_idx in auto_ab_cols.items():
            if ab_idx != -1:
                ab_raw = self.val_data[ab_name].values[aih_mask]
                ab_nan_mask = ab_nan_mask | np.isnan(ab_raw)
        # 缺失值不参与分层
        typical_aih_mask = typical_aih_mask & ~ab_nan_mask
        atypical_aih_mask = atypical_aih_mask & ~ab_nan_mask

        # 统计分组样本量（突出核心特征）
        typical_count = np.sum(typical_aih_mask)
        atypical_count = np.sum(atypical_aih_mask)
        logger.info(f"=== AIH分层统计（核心：ANA/LKM-1） ===")
        logger.info(f"典型AIH（ANA/LKM-1任一阳）：{typical_count}")
        logger.info(f"非典型AIH（ANA/LKM-1均阴）：{atypical_count}")
        logger.info(f"抗体值缺失数：{np.sum(ab_nan_mask)}")

        # 2. PCA降维（复用逻辑）
        pca = PCA(n_components=2)
        X_aih_pca = pca.fit_transform(X_val_aih_scaled)
        pc1_var = round(pca.explained_variance_ratio_[0] * 100, 2)

        # 3. 计算AIH预测概率（top特征）
        top_indices = [self.reduced_feature_names.index(f) for f in self.top_features if
                       f in self.reduced_feature_names]
        X_val_reduced_aih = self.X_val_reduced[aih_mask, :]
        X_val_top_aih = X_val_reduced_aih[:, top_indices] if len(top_indices) > 0 else X_val_reduced_aih
        aih_proba = self.best_model.predict_proba(X_val_top_aih)[:, 0]  # AIH预测概率

        # 4. 绘制典型/非典型散点图（仅基于ANA/LKM-1分层）
        fig, ax = plt.subplots(figsize=(8, 6))
        # 典型AIH散点（ANA/LKM-1任一阳）
        if np.any(typical_aih_mask):
            ax.scatter(
                aih_proba[typical_aih_mask], X_aih_pca[typical_aih_mask, 0],
                color='#E64B35', label='Typical AIH (ANA/LKM-1+)', alpha=0.8, s=60,
                edgecolors='black', linewidth=0.5
            )
        # 非典型AIH散点（ANA/LKM-1均阴）
        if np.any(atypical_aih_mask):
            ax.scatter(
                aih_proba[atypical_aih_mask], X_aih_pca[atypical_aih_mask, 0],
                color='#4DBBD5', label='Atypical AIH (ANA/LKM-1-)', alpha=0.8, s=60,
                edgecolors='black', linewidth=0.5, marker='s'
            )
        # 诊断阈值线
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7,
                   label='AIH Diagnosis Threshold (0.5)')

        # 图表样式（突出核心分层依据）
        ax.set_xlabel('Predicted Probability for AIH (ML Model)', fontsize=9)
        ax.set_ylabel(f'PC1 ({pc1_var}%)', fontsize=9)
        ax.set_title(
            f'AIH Diagnosis Prediction vs PC1 ({pc1_var}%)\nStratified by ANA/LKM-1 Status',
            fontsize=10, pad=15
        )
        ax.legend(fontsize=8, loc='best')   #'upper right'
        ax.grid(axis='both', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # 样本量标注（仅抗体相关）
        ax.text(
            0.05, 0.95,
            f'Typical AIH: {typical_count}\nAtypical AIH: {atypical_count}\nMissing Ab values: {np.sum(ab_nan_mask)}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'aih_diagnosis_scatter_PCA_{self.feature_type_filter}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'aih_diagnosis_scatter_PCA_{self.feature_type_filter}.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'aih_diagnosis_scatter_PCA_{self.feature_type_filter}.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

        # 5. AIH核心特征箱线图（分层对比，仅ANA/LKM-1）
        key_features = ['ALT', 'AST', 'ANA']  # 保留核心特征，ANA为核心抗体
        key_features = [f for f in key_features if f in self.feature_columns]
        if len(key_features) >= 2:
            n_feats = len(key_features)
            fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 5))
            if n_feats == 1:
                axes = [axes]

            for idx, feat in enumerate(key_features):
                ax = axes[idx]
                feat_idx = self.feature_columns.index(feat)
                # 按典型/非典型分层提取特征值
                feat_vals_typical = X_val_aih_scaled[typical_aih_mask, feat_idx] if np.any(
                    typical_aih_mask) else np.array([])
                feat_vals_atypical = X_val_aih_scaled[atypical_aih_mask, feat_idx] if np.any(
                    atypical_aih_mask) else np.array([])

                data = [feat_vals_typical, feat_vals_atypical]
                box_plot = ax.boxplot(data, labels=['Typical (ANA/LKM-1+)', 'Atypical (ANA/LKM-1-)'],
                                      patch_artist=True, boxprops=dict(alpha=0.7),
                                      medianprops=dict(color='darkred', linewidth=1.5))
                box_plot['boxes'][0].set_facecolor('#E64B35')
                if len(box_plot['boxes']) > 1:
                    box_plot['boxes'][1].set_facecolor('#4DBBD5')

                # 散点叠加
                for i, d in enumerate(data):
                    if len(d) > 0:
                        x = np.random.normal(i + 1, 0.08, size=len(d))
                        ax.scatter(x, d, color='black', alpha=0.6, s=15, zorder=3)

                ax.set_xlabel('AIH Subtype (Core: ANA/LKM-1)', fontsize=8)
                ax.set_ylabel(f'{feat} (Standardized)', fontsize=8)
                ax.set_title(feat, fontsize=9, pad=8)
                ax.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(labelsize=7)

            fig.suptitle(f'AIH Key Features: Stratified by ANA/LKM-1 Status (Validation Set)',
                         fontsize=11, y=0.98)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'aih_stratified_boxplot_{self.feature_type_filter}.png'),
                dpi=600, bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'aih_stratified_boxplot_{self.feature_type_filter}.pdf'),
                dpi=600,
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'aih_stratified_boxplot_{self.feature_type_filter}.svg'),
                dpi=600,
                format='svg',
                bbox_inches='tight'
            )
            plt.close()

    def plot_os_stratified_analysis(self):
        """OS患者按典型/非典型分层分析（验证集）：
        典型OS: 同时满足 PBC抗体阳性 + AIH抗体阳性
                - PBC抗体：AMA/AMA-M2 任一阳性（1）
                - AIH抗体：ANA/Anti-LKM-1/Anti-SLA/LP 任一阳性（1）
        非典型OS: 不同时满足PBC+AIH抗体阳性（仅其一阳性/均阴性）
        """
        logger.info("Generating OS patients stratified analysis (typical vs atypical)...")

        # 筛选验证集中的OS患者（标签2）
        os_mask = self.y_val == 2
        os_count = np.sum(os_mask)
        if os_count == 0:
            logger.warning("No OS patients in validation set, skip OS stratified analysis")
            return

        # 提取OS患者核心数据
        X_val_os_scaled = self.X_val_scaled[os_mask, :]
        X_val_raw_os = self.val_data[self.feature_columns][os_mask].copy()  # 改为copy避免视图问题
        n_os_samples = len(X_val_raw_os)  # 获取OS患者样本数

        # 1. 定义PBC/AIH抗体阳性判断函数（核心：仅判断抗体，无条数逻辑）
        def _is_pbc_ab_positive(X):
            """判断每个样本是否PBC抗体阳性（AMA/AMA-M2任一阳），返回布尔数组"""
            pbc_ab_pos = np.zeros(len(X), dtype=bool)
            # PBC核心抗体：AMA、AMA-M2
            pbc_ab_cols = ['AMA', 'AMA-M2']
            for col in pbc_ab_cols:
                if col in X.columns:
                    # 抗体值1=阳性，0=阴性，nan视为阴性
                    ab_vals = X[col].fillna(0).values
                    pbc_ab_pos = pbc_ab_pos | (ab_vals == 1)
            return pbc_ab_pos

        def _is_aih_ab_positive(X):
            """判断每个样本是否AIH抗体阳性（ANA/Anti-LKM-1/Anti-SLA/LP任一阳），返回布尔数组"""
            aih_ab_pos = np.zeros(len(X), dtype=bool)
            # AIH核心抗体：ANA、Anti-LKM-1、Anti-SLA/LP
            aih_ab_cols = ['ANA', 'Anti-LKM-1', 'Anti-SLA/LP']
            for col in aih_ab_cols:
                if col in X.columns:
                    # 抗体值1=阳性，0=阴性，nan视为阴性
                    ab_vals = X[col].fillna(0).values
                    aih_ab_pos = aih_ab_pos | (ab_vals == 1)
            return aih_ab_pos

        # 2. 典型OS判断（核心：同时满足PBC+AIH抗体阳性）
        pbc_ab_positive = _is_pbc_ab_positive(X_val_raw_os)
        aih_ab_positive = _is_aih_ab_positive(X_val_raw_os)

        # 维度校验（确保和OS样本数匹配）
        assert len(pbc_ab_positive) == n_os_samples, f"PBC抗体判断维度错误：{len(pbc_ab_positive)} vs {n_os_samples}"
        assert len(aih_ab_positive) == n_os_samples, f"AIH抗体判断维度错误：{len(aih_ab_positive)} vs {n_os_samples}"

        # 典型/非典型分组
        typical_os_mask = pbc_ab_positive & aih_ab_positive  # 同时阳性=典型OS
        atypical_os_mask = ~typical_os_mask  # 不同时阳性=非典型OS

        # 处理抗体缺失值（单独标注缺失样本，避免干扰分组）
        # 缺失值定义：任一核心抗体字段为nan
        ab_nan_mask = np.zeros(n_os_samples, dtype=bool)
        all_ab_cols = ['AMA', 'AMA-M2', 'ANA', 'Anti-LKM-1', 'Anti-SLA/LP']
        for col in all_ab_cols:
            if col in X_val_raw_os.columns:
                ab_nan_mask = ab_nan_mask | X_val_raw_os[col].isna().values
        # 缺失值排除在分组外
        typical_os_mask = typical_os_mask & ~ab_nan_mask
        atypical_os_mask = atypical_os_mask & ~ab_nan_mask

        # 统计样本量（细化抗体阳性分布）
        typical_count = np.sum(typical_os_mask)
        atypical_count = np.sum(atypical_os_mask)
        # 额外统计：非典型OS中仅PBC抗体阳/仅AIH抗体阳/均阴性的数量
        only_pbc_ab = (~typical_os_mask) & pbc_ab_positive & ~aih_ab_positive & ~ab_nan_mask
        only_aih_ab = (~typical_os_mask) & ~pbc_ab_positive & aih_ab_positive & ~ab_nan_mask
        both_ab_neg = (~typical_os_mask) & ~pbc_ab_positive & ~aih_ab_positive & ~ab_nan_mask
        logger.info(f"=== OS分层统计（核心：PBC+AIH抗体同时阳性） ===")
        logger.info(f"典型OS（双抗体阳）：{typical_count}")
        logger.info(
            f"非典型OS：{atypical_count}（仅PBC抗体阳：{np.sum(only_pbc_ab)} | 仅AIH抗体阳：{np.sum(only_aih_ab)} | 均阴性：{np.sum(both_ab_neg)}）")
        logger.info(f"抗体值缺失样本：{np.sum(ab_nan_mask)}")

        # 3. PCA降维
        pca = PCA(n_components=2)
        X_os_pca = pca.fit_transform(X_val_os_scaled)
        pc1_var = round(pca.explained_variance_ratio_[0] * 100, 2)

        # 4. OS预测概率（确保维度匹配）
        top_indices = [self.reduced_feature_names.index(f) for f in self.top_features if
                       f in self.reduced_feature_names]
        X_val_reduced_os = self.X_val_reduced[os_mask, :]
        X_val_top_os = X_val_reduced_os[:, top_indices] if len(top_indices) > 0 else X_val_reduced_os
        os_proba = self.best_model.predict_proba(X_val_top_os)[:, 2]  # OS预测概率（标签2）
        assert len(os_proba) == n_os_samples, f"OS预测概率维度错误：{len(os_proba)} vs {n_os_samples}"

        # 5. OS分层散点图（PCA轴）- 突出抗体分层
        fig, ax = plt.subplots(figsize=(8, 6))

        # 典型OS散点（双抗体阳）
        if np.any(typical_os_mask):
            x_typical = os_proba[typical_os_mask]
            y_typical = X_os_pca[typical_os_mask, 0]
            assert len(x_typical) == len(y_typical), f"典型OS散点维度不匹配：{len(x_typical)} vs {len(y_typical)}"
            ax.scatter(
                x_typical, y_typical,
                color='#00A087', label='Typical OS (PBC+AIH Ab+)', alpha=0.8, s=60, edgecolors='black', linewidth=0.5
            )

        # 非典型OS散点（不同时阳）
        if np.any(atypical_os_mask):
            x_atypical = os_proba[atypical_os_mask]
            y_atypical = X_os_pca[atypical_os_mask, 0]
            assert len(x_atypical) == len(y_atypical), f"非典型OS散点维度不匹配：{len(x_atypical)} vs {len(y_atypical)}"
            ax.scatter(
                x_atypical, y_atypical,
                color='#3C5488', label='Atypical OS (Not PBC+AIH Ab+)', alpha=0.8, s=60, edgecolors='black',
                linewidth=0.5, marker='s'
            )

        # 诊断阈值
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label='OS Diagnosis Threshold (0.5)')

        # 图表样式（突出抗体分层）
        ax.set_xlabel('Predicted Probability for OS (ML Model)', fontsize=9)
        ax.set_ylabel(f'PC1 ({pc1_var}%)', fontsize=9)
        ax.set_title(
            f'OS Diagnosis Prediction vs PC1 ({pc1_var}%)\nStratified by PBC+AIH Autoantibody Status (Validation Set)',
            fontsize=10, pad=15
        )
        ax.legend(fontsize=8, loc='best')
        ax.grid(axis='both', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # 样本量标注（细化非典型OS构成）
        ax.text(
            0.05, 0.95,
            f'Typical OS: {typical_count}\nAtypical OS: {atypical_count}\n- Only PBC Ab+: {np.sum(only_pbc_ab)}\n- Only AIH Ab+: {np.sum(only_aih_ab)}\n- Both Ab-: {np.sum(both_ab_neg)}',
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'os_diagnosis_scatter_PCA_{self.feature_type_filter}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'os_diagnosis_scatter_PCA_{self.feature_type_filter}.pdf'),
            dpi=600,
            format='pdf',
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(self.save_root, self.figure_dir,
                         f'os_diagnosis_scatter_PCA_{self.feature_type_filter}.svg'),
            dpi=600,
            format='svg',
            bbox_inches='tight'
        )
        plt.close()

        # 6. OS核心特征箱线图（按抗体分层）
        key_features = ['ALP(35-100)', 'IgG', 'AMA-M2', 'ALT', 'ANA']  # 保留核心抗体/生化特征
        key_features = [f for f in key_features if f in self.feature_columns]
        if len(key_features) >= 2:
            n_feats = len(key_features)
            fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 5))
            if n_feats == 1:
                axes = [axes]

            for idx, feat in enumerate(key_features):
                ax = axes[idx]
                feat_idx = self.feature_columns.index(feat)
                # 按典型/非典型分层提取特征值
                feat_vals_typical = X_val_os_scaled[typical_os_mask, feat_idx] if np.any(typical_os_mask) else np.array(
                    [])
                feat_vals_atypical = X_val_os_scaled[atypical_os_mask, feat_idx] if np.any(
                    atypical_os_mask) else np.array([])

                data = [feat_vals_typical, feat_vals_atypical]
                box_plot = ax.boxplot(data, labels=['Typical (PBC+AIH Ab+)', 'Atypical'], patch_artist=True,
                                      boxprops=dict(alpha=0.7), medianprops=dict(color='darkred', linewidth=1.5))
                box_plot['boxes'][0].set_facecolor('#00A087')
                if len(box_plot['boxes']) > 1:
                    box_plot['boxes'][1].set_facecolor('#3C5488')

                # 散点叠加
                for i, d in enumerate(data):
                    if len(d) > 0:
                        x = np.random.normal(i + 1, 0.08, size=len(d))
                        ax.scatter(x, d, color='black', alpha=0.6, s=15, zorder=3)

                ax.set_xlabel('OS Subtype (Core: PBC+AIH Ab)', fontsize=8)
                ax.set_ylabel(f'{feat} (Standardized)', fontsize=8)
                ax.set_title(feat, fontsize=9, pad=8)
                ax.grid(axis='y', color=COLOR_PALETTE['grid'], linestyle='-', linewidth=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(labelsize=7)

            fig.suptitle(f'OS Key Features: Stratified by PBC+AIH Autoantibody Status (Validation Set)',
                         fontsize=11, y=0.98)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'os_stratified_boxplot_{self.feature_type_filter}.png'),
                dpi=600, bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'os_stratified_boxplot_{self.feature_type_filter}.pdf'),
                dpi=600,
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(self.save_root, self.figure_dir,
                             f'os_stratified_boxplot_{self.feature_type_filter}.svg'),
                dpi=600,
                format='svg',
                bbox_inches='tight'
            )
            plt.close()

        # 7. OS分层诊断性能（修正AUC计算逻辑：多分类转二分类）
        # 典型组性能
        typical_auc = np.nan
        typical_sensitivity = np.nan
        typical_specificity = np.nan
        if np.any(typical_os_mask):
            try:
                # 二分类AUC：预测为OS（概率） vs 真实OS标签
                typical_auc = roc_auc_score(
                    y_true=np.ones(len(os_proba[typical_os_mask])),  # 真实均为OS
                    y_score=os_proba[typical_os_mask]
                )
                # 计算灵敏度/特异度（阈值0.5）
                typical_pred = (os_proba[typical_os_mask] >= 0.5).astype(int)
                typical_sensitivity = np.sum(typical_pred == 1) / len(typical_pred)
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for Typical OS: {e}")

        # 非典型组性能
        atypical_auc = np.nan
        atypical_sensitivity = np.nan
        atypical_specificity = np.nan
        if np.any(atypical_os_mask):
            try:
                atypical_auc = roc_auc_score(
                    y_true=np.ones(len(os_proba[atypical_os_mask])),
                    y_score=os_proba[atypical_os_mask]
                )
                atypical_pred = (os_proba[atypical_os_mask] >= 0.5).astype(int)
                atypical_sensitivity = np.sum(atypical_pred == 1) / len(atypical_pred)
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for Atypical OS: {e}")

        # 保存结果（替换原条数统计，改为抗体阳性率）
        stratified_results = pd.DataFrame({
            'OS_Subtype': ['Typical (PBC+AIH Ab+)', 'Atypical (Not both Ab+)'],
            'Sample_Count': [typical_count, atypical_count],
            'ML_AUC': [typical_auc, atypical_auc],
            'ML_Sensitivity': [typical_sensitivity, atypical_sensitivity],
            'PBC_Ab_Positive_Rate': [1.0 if typical_count > 0 else np.nan,  # 典型组PBC抗体全阳
                                     np.sum(pbc_ab_positive[
                                                atypical_os_mask]) / atypical_count if atypical_count > 0 else np.nan],
            'AIH_Ab_Positive_Rate': [1.0 if typical_count > 0 else np.nan,  # 典型组AIH抗体全阳
                                     np.sum(aih_ab_positive[
                                                atypical_os_mask]) / atypical_count if atypical_count > 0 else np.nan]
        })
        stratified_results.to_csv(
            os.path.join(self.save_root, self.results_dir,
                         f'os_stratified_results_{self.feature_type_filter}.csv'),
            index=False, encoding='utf-8-sig'
        )
        logger.info("OS stratified analysis (by PBC+AIH autoantibody) plots and results saved")

    # def write_predictions_to_excel(self):
    #     """将模型预测结果写入原始Excel并保存新文件到结果目录"""
    #     logger.info("\nWriting prediction results back to original Excel sheet...")
    #
    #     # 1. 准备最佳模型的预测结果（训练集+验证集）
    #     top_indices = [self.reduced_feature_names.index(f) for f in self.top_features if
    #                    f in self.reduced_feature_names]
    #
    #     # 训练集预测
    #     X_train_top = self.X_train_reduced[:, top_indices]
    #     train_pred = self.best_model.predict(X_train_top)
    #     train_pred_proba = self.best_model.predict_proba(X_train_top)
    #
    #     # 验证集预测
    #     X_val_top = self.X_val_reduced[:, top_indices]
    #     val_pred = self.best_model.predict(X_val_top)
    #     val_pred_proba = self.best_model.predict_proba(X_val_top)
    #
    #     # 2. 映射预测标签到类别名称（AIH/PBC/OS/CTR）
    #     train_pred_names = [self.class_name_mapping[p] for p in train_pred]
    #     val_pred_names = [self.class_name_mapping[p] for p in val_pred]
    #
    #     # 3. 给原始数据添加预测列
    #     # 训练集添加预测结果
    #     train_data_with_pred = self.train_data.copy()
    #     train_data_with_pred['ML_Predicted_Label'] = train_pred
    #     train_data_with_pred['ML_Predicted_Class'] = train_pred_names
    #     # 添加各类别概率列
    #     for i, cls_name in enumerate(self.class_names):
    #         train_data_with_pred[f'ML_Prob_{cls_name}'] = train_pred_proba[:, i]
    #
    #     # 验证集添加预测结果
    #     val_data_with_pred = self.val_data.copy()
    #     val_data_with_pred['ML_Predicted_Label'] = val_pred
    #     val_data_with_pred['ML_Predicted_Class'] = val_pred_names
    #     for i, cls_name in enumerate(self.class_names):
    #         val_data_with_pred[f'ML_Prob_{cls_name}'] = val_pred_proba[:, i]
    #
    #     # 4. 合并训练/验证集并匹配原始数据顺序（按patient_id）
    #     combined_pred = pd.concat([train_data_with_pred, val_data_with_pred], ignore_index=True)
    #     raw_data_with_pred = self.raw_data.merge(
    #         combined_pred[['patient_id', 'ML_Predicted_Label', 'ML_Predicted_Class'] +
    #                       [f'ML_Prob_{cls_name}' for cls_name in self.class_names]],
    #         on='patient_id', how='left'
    #     )
    #
    #     # 5. 保存新Excel文件到结果目录
    #     output_excel_path = os.path.join(
    #         self.save_root, self.results_dir,
    #         f'prediction_results_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}.xlsx'
    #     )
    #
    #     # 写入原sheet（analysis_2_584）+ 新增汇总sheet
    #     with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    #         # 原sheet保留所有列 + 预测列
    #         raw_data_with_pred.to_excel(writer, sheet_name='analysis_2_584', index=False)
    #
    #         # 新增预测汇总sheet（更易查看）
    #         prediction_summary = pd.DataFrame({
    #             'Patient_ID': raw_data_with_pred['patient_id'],
    #             'Dataset_Type': raw_data_with_pred['dataset_type'],
    #             'True_Label': raw_data_with_pred['group_label'],
    #             'True_Class': [self.class_name_mapping.get(l - self.label_offset, 'Unknown')
    #                            for l in raw_data_with_pred['group_label']],
    #             'ML_Predicted_Label': raw_data_with_pred['ML_Predicted_Label'],
    #             'ML_Predicted_Class': raw_data_with_pred['ML_Predicted_Class'],
    #             **{f'ML_Prob_{cls_name}': raw_data_with_pred[f'ML_Prob_{cls_name}']
    #                for cls_name in self.class_names}
    #         })
    #         # 计算预测准确率
    #         correct_pred = (
    #             (raw_data_with_pred['group_label'] - self.label_offset) == raw_data_with_pred['ML_Predicted_Label']
    #         ).fillna(False)
    #         prediction_summary['Is_Correct'] = correct_pred
    #         prediction_summary.to_excel(writer, sheet_name='Prediction_Summary', index=False)
    #
    #     logger.info(f"✅ 预测结果已保存至: {output_excel_path}")
    #     logger.info(f"📊 整体预测准确率: {correct_pred.sum() / len(correct_pred):.4f} (排除缺失值)")
    #     logger.info(f"📈 有效预测样本数: {len(raw_data_with_pred.dropna(subset=['ML_Predicted_Class']))}")

    def write_predictions_to_excel(self):
        """将模型预测结果 AND 临床指南诊断结果写入原始Excel并保存新文件到结果目录"""
        logger.info("\nWriting prediction results (ML + Clinical Guideline) back to original Excel sheet...")

        # 1. 准备最佳模型的预测结果（训练集+验证集）
        top_indices = [self.reduced_feature_names.index(f) for f in self.top_features if
                       f in self.reduced_feature_names]

        # 训练集预测 (ML)
        X_train_top = self.X_train_reduced[:, top_indices]
        train_pred = self.best_model.predict(X_train_top)
        train_pred_proba = self.best_model.predict_proba(X_train_top)

        # 验证集预测 (ML)
        X_val_top = self.X_val_reduced[:, top_indices]
        val_pred = self.best_model.predict(X_val_top)
        val_pred_proba = self.best_model.predict_proba(X_val_top)

        # 2. 映射预测标签到类别名称（AIH/PBC/OS/CTR）
        train_pred_names = [self.class_name_mapping[p] for p in train_pred]
        val_pred_names = [self.class_name_mapping[p] for p in val_pred]

        # ==================== 新增：临床指南诊断结果 ====================
        # 获取原始特征数据用于临床诊断
        X_train_raw = self.train_data[self.feature_columns]
        X_val_raw = self.val_data[self.feature_columns]

        # 调用临床指南诊断函数
        train_clinical_pred, _ = self._clinical_guideline_diagnosis(X_train_raw)
        val_clinical_pred, _ = self._clinical_guideline_diagnosis(X_val_raw)

        # 映射临床指南预测标签到名称
        train_clinical_pred_names = [self.class_name_mapping[int(p)] if 0 <= int(p) < self.n_classes else 'Unknown' for
                                     p in train_clinical_pred]
        val_clinical_pred_names = [self.class_name_mapping[int(p)] if 0 <= int(p) < self.n_classes else 'Unknown' for p
                                   in val_clinical_pred]
        # =================================================================

        # 3. 给原始数据添加预测列
        # 训练集添加预测结果
        train_data_with_pred = self.train_data.copy()
        train_data_with_pred['ML_Predicted_Label'] = train_pred
        train_data_with_pred['ML_Predicted_Class'] = train_pred_names
        # 新增：添加临床指南预测列
        train_data_with_pred['Clinical_Predicted_Label'] = train_clinical_pred
        train_data_with_pred['Clinical_Predicted_Class'] = train_clinical_pred_names

        # 添加各类别概率列 (ML)
        for i, cls_name in enumerate(self.class_names):
            train_data_with_pred[f'ML_Prob_{cls_name}'] = train_pred_proba[:, i]

        # 验证集添加预测结果
        val_data_with_pred = self.val_data.copy()
        val_data_with_pred['ML_Predicted_Label'] = val_pred
        val_data_with_pred['ML_Predicted_Class'] = val_pred_names
        # 新增：添加临床指南预测列
        val_data_with_pred['Clinical_Predicted_Label'] = val_clinical_pred
        val_data_with_pred['Clinical_Predicted_Class'] = val_clinical_pred_names

        for i, cls_name in enumerate(self.class_names):
            val_data_with_pred[f'ML_Prob_{cls_name}'] = val_pred_proba[:, i]

        # 4. 合并训练/验证集并匹配原始数据顺序（按patient_id）
        combined_pred = pd.concat([train_data_with_pred, val_data_with_pred], ignore_index=True)

        # 定义需要合并的新列
        merge_cols = ['patient_id',
                      'ML_Predicted_Label', 'ML_Predicted_Class',
                      'Clinical_Predicted_Label', 'Clinical_Predicted_Class'] + \
                     [f'ML_Prob_{cls_name}' for cls_name in self.class_names]

        raw_data_with_pred = self.raw_data.merge(
            combined_pred[merge_cols],
            on='patient_id', how='left'
        )

        # 5. 保存新Excel文件到结果目录
        output_excel_path = os.path.join(
            self.save_root, self.results_dir,
            f'prediction_results_{self.dimension_reduction_method}_{self.feature_selection_method}_{self.feature_type_filter}.xlsx'
        )

        # 写入原sheet（analysis_2_584）+ 新增汇总sheet
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            # 原sheet保留所有列 + 预测列
            raw_data_with_pred.to_excel(writer, sheet_name='analysis_2_584', index=False)

            # 新增预测汇总sheet（更易查看）
            prediction_summary = pd.DataFrame({
                'Patient_ID': raw_data_with_pred['patient_id'],
                'Dataset_Type': raw_data_with_pred['dataset_type'],
                'True_Label': raw_data_with_pred['group_label'],
                'True_Class': [self.class_name_mapping.get(l - self.label_offset, 'Unknown')
                               for l in raw_data_with_pred['group_label']],
                # ML 结果
                'ML_Predicted_Label': raw_data_with_pred['ML_Predicted_Label'],
                'ML_Predicted_Class': raw_data_with_pred['ML_Predicted_Class'],
                # 新增：临床指南结果
                'Clinical_Predicted_Label': raw_data_with_pred['Clinical_Predicted_Label'],
                'Clinical_Predicted_Class': raw_data_with_pred['Clinical_Predicted_Class'],
                **{f'ML_Prob_{cls_name}': raw_data_with_pred[f'ML_Prob_{cls_name}']
                   for cls_name in self.class_names}
            })
            # 计算预测准确率 (ML)
            correct_pred = (
                (raw_data_with_pred['group_label'] - self.label_offset) == raw_data_with_pred['ML_Predicted_Label']
            ).fillna(False)
            prediction_summary['ML_Is_Correct'] = correct_pred

            # 新增：计算临床指南准确率
            correct_clinical = (
                (raw_data_with_pred['group_label'] - self.label_offset) == raw_data_with_pred[
                    'Clinical_Predicted_Label']
            ).fillna(False)
            prediction_summary['Clinical_Is_Correct'] = correct_clinical

            prediction_summary.to_excel(writer, sheet_name='Prediction_Summary', index=False)

        logger.info(f"✅ 预测结果 (ML + 临床指南) 已保存至: {output_excel_path}")
        logger.info(f"📊 ML 整体预测准确率: {correct_pred.sum() / len(correct_pred):.4f}")
        logger.info(f"📊 临床指南整体预测准确率: {correct_clinical.sum() / len(correct_clinical):.4f}")


# ====================== Main Execution =======================
def main():
    # 解析命令行参数
    args = parse_args()

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
    os.environ["XGBOOST_CACHE_SIZE"] = "0"

    # 设置日志
    global logger
    logger = setup_logging(args, args.dimension_reduction_method, args.feature_selection_method)

    logger.info("=" * 80)
    logger.info("Autoimmune Liver Disease Diagnostic Model Pipeline")
    logger.info("(Clinical Logic + Feature Type Filter + GPU Acceleration)")
    logger.info("=" * 80)
    logger.info(f"Configuration Summary:")
    logger.info(f"  - Feature Type Filter: {args.feature_type_filter}")
    logger.info(f"  - Imputation Method: {args.imputation_method}")
    logger.info(f"  - Feature Selection Method: {args.feature_selection_method.upper()}")
    logger.info(f"  - Dimension Reduction: {args.dimension_reduction_method.upper()}")
    logger.info(f"  - Selected Features Count: {args.n_selected}")
    logger.info(f"  - GPU Acceleration: {'Enabled' if args.use_gpu else 'Disabled'}")

    logger.info("=" * 80)

    try:
        # 初始化分析器
        analyzer = MedicalDataAnalyzer(args)

        # 执行完整流程
        analyzer.load_data() \
            .preprocess_data() \
            .filter_lipid_features() \
            .reduce_dimension() \
            .select_top_features() \
            .grid_search_with_cv() \
            .evaluate_models() \
            .generate_visualizations()

        logger.info("=" * 80)
        logger.info("Pipeline Completed Successfully!")
        logger.info(f"Results Directory: results_3/ (includes feature type filter tag)")
        logger.info(f"Figures Directory: figure_3/ (journal-grade visualizations)")
        logger.info(f"Best Model Directory: saved_best_model/")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Pipeline Failed with Error: {str(e)}", exc_info=True)
        logger.error("=" * 80)
        raise


if __name__ == "__main__":
    main()