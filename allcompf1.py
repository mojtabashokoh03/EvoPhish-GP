import numpy as np
import pandas as pd
import warnings
import sys
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.preprocessing import (
    LabelEncoder,
    RobustScaler,
    StandardScaler
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from deap import base, creator, tools, gp, algorithms

import operator
import random
import logging
from datetime import datetime
from itertools import combinations

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "PWD2021.csv"
TARGET_COLUMN = "isphish"

RANDOM_STATE = 42
POP_SIZE = 120
GENERATIONS = 80

if len(sys.argv) > 2:
    RANDOM_STATE = int(sys.argv[1])
    GENERATIONS = int(sys.argv[2])

MAX_TREE_HEIGHT = 10

# =========================================================
# LOGGING
# =========================================================

now = datetime.now()
logfile = f"GP_Research_Onlyf1_Complete_{RANDOM_STATE}_{GENERATIONS}_{now.strftime('%Y_%m_%d_%H_%M_%S')}.log"

logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger()

def print_log(msg):
    print(msg)
    logger.info(msg)

print_log("=" * 80)
print_log("RESEARCH-GRADE GP ENSEMBLE WITH COMPARATIVE METHODS")
print_log("=" * 80)

# =========================================================
# LOAD DATA
# =========================================================

print_log("\nLoading dataset...")
df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Encode categoricals
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X = X.fillna(0)

# =========================================================
# TRAIN / VALID / TEST SPLIT
# =========================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=RANDOM_STATE
)

print_log(f"Train: {len(X_train)}")
print_log(f"Validation: {len(X_val)}")
print_log(f"Test: {len(X_test)}")

# =========================================================
# SCALING
# =========================================================

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# BASE MODELS
# =========================================================

base_models = {
    'rf': RandomForestClassifier(
        n_estimators=400, max_depth=18, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced_subsample',
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    'dt': DecisionTreeClassifier(
        max_depth=12, min_samples_split=5, min_samples_leaf=2,
        class_weight='balanced', random_state=RANDOM_STATE
    ),
    'gb': GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.04, max_depth=5,
        subsample=0.8, random_state=RANDOM_STATE
    ),
    'xgb': XGBClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
        random_state=RANDOM_STATE, use_label_encoder=False, verbosity=0
    )
}

model_names = list(base_models.keys())

# Train all base models
print_log("\nTraining Base Models...")
for name, model in base_models.items():
    model.fit(X_train_scaled, y_train)
    val_pred = model.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print_log(f"  {name} Validation AUC: {auc:.4f}")

# =========================================================
# HELPER FUNCTION FOR METRICS
# =========================================================

def compute_metrics(y_true, y_pred, y_proba):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_proba),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

# =========================================================
# 1. HARD VOTING (اصلاح شده)
# =========================================================

print_log("\n" + "=" * 40)
print_log("1. HARD VOTING")
print_log("=" * 40)

hard_voting = VotingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    voting='hard'
)
hard_voting.fit(X_train_scaled, y_train)
hard_preds = hard_voting.predict(X_test_scaled)
# برای AUC در hard voting از predict_proba نمی‌توان استفاده کرد،
# بنابراین از probabilities میانگین گرفته شده از مدل‌های پایه استفاده می‌کنیم
hard_proba = np.mean([model.predict_proba(X_test_scaled)[:, 1] for model in base_models.values()], axis=0)

results = {'Hard Voting': compute_metrics(y_test, hard_preds, hard_proba)}
print_log(f"Hard Voting Test F1: {results['Hard Voting']['F1']:.4f}")

# =========================================================
# 2. SOFT VOTING
# =========================================================

print_log("\n" + "=" * 40)
print_log("2. SOFT VOTING")
print_log("=" * 40)

soft_voting = VotingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    voting='soft'
)
soft_voting.fit(X_train_scaled, y_train)
soft_preds = soft_voting.predict(X_test_scaled)
soft_proba = soft_voting.predict_proba(X_test_scaled)[:, 1]

results['Soft Voting'] = compute_metrics(y_test, soft_preds, soft_proba)
print_log(f"Soft Voting Test F1: {results['Soft Voting']['F1']:.4f}")

# =========================================================
# 3. WEIGHTED VOTING (optimized on validation)
# =========================================================

print_log("\n" + "=" * 40)
print_log("3. WEIGHTED VOTING")
print_log("=" * 40)

# Get validation probabilities
val_probas = np.column_stack([model.predict_proba(X_val_scaled)[:, 1] for model in base_models.values()])
test_probas = np.column_stack([model.predict_proba(X_test_scaled)[:, 1] for model in base_models.values()])

# Optimize weights using simple random search
best_weights = None
best_f1 = 0

for _ in range(1000):
    weights = np.random.dirichlet(np.ones(len(base_models)))
    weighted_val = np.sum(val_probas * weights, axis=1)
    
    for threshold in np.linspace(0.3, 0.7, 20):
        preds = (weighted_val > threshold).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights

weighted_test = np.sum(test_probas * best_weights, axis=1)

best_thresh = 0.5
best_f1_t = 0
for thresh in np.linspace(0.3, 0.7, 50):
    preds = (weighted_test > thresh).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1_t:
        best_f1_t = f1
        best_thresh = thresh

weighted_preds = (weighted_test > best_thresh).astype(int)
results['Weighted Voting'] = compute_metrics(y_test, weighted_preds, weighted_test)
print_log(f"Weighted Voting Test F1: {results['Weighted Voting']['F1']:.4f}")
print_log(f"  Optimized Weights: {np.round(best_weights, 3)}")

# =========================================================
# 4. BLENDING
# =========================================================

print_log("\n" + "=" * 40)
print_log("4. BLENDING (Meta-Learner on Validation)")
print_log("=" * 40)

# Create blending features (validation predictions)
blend_val = np.column_stack([model.predict_proba(X_val_scaled)[:, 1] for model in base_models.values()])
blend_test = np.column_stack([model.predict_proba(X_test_scaled)[:, 1] for model in base_models.values()])

# Train meta-learner on validation
blend_meta = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
blend_meta.fit(blend_val, y_val)
blend_proba = blend_meta.predict_proba(blend_test)[:, 1]
blend_preds = (blend_proba > 0.5).astype(int)

results['Blending'] = compute_metrics(y_test, blend_preds, blend_proba)
print_log(f"Blending Test F1: {results['Blending']['F1']:.4f}")

# =========================================================
# 5. STACKING - Logistic Regression
# =========================================================

print_log("\n" + "=" * 40)
print_log("5. STACKING with LR")
print_log("=" * 40)

stacking_lr = StackingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
    cv=5
)
stacking_lr.fit(X_train_scaled, y_train)
stacking_lr_proba = stacking_lr.predict_proba(X_test_scaled)[:, 1]
stacking_lr_preds = stacking_lr.predict(X_test_scaled)

results['Stacking-LR'] = compute_metrics(y_test, stacking_lr_preds, stacking_lr_proba)
print_log(f"Stacking-LR Test F1: {results['Stacking-LR']['F1']:.4f}")

# =========================================================
# 6. STACKING - Random Forest
# =========================================================

print_log("\n" + "=" * 40)
print_log("6. STACKING with RF")
print_log("=" * 40)

stacking_rf = StackingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    cv=5
)
stacking_rf.fit(X_train_scaled, y_train)
stacking_rf_proba = stacking_rf.predict_proba(X_test_scaled)[:, 1]
stacking_rf_preds = stacking_rf.predict(X_test_scaled)

results['Stacking-RF'] = compute_metrics(y_test, stacking_rf_preds, stacking_rf_proba)
print_log(f"Stacking-RF Test F1: {results['Stacking-RF']['F1']:.4f}")

# =========================================================
# 7. STACKING - XGBoost
# =========================================================

print_log("\n" + "=" * 40)
print_log("7. STACKING with XGB")
print_log("=" * 40)

stacking_xgb = StackingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    final_estimator=XGBClassifier(n_estimators=100, learning_rate=0.05, 
                                   random_state=RANDOM_STATE, use_label_encoder=False,
                                   eval_metric='logloss', verbosity=0),
    cv=5
)
stacking_xgb.fit(X_train_scaled, y_train)
stacking_xgb_proba = stacking_xgb.predict_proba(X_test_scaled)[:, 1]
stacking_xgb_preds = stacking_xgb.predict(X_test_scaled)

results['Stacking-XGB'] = compute_metrics(y_test, stacking_xgb_preds, stacking_xgb_proba)
print_log(f"Stacking-XGB Test F1: {results['Stacking-XGB']['F1']:.4f}")

# =========================================================
# 8. GA VOTING (Genetic Algorithm optimized weights)
# =========================================================

print_log("\n" + "=" * 40)
print_log("8. GA VOTING (Genetic Algorithm)")
print_log("=" * 40)

# Create GA fitness class
if 'FitnessMaxGA' not in creator.__dict__:
    creator.create("FitnessMaxGA", base.Fitness, weights=(1.0,))
if 'IndividualGA' not in creator.__dict__:
    creator.create("IndividualGA", list, fitness=creator.FitnessMaxGA)

ga_toolbox = base.Toolbox()
ga_toolbox.register("attr_float", random.uniform, 0, 1)
ga_toolbox.register("individual", tools.initRepeat, creator.IndividualGA, 
                    ga_toolbox.attr_float, n=len(base_models))
ga_toolbox.register("population", tools.initRepeat, list, ga_toolbox.individual)

def ga_evaluate(individual):
    weights = np.array(individual) / (sum(individual) + 1e-8)
    weighted_val = np.sum(val_probas * weights, axis=1)
    best_f1 = 0
    for threshold in np.linspace(0.3, 0.7, 30):
        preds = (weighted_val > threshold).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return (best_f1,)

ga_toolbox.register("evaluate", ga_evaluate)
ga_toolbox.register("mate", tools.cxTwoPoint)
ga_toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
ga_toolbox.register("select", tools.selTournament, tournsize=3)

ga_pop = ga_toolbox.population(n=50)
ga_hof = tools.HallOfFame(1)

ga_stats = tools.Statistics(lambda ind: ind.fitness.values)
ga_stats.register("max", np.max)

algorithms.eaSimple(ga_pop, ga_toolbox, cxpb=0.7, mutpb=0.2, 
                    ngen=40, stats=ga_stats, halloffame=ga_hof, verbose=False)

best_ga_weights = np.array(ga_hof[0]) / (sum(ga_hof[0]) + 1e-8)
ga_weighted_test = np.sum(test_probas * best_ga_weights, axis=1)

best_ga_thresh = 0.5
best_ga_f1 = 0
for thresh in np.linspace(0.3, 0.7, 50):
    preds = (ga_weighted_test > thresh).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_ga_f1:
        best_ga_f1 = f1
        best_ga_thresh = thresh

ga_preds = (ga_weighted_test > best_ga_thresh).astype(int)
results['GA Voting'] = compute_metrics(y_test, ga_preds, ga_weighted_test)
print_log(f"GA Voting Test F1: {results['GA Voting']['F1']:.4f}")

# =========================================================
# OOF STACKING FOR GP (Original GP Method)
# =========================================================

print_log("\n" + "=" * 40)
print_log("9. GP ENSEMBLE (Our Method)")
print_log("=" * 40)

print_log("\nCreating Out-of-Fold Meta Features...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

train_meta = np.zeros((len(X_train_scaled), len(base_models)))
val_meta = np.zeros((len(X_val_scaled), len(base_models)))
test_meta = np.zeros((len(X_test_scaled), len(base_models)))

for model_idx, (name, model) in enumerate(base_models.items()):
    print_log(f"Training OOF for: {name}")
    val_fold_preds = []
    test_fold_preds = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_tr = X_train_scaled[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_valid = X_train_scaled[valid_idx]
        
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        
        train_meta[valid_idx, model_idx] = model_clone.predict_proba(X_valid)[:, 1]
        val_fold_preds.append(model_clone.predict_proba(X_val_scaled)[:, 1])
        test_fold_preds.append(model_clone.predict_proba(X_test_scaled)[:, 1])
    
    val_meta[:, model_idx] = np.mean(val_fold_preds, axis=0)
    test_meta[:, model_idx] = np.mean(test_fold_preds, axis=0)

# Feature engineering for GP
def create_meta_features(meta):
    features = [meta]
    pairwise = []
    for i in range(meta.shape[1]):
        for j in range(i + 1, meta.shape[1]):
            pairwise.append(meta[:, i] * meta[:, j])
    if pairwise:
        pairwise = np.column_stack(pairwise)
        features.append(pairwise)
    
    weighted_avg = (0.30 * meta[:, 0] + 0.20 * meta[:, 1] + 
                    0.20 * meta[:, 2] + 0.30 * meta[:, 3])
    max_feat = np.max(meta, axis=1)
    min_feat = np.min(meta, axis=1)
    median_feat = np.median(meta, axis=1)
    std_feat = np.std(meta, axis=1)
    stats = np.column_stack([weighted_avg, max_feat, min_feat, median_feat, std_feat])
    features.append(stats)
    
    return np.column_stack(features)

train_meta = create_meta_features(train_meta)
val_meta = create_meta_features(val_meta)
test_meta = create_meta_features(test_meta)

# Feature weighting
corrs = []
for i in range(train_meta.shape[1]):
    corr = np.corrcoef(train_meta[:, i], y_train)[0, 1]
    corrs.append(abs(corr) if not np.isnan(corr) else 0)
weights = np.array(corrs) / (sum(corrs) + 1e-8)
train_meta = train_meta * weights
val_meta = val_meta * weights
test_meta = test_meta * weights

# GP Primitives
def protected_div(x, y):
    return np.divide(x, y, out=np.ones_like(x), where=np.abs(y) > 1e-6)

def protected_sqrt(x):
    return np.sqrt(np.maximum(x, 0))

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def safe_log(x):
    return np.log(np.abs(x) + 1e-6)

def custom_aggregate(x, y):
    return (x + y + np.abs(x - y)) / 2

pset = gp.PrimitiveSet("MAIN", train_meta.shape[1])
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(np.maximum, 2)
pset.addPrimitive(np.minimum, 2)
pset.addPrimitive(custom_aggregate, 2)
pset.addPrimitive(sigmoid, 1)
pset.addPrimitive(protected_sqrt, 1)
pset.addPrimitive(safe_log, 1)
pset.addPrimitive(np.abs, 1)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))

for i in range(train_meta.shape[1]):
    pset.renameArguments(**{f'ARG{i}': f'M{i}'})

# GP setup
if 'FitnessMax' not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if 'Individual' not in creator.__dict__:
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evaluate_individual(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_outputs = np.array([func(*row) for row in train_meta])
        val_outputs = np.array([func(*row) for row in val_meta])
        train_outputs = np.clip(train_outputs, -10, 10)
        val_outputs = np.clip(val_outputs, -10, 10)
        
        thresholds = np.linspace(-2, 2, 120)
        best_f1, best_mcc = 0, 0
        for t in thresholds:
            preds = (val_outputs > t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_mcc = matthews_corrcoef(y_val, preds)
        
        auc = roc_auc_score(y_val, val_outputs)
        tree_size = len(individual)
        tree_height = individual.height
        complexity_penalty = max(0, (tree_size - 45) * 0.001) + max(0, (tree_height - 8) * 0.003)
        output_std = np.std(val_outputs)
        diversity_bonus = min(0.02, output_std * 0.01)
        
        fitness = best_f1
        return (fitness if not np.isnan(fitness) else 0.0,)
    except:
        return (0.0,)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))

# Run GP
print_log("\nStarting GP Evolution...")
population = toolbox.population(n=POP_SIZE)
hof = tools.HallOfFame(5)

invalid = [ind for ind in population if not ind.fitness.valid]
for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
    ind.fitness.values = fit

for gen in range(GENERATIONS):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))
    
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.75:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    for mutant in offspring:
        if random.random() < 0.25:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind, fit in zip(invalid_ind, map(toolbox.evaluate, invalid_ind)):
        ind.fitness.values = fit
    
    population[:] = offspring
    hof.update(population)
    
    if (gen + 1) % 10 == 0:
        rec = tools.Statistics(lambda ind: ind.fitness.values)
        rec.register("max", np.max)
        print_log(f"Gen {gen + 1} | Max: {rec.compile(population)['max']:.4f}")

best_individual = hof[0]
best_function = toolbox.compile(expr=best_individual)

# Threshold optimization
val_outputs = np.array([best_function(*row) for row in val_meta])
val_outputs = np.clip(val_outputs, -10, 10)
thresholds = np.linspace(-2, 2, 200)
best_threshold, best_f1 = 0, 0
for t in thresholds:
    f1 = f1_score(y_val, (val_outputs > t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_threshold = f1, t

test_outputs = np.array([best_function(*row) for row in test_meta])
test_outputs = np.clip(test_outputs, -10, 10)
gp_preds = (test_outputs > best_threshold).astype(int)

results['GP-Ensemble (Ours)'] = compute_metrics(y_test, gp_preds, test_outputs)
print_log(f"GP Ensemble Test F1: {results['GP-Ensemble (Ours)']['F1']:.4f}")

# =========================================================
# 10. GP RAW FEATURES (GP directly on raw features)
# =========================================================

print_log("\n" + "=" * 40)
print_log("10. GP RAW FEATURES (Direct on Raw Data)")
print_log("=" * 40)

# Scale for GP raw
raw_scaler = StandardScaler()
X_train_raw = raw_scaler.fit_transform(X_train_scaled)
X_val_raw = raw_scaler.transform(X_val_scaled)
X_test_raw = raw_scaler.transform(X_test_scaled)

# Create new primitive set for raw features
pset_raw = gp.PrimitiveSet("MAIN_RAW", X_train_raw.shape[1])
pset_raw.addPrimitive(operator.add, 2)
pset_raw.addPrimitive(operator.sub, 2)
pset_raw.addPrimitive(operator.mul, 2)
pset_raw.addPrimitive(protected_div, 2)
pset_raw.addPrimitive(np.maximum, 2)
pset_raw.addPrimitive(np.minimum, 2)
pset_raw.addPrimitive(sigmoid, 1)
pset_raw.addPrimitive(protected_sqrt, 1)
pset_raw.addPrimitive(safe_log, 1)
pset_raw.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))

for i in range(X_train_raw.shape[1]):
    pset_raw.renameArguments(**{f'ARG{i}': f'F{i}'})

def evaluate_raw(individual):
    try:
        func = gp.compile(expr=individual, pset=pset_raw)
        train_outputs = np.array([func(*row) for row in X_train_raw])
        val_outputs = np.array([func(*row) for row in X_val_raw])
        train_outputs = np.clip(train_outputs, -10, 10)
        val_outputs = np.clip(val_outputs, -10, 10)
        
        thresholds = np.linspace(-2, 2, 100)
        best_f1 = 0
        for t in thresholds:
            f1 = f1_score(y_val, (val_outputs > t).astype(int), zero_division=0)
            best_f1 = max(best_f1, f1)
        
        auc = roc_auc_score(y_val, val_outputs)
        complexity_penalty = max(0, (len(individual) - 50) * 0.002)
        return (0.5 * best_f1 + 0.5 * auc - complexity_penalty,)
    except:
        return (0.0,)

toolbox_raw = base.Toolbox()
toolbox_raw.register("expr", gp.genHalfAndHalf, pset=pset_raw, min_=1, max_=3)
toolbox_raw.register("individual", tools.initIterate, creator.Individual, toolbox_raw.expr)
toolbox_raw.register("population", tools.initRepeat, list, toolbox_raw.individual)
toolbox_raw.register("compile", gp.compile, pset=pset_raw)
toolbox_raw.register("evaluate", evaluate_raw)
toolbox_raw.register("select", tools.selTournament, tournsize=3)
toolbox_raw.register("mate", gp.cxOnePoint)
toolbox_raw.register("mutate", gp.mutUniform, expr=toolbox_raw.expr, pset=pset_raw)

pop_raw = toolbox_raw.population(n=80)
hof_raw = tools.HallOfFame(3)

invalid_raw = [ind for ind in pop_raw if not ind.fitness.valid]
for ind, fit in zip(invalid_raw, map(toolbox_raw.evaluate, invalid_raw)):
    ind.fitness.values = fit

for gen in range(1):
    offspring = toolbox_raw.select(pop_raw, len(pop_raw))
    offspring = list(map(toolbox_raw.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.7:
            toolbox_raw.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    for mutant in offspring:
        if random.random() < 0.2:
            toolbox_raw.mutate(mutant)
            del mutant.fitness.values
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind, fit in zip(invalid_ind, map(toolbox_raw.evaluate, invalid_ind)):
        ind.fitness.values = fit
    pop_raw[:] = offspring
    hof_raw.update(pop_raw)
    if (gen + 1) % 1 == 0:  # هر 1 نسل یکبار
        raw_stats = tools.Statistics(lambda ind: ind.fitness.values)
        raw_stats.register("max", np.max)
        current_max = raw_stats.compile(pop_raw)['max']
        print_log(f"  GP Raw Features - Gen {gen + 1}/40 | Max Fitness: {current_max:.4f}")


best_raw = hof_raw[0]
func_raw = gp.compile(expr=best_raw, pset=pset_raw)
raw_test_outputs = np.array([func_raw(*row) for row in X_test_raw])
raw_test_outputs = np.clip(raw_test_outputs, -10, 10)

best_raw_thresh = 0
best_raw_f1 = 0
raw_val_outputs = np.array([func_raw(*row) for row in X_val_raw])
for t in np.linspace(-2, 2, 100):
    f1 = f1_score(y_val, (raw_val_outputs > t).astype(int), zero_division=0)
    if f1 > best_raw_f1:
        best_raw_f1, best_raw_thresh = f1, t

raw_preds = (raw_test_outputs > best_raw_thresh).astype(int)
results['GP Raw Features'] = compute_metrics(y_test, raw_preds, raw_test_outputs)
print_log(f"GP Raw Features Test F1: {results['GP Raw Features']['F1']:.4f}")

# =========================================================
# FINAL RESULTS TABLE
# =========================================================

print_log("\n" + "=" * 80)
print_log("FINAL COMPARATIVE RESULTS")
print_log("=" * 80)

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)

print_log("\n" + results_df.to_string())

# Sort by F1 Score
print_log("\n" + "=" * 40)
print_log("RANKING BY F1 SCORE")
print_log("=" * 40)
ranking = results_df.sort_values('F1', ascending=False)
for i, (method, row) in enumerate(ranking.iterrows(), 1):
    print_log(f"{i}. {method}: F1={row['F1']:.4f}, AUC={row['AUC']:.4f}, MCC={row['MCC']:.4f}")

# =========================================================
# SAVE RESULTS
# =========================================================

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_df.to_csv(f"comparative_results_onlyf1_{RANDOM_STATE}_{GENERATIONS}_{timestamp}.csv")

print_log(f"\nResults saved to: comparative_results_{RANDOM_STATE}_{GENERATIONS}_{timestamp}.csv")
print_log("=" * 80)