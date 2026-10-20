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
    RobustScaler
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
    GradientBoostingClassifier
)

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from deap import base, creator, tools, gp

import operator
import random
import logging
from datetime import datetime
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
logfile = f"GP_Research_Grade_Onlyf1_Final_randomstat {RANDOM_STATE}_Generation {GENERATIONS}_{now.strftime('%Y_%m_%d_%H_%M_%S')}.log"

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
print_log("RESEARCH-GRADE GP ENSEMBLE FOR PHISHING DETECTION")
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
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.1765,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

# 70 / 15 / 15

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
        n_estimators=400,
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    'dt': DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),

    'gb': GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        random_state=RANDOM_STATE
    ),

    'xgb': XGBClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0
    )
}

# =========================================================
# OOF STACKING (NO LEAKAGE)
# =========================================================

print_log("\nCreating Out-of-Fold Meta Features...")

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

train_meta = np.zeros((len(X_train_scaled), len(base_models)))
val_meta = np.zeros((len(X_val_scaled), len(base_models)))
test_meta = np.zeros((len(X_test_scaled), len(base_models)))

model_names = list(base_models.keys())

for model_idx, (name, model) in enumerate(base_models.items()):

    print_log(f"Training OOF for: {name}")

    val_fold_preds = []
    test_fold_preds = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_scaled, y_train)):

        X_tr = X_train_scaled[train_idx]
        y_tr = y_train.iloc[train_idx]

        X_valid = X_train_scaled[valid_idx]

        model.fit(X_tr, y_tr)

        # OOF predictions
        train_meta[valid_idx, model_idx] = model.predict_proba(X_valid)[:, 1]

        # Validation predictions
        val_preds = model.predict_proba(X_val_scaled)[:, 1]
        val_fold_preds.append(val_preds)

        # Test predictions
        test_preds = model.predict_proba(X_test_scaled)[:, 1]
        test_fold_preds.append(test_preds)

    # Average folds
    val_meta[:, model_idx] = np.mean(val_fold_preds, axis=0)
    test_meta[:, model_idx] = np.mean(test_fold_preds, axis=0)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

print_log("\nCreating Advanced Meta Features...")


def create_meta_features(meta):

    features = [meta]

    # Pairwise products
    pairwise = []

    for i in range(meta.shape[1]):
        for j in range(i + 1, meta.shape[1]):
            pairwise.append(meta[:, i] * meta[:, j])

    pairwise = np.column_stack(pairwise)
    features.append(pairwise)

    # Statistics
    weighted_avg = (
        0.30 * meta[:, 0] +
        0.20 * meta[:, 1] +
        0.20 * meta[:, 2] +
        0.30 * meta[:, 3]
    )

    max_feat = np.max(meta, axis=1)
    min_feat = np.min(meta, axis=1)
    median_feat = np.median(meta, axis=1)
    std_feat = np.std(meta, axis=1)

    stats = np.column_stack([
        weighted_avg,
        max_feat,
        min_feat,
        median_feat,
        std_feat
    ])

    features.append(stats)

    return np.column_stack(features)


train_meta = create_meta_features(train_meta)
val_meta = create_meta_features(val_meta)
test_meta = create_meta_features(test_meta)

print_log(f"Meta features shape: {train_meta.shape}")

# =========================================================
# FEATURE WEIGHTING
# =========================================================

corrs = []

for i in range(train_meta.shape[1]):

    corr = np.corrcoef(train_meta[:, i], y_train)[0, 1]

    if np.isnan(corr):
        corr = 0

    corrs.append(abs(corr))

weights = np.array(corrs)
weights = weights / (weights.sum() + 1e-8)

train_meta = train_meta * weights
val_meta = val_meta * weights
test_meta = test_meta * weights

# =========================================================
# GP PRIMITIVES
# =========================================================

print_log("\nSetting up Genetic Programming...")


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

# Binary operators
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(np.maximum, 2)
pset.addPrimitive(np.minimum, 2)
pset.addPrimitive(custom_aggregate, 2)

# Unary operators
pset.addPrimitive(sigmoid, 1)
pset.addPrimitive(protected_sqrt, 1)
pset.addPrimitive(safe_log, 1)
pset.addPrimitive(np.abs, 1)

# Constants
pset.addEphemeralConstant(
    "rand101",
    lambda: random.uniform(-1, 1)
)

# Rename arguments
for i in range(train_meta.shape[1]):
    pset.renameArguments(**{f'ARG{i}': f'M{i}'})

# =========================================================
# GP CLASSES
# =========================================================

if 'FitnessMax' not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

if 'Individual' not in creator.__dict__:
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# =========================================================
# TOOLBOX
# =========================================================

toolbox = base.Toolbox()

toolbox.register(
    "expr",
    gp.genHalfAndHalf,
    pset=pset,
    min_=1,
    max_=4
)

toolbox.register(
    "individual",
    tools.initIterate,
    creator.Individual,
    toolbox.expr
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)

toolbox.register("compile", gp.compile, pset=pset)

# =========================================================
# FITNESS FUNCTION
# =========================================================

print_log("\nCreating Fitness Function...")


def evaluate_individual(individual):

    try:

        func = toolbox.compile(expr=individual)

        train_outputs = np.array([
            func(*row)
            for row in train_meta
        ])

        val_outputs = np.array([
            func(*row)
            for row in val_meta
        ])

        train_outputs = np.clip(train_outputs, -10, 10)
        val_outputs = np.clip(val_outputs, -10, 10)

        # Threshold search on validation only
        thresholds = np.linspace(-2, 2, 120)

        best_f1 = 0
        best_mcc = 0

        for t in thresholds:

            preds = (val_outputs > t).astype(int)

            f1 = f1_score(y_val, preds, zero_division=0)
            mcc = matthews_corrcoef(y_val, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_mcc = mcc

        auc = roc_auc_score(y_val, val_outputs)

        # Complexity penalty
        tree_size = len(individual)
        tree_height = individual.height

        complexity_penalty = 0

        if tree_size > 45:
            complexity_penalty += (tree_size - 45) * 0.001

        if tree_height > 8:
            complexity_penalty += (tree_height - 8) * 0.003

        # Diversity bonus
        output_std = np.std(val_outputs)
        diversity_bonus = min(0.02, output_std * 0.01)

        # Final fitness
        fitness = best_f1
        if np.isnan(fitness):
            return (0.0,)

        return (fitness,)

    except:
        return (0.0,)


toolbox.register("evaluate", evaluate_individual)

# =========================================================
# EVOLUTION OPERATORS
# =========================================================

toolbox.register(
    "select",
    tools.selTournament,
    tournsize=5
)

toolbox.register("mate", gp.cxOnePoint)


toolbox.register(
    "expr_mut",
    gp.genFull,
    min_=0,
    max_=2
)


toolbox.register(
    "mutate",
    gp.mutUniform,
    expr=toolbox.expr_mut,
    pset=pset
)

# Height limits

toolbox.decorate(
    "mate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=MAX_TREE_HEIGHT
    )
)


toolbox.decorate(
    "mutate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=MAX_TREE_HEIGHT
    )
)

# =========================================================
# RUN GP
# =========================================================

print_log("\nStarting Evolution...")

population = toolbox.population(n=POP_SIZE)

hof = tools.HallOfFame(5)

stats = tools.Statistics(lambda ind: ind.fitness.values)

stats.register("avg", np.mean)
stats.register("max", np.max)
stats.register("std", np.std)

# Initial evaluation
invalid = [ind for ind in population if not ind.fitness.valid]

fitnesses = map(toolbox.evaluate, invalid)

for ind, fit in zip(invalid, fitnesses):
    ind.fitness.values = fit

for gen in range(GENERATIONS):

    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        if random.random() < 0.75:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Mutation
    for mutant in offspring:

        if random.random() < 0.25:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate invalid
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

    fitnesses = map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

    hof.update(population)

    record = stats.compile(population)

    print_log(
        f"Gen {gen + 1} | "
        f"Avg: {record['avg']:.4f} | "
        f"Max: {record['max']:.4f}"
    )

# =========================================================
# BEST INDIVIDUAL
# =========================================================

best_individual = hof[0]

print_log("\nBEST GP EXPRESSION")
print_log("=" * 80)
print_log(str(best_individual))
print_log(f"Tree Height: {best_individual.height}")
print_log(f"Tree Size: {len(best_individual)}")
print_log(f"Fitness: {best_individual.fitness.values[0]:.4f}")

best_function = toolbox.compile(expr=best_individual)

# =========================================================
# THRESHOLD OPTIMIZATION
# =========================================================

print_log("\nOptimizing Threshold...")

val_outputs = np.array([
    best_function(*row)
    for row in val_meta
])

val_outputs = np.clip(val_outputs, -10, 10)

thresholds = np.linspace(-2, 2, 200)

best_threshold = 0
best_f1 = 0

for t in thresholds:

    preds = (val_outputs > t).astype(int)

    f1 = f1_score(y_val, preds, zero_division=0)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print_log(f"Best Threshold: {best_threshold:.4f}")
print_log(f"Validation F1: {best_f1:.4f}")

# =========================================================
# FINAL TEST EVALUATION
# =========================================================

print_log("\nFinal Test Evaluation...")

# Retrain base models on train+val

X_final = np.vstack([X_train_scaled, X_val_scaled])
y_final = pd.concat([y_train, y_val])

final_results = {}

for name, model in base_models.items():

    model.fit(X_final, y_final)

    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    final_results[name] = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1': f1_score(y_test, preds),
        'AUC': roc_auc_score(y_test, proba),
        'MCC': matthews_corrcoef(y_test, preds)
    }

# =========================================================
# REBUILD TEST META FEATURES USING FINAL MODELS
# =========================================================

final_test_meta = np.zeros((len(X_test_scaled), len(base_models)))

for model_idx, (name, model) in enumerate(base_models.items()):
    final_test_meta[:, model_idx] = model.predict_proba(X_test_scaled)[:, 1]

final_test_meta = create_meta_features(final_test_meta)
final_test_meta = final_test_meta * weights

# GP Ensemble

test_outputs = np.array([
    best_function(*row)
    for row in final_test_meta
])

test_outputs = np.clip(test_outputs, -10, 10)

gp_preds = (test_outputs > best_threshold).astype(int)

final_results['GP-Ensemble-Ours'] = {
    'Accuracy': accuracy_score(y_test, gp_preds),
    'Precision': precision_score(y_test, gp_preds),
    'Recall': recall_score(y_test, gp_preds),
    'F1': f1_score(y_test, gp_preds),
    'AUC': roc_auc_score(y_test, test_outputs),
    'MCC': matthews_corrcoef(y_test, gp_preds)
}

results_df = pd.DataFrame(final_results).T

print_log("\nFINAL RESULTS")
print_log("=" * 80)
print_log(results_df.round(4))

# =========================================================
# ABLATION HELPERS
# =========================================================

print_log("\nABLATION SUGGESTIONS")
print_log("=" * 80)
print_log("1. Remove feature weighting")
print_log("2. Remove pairwise interactions")
print_log("3. Replace GP with Logistic Regression")
print_log("4. Use only XGBoost")
print_log("5. Remove complexity penalty")

# =========================================================
# SAVE RESULTS
# =========================================================

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

results_df.to_csv(f"research_results_Onlyf1_Final_randomstat {RANDOM_STATE}_Generation {GENERATIONS}_{timestamp}.csv")

print_log("\nSaved Results Successfully")
print_log("=" * 80)
