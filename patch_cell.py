import json

with open(r'D:\ml_project_2\project2\main.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

new_source = r"""from itertools import product as _product
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
#  TASK 2 - JOINT HYPER-PARAMETER SEARCH
#
#  Searches over feature engineering params AND model params simultaneously.
#  Uses a single stratified 80/20 train/val split - fast but noisier than CV.
#  Results are ranked by weighted F1 on the val set.
#
#  Estimated runtime (CPU):  ~2-4 min
# =============================================================================

# -- Feature engineering search grids -----------------------------------------
T2_SEARCH_RESNET_PCA  = [n_90, n_95, n_99]
T2_SEARCH_HIST_FACTOR = [1, 2, 6]
T2_SEARCH_ADD_PCA     = [None, 10, 15]

# -- Model search grids --------------------------------------------------------
T2_SEARCH_LOGREG = [
    {'C': 0.1},
    {'C': 1},
    {'C': 10},
]

T2_SEARCH_SVM = [
    {'kernel': 'linear', 'C': 0.1},
    {'kernel': 'linear', 'C': 1},
    {'kernel': 'linear', 'C': 10},
    {'kernel': 'rbf',    'C': 1,   'gamma': 'scale'},
    {'kernel': 'rbf',    'C': 10,  'gamma': 'scale'},
    {'kernel': 'rbf',    'C': 100, 'gamma': 'scale'},
    {'kernel': 'rbf',    'C': 10,  'gamma': 'auto'},
]

T2_SEARCH_RF = [
    {'n_estimators': 200, 'max_features': 'sqrt'},
    {'n_estimators': 300, 'max_features': 'sqrt'},
    {'n_estimators': 300, 'max_features': 'log2'},
]

T2_SEARCH_XGB = [
    {'n_estimators': 100, 'max_depth': 6,  'learning_rate': 0.1},
    {'n_estimators': 200, 'max_depth': 4,  'learning_rate': 0.1},
    {'n_estimators': 300, 'max_depth': 4,  'learning_rate': 0.05},
]

T2_SEARCH_MLP = [
    {'hidden': (256, 128),      'activation': nn.GELU, 'dropout': 0.2, 'lr': 1e-3},
    {'hidden': (512, 256),      'activation': nn.GELU, 'dropout': 0.2, 'lr': 1e-3},
    {'hidden': (256, 128),      'activation': nn.ReLU, 'dropout': 0.1, 'lr': 3e-4},
]
# -----------------------------------------------------------------------------

# -- Pre-load raw CSVs once ---------------------------------------------------
_s_col_tr = _load_and_align(r'Data\task2_data\color_histogram.csv',    train_ids)
_s_hog_tr = _load_and_align(r'Data\task2_data\hog_pca.csv',            train_ids)
_s_add_tr = _load_and_align(r'Data\task2_data\additional_features.csv', train_ids)

# -- Pre-compute ResNet PCA projections ---------------------------------------
print('Pre-computing ResNet PCA projections...')
_resnet_cache: dict[int, np.ndarray] = {}
for _nc in T2_SEARCH_RESNET_PCA:
    _p = PCA(n_components=_nc, random_state=42)
    _resnet_cache[_nc] = _p.fit_transform(X_t2).astype(np.float64)
    print(f'  n_components={_nc}: {_resnet_cache[_nc].shape[1]}-d')

# -- Single stratified 80/20 split -------------------------------------------
# Reuse the same fold-0 indices for every config so comparisons are fair
_tr_idx, _va_idx = next(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_t2, y_t2)
)
print(f'\nTrain split: {len(_tr_idx)} samples  |  Val split: {len(_va_idx)} samples\n')

def _build_search_features(n_comp: int, hist_factor: int, add_pca_n) -> np.ndarray:
    """Build the full combined feature matrix (all 417 rows) for a given config.
    Scalers and PCA are always fit on the train split only to avoid leakage."""
    Xtr = _resnet_cache[n_comp]                              # ResNet PCA (as-is)

    # Colour histogram
    col    = _reduce_histogram(_s_col_tr.copy(), hist_factor)
    sc_col = StandardScaler().fit(col[_tr_idx])              # fit on train split
    Xtr    = np.hstack([Xtr, sc_col.transform(col)])

    # HOG PCA (as-is, already PCA-reduced)
    Xtr = np.hstack([Xtr, _s_hog_tr.astype(np.float64)])

    # Additional features
    sc_add = StandardScaler().fit(_s_add_tr[_tr_idx])        # fit on train split
    add_s  = sc_add.transform(_s_add_tr)
    if add_pca_n is not None:
        pa    = _PCA(n_components=add_pca_n, random_state=42)
        pa.fit(add_s[_tr_idx])                               # fit PCA on train split
        add_s = pa.transform(add_s)
    Xtr = np.hstack([Xtr, add_s])

    return Xtr  # shape: (417, total_dims)

# -- Main search loop ---------------------------------------------------------
_feat_configs = list(_product(T2_SEARCH_RESNET_PCA, T2_SEARCH_HIST_FACTOR, T2_SEARCH_ADD_PCA))
_n_model_cfgs = (len(T2_SEARCH_LOGREG) + len(T2_SEARCH_SVM) + len(T2_SEARCH_RF)
                 + len(T2_SEARCH_XGB)  + len(T2_SEARCH_MLP))
_total = len(_feat_configs) * _n_model_cfgs

print(f'{len(_feat_configs)} feature configs x {_n_model_cfgs} model configs = {_total} evaluations\n')

_search_results: list[dict] = []

def _record_result(model, params, n_comp, hist_f, add_n, dims, f1_val):
    _search_results.append({
        'model':       model,
        'params':      params,
        'resnet_pca':  n_comp,
        'hist_factor': hist_f,
        'add_pca':     str(add_n),
        'dims':        dims,
        'f1':          float(f1_val),
    })

_pbar = tqdm.tqdm(total=_total, desc='Search')

for _nc, _hf, _an in _feat_configs:
    _X     = _build_search_features(_nc, _hf, _an)
    _Xtr_s = _X[_tr_idx]
    _Xva_s = _X[_va_idx]
    _ytr_s = y_t2[_tr_idx]
    _yva_s = y_t2[_va_idx]
    _d     = _X.shape[1]

    # -- Logistic Regression --------------------------------------------------
    for p in T2_SEARCH_LOGREG:
        clf = LogisticRegression(max_iter=2000, **p)
        clf.fit(_Xtr_s, _ytr_s)
        _record_result('LogReg', f"C={p['C']}", _nc, _hf, _an, _d,
                       f1_score(_yva_s, clf.predict(_Xva_s), average='weighted'))
        _pbar.update(1)

    # -- SVM ------------------------------------------------------------------
    for p in T2_SEARCH_SVM:
        clf = SVC(decision_function_shape='ovr', **p)
        clf.fit(_Xtr_s, _ytr_s)
        g = f" gamma={p['gamma']}" if 'gamma' in p else ''
        _record_result('SVM', f"kernel={p['kernel']} C={p['C']}{g}", _nc, _hf, _an, _d,
                       f1_score(_yva_s, clf.predict(_Xva_s), average='weighted'))
        _pbar.update(1)

    # -- Random Forest --------------------------------------------------------
    for p in T2_SEARCH_RF:
        clf = RandomForestClassifier(random_state=42, n_jobs=-1, **p)
        clf.fit(_Xtr_s, _ytr_s)
        _record_result('RF', f"n={p['n_estimators']} feat={p['max_features']}", _nc, _hf, _an, _d,
                       f1_score(_yva_s, clf.predict(_Xva_s), average='weighted'))
        _pbar.update(1)

    # -- XGBoost --------------------------------------------------------------
    for p in T2_SEARCH_XGB:
        clf = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', verbosity=0, **p)
        clf.fit(_Xtr_s, _ytr_s)
        _record_result('XGBoost', f"n={p['n_estimators']} depth={p['max_depth']} lr={p['learning_rate']}",
                       _nc, _hf, _an, _d,
                       f1_score(_yva_s, clf.predict(_Xva_s), average='weighted'))
        _pbar.update(1)

    # -- MLP ------------------------------------------------------------------
    for p in T2_SEARCH_MLP:
        _, m = train_mlp(_Xtr_s, _ytr_s, _Xva_s, _yva_s,
                         hidden=p['hidden'], activation=p['activation'],
                         dropout=p['dropout'], lr=p['lr'], epochs=300)
        act_name = p['activation'].__name__
        _record_result('MLP', f"hidden={p['hidden']} act={act_name} drop={p['dropout']}",
                       _nc, _hf, _an, _d, m['f1'])
        _pbar.update(1)

_pbar.close()

# -- Display top 10 -----------------------------------------------------------
_res_df = (pd.DataFrame(_search_results)
             .sort_values('f1', ascending=False)
             .reset_index(drop=True))

_W = 108
print(f'\n{"="*_W}')
print(f'  TOP 10  --  {len(_search_results)} configs evaluated, ranked by weighted F1 (single val split)')
print(f'{"="*_W}')
print(f'  {"#":<3}  {"Model":<8}  {"F1":<8}  {"Model params":<38}  {"PCA":<6}  {"Hist":<5}  {"AddPCA":<7}  Dims')
print(f'  {"-"*(_W-2)}')
for _rank, _row in _res_df.head(10).iterrows():
    print(
        f'  {_rank+1:<3}  '
        f'{_row["model"]:<8}  '
        f'{_row["f1"]:.4f}    '
        f'{_row["params"]:<38}  '
        f'{_row["resnet_pca"]:<6}  '
        f'{_row["hist_factor"]:<5}  '
        f'{_row["add_pca"]:<7}  '
        f'{_row["dims"]}'
    )
print(f'{"="*_W}')
print(f'\nFull results in _res_df  --  e.g. _res_df[_res_df.model == "SVM"].head(5)')"""

for cell in nb['cells']:
    if cell.get('id') == '69b6167a':
        cell['source'] = new_source
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Found and updated cell 69b6167a')
        break
else:
    print('ERROR: cell not found')

with open(r'D:\ml_project_2\project2\main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Saved.')
