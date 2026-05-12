"""
TCO АГЕНТ v4.1 — оптимизированный размер моделей, готовность к деплою.
"""
import os, warnings, pickle, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_percentage_error, r2_score,
                             mean_absolute_error, ndcg_score)
import xgboost as xgb

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 8)

# ═══════════════ НАСТРОЙКИ ═══════════════
SOBSTV_FILE = "data/sobstvennost.csv"
ARENDA_FILE = "data/arenda.csv"

TAX_RATE, DISCOUNT_RATE = 0.20, 0.10
DEFAULT_T = 10
VAT_RATE, OPEX_RATE, CAPEX_RATE, DEPR_RATE, CS_RATE, GROWTH_RATE = 0.20, 0.03, 0.01, 0.02, 0.05, 0.02

RANDOM_STATE, TEST_SIZE, MIN_COST, TOP_K = 42, 0.20, 100, 10

DEMO_QUERY = {
    'region': '66', 'segmentt': 'Ритейл',
    'obsh_plosh': (100, 400), 'quan': (1, 10),
    'building_age': (0, 40), 'first_line': (1, 1),
    'tco_pred_ths': (0, 50000),
}
QUERY_WEIGHTS = {
    'region': 0.25, 'segmentt': 0.25, 'obsh_plosh': 0.15,
    'quan': 0.05, 'building_age': 0.10, 'first_line': 0.02,
    'tco_pred_ths': 0.18,
}

# ---------- Утилиты ----------
def safe_numeric(series):
    s = series.astype(str).str.strip().str.replace(' ', '', regex=False)
    s = s.str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')

def load_and_clean(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, sep=None, engine='python', encoding='cp1251')
    print(f"[1] Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")
    df = df.replace(['-', '--', 'null', 'NULL', 'None', 'nan', 'NaN', ''], np.nan)
    for col in df.columns:
        if df[col].dtype == 'object':
            s = df[col].astype(str).str.strip().str.replace(' ', '', regex=False).str.replace(',', '.', regex=False)
            s = s.replace(['-', '--', 'null', 'NULL', 'None'], np.nan)
            converted = pd.to_numeric(s, errors='coerce')
            if converted.notna().sum() > len(df) * 0.1:
                df[col] = converted
    return df

# ---------- Подготовка данных ----------
def prepare_cost_data_own(df):
    cost_cols = ['balans_v', 'ostat_v', 'rinoch_v']
    for col in cost_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = safe_numeric(df[col])
    df['Inv'] = df['balans_v'].fillna(df['ostat_v'])
    if 'region' in df.columns and 'tipob' in df.columns:
        group_med = df.groupby(['region', 'tipob'])['Inv'].transform('median')
        df['Inv'] = df['Inv'].fillna(group_med)
    overall_med = df['Inv'].median()
    if pd.isna(overall_med) or overall_med <= 0:
        overall_med = 10_000
    df['Inv'] = df['Inv'].fillna(overall_med)
    df.loc[df['Inv'] <= 0, 'Inv'] = overall_med
    df['Balans_for_Depr'] = df['balans_v'].fillna(df['Inv'])
    df['Balans_for_Depr'] = df['Balans_for_Depr'].clip(lower=MIN_COST / 10)
    return df

def prepare_lease_data(df):
    for col in ['arend_st', 'obsh_plosh', 'perem_ap']:
        if col in df.columns:
            df[col] = safe_numeric(df[col])
    df = df.dropna(subset=['arend_st', 'obsh_plosh'])
    df = df[(df['arend_st'] > 0) & (df['obsh_plosh'] > 0)].copy()
    df['OPEX'] = df['arend_st'] * 12 * df['obsh_plosh'] / (1 + VAT_RATE)
    if 'perem_ap' in df.columns:
        df['T_lease'] = safe_numeric(df['perem_ap']).fillna(DEFAULT_T)
        df.loc[df['T_lease'] <= 0, 'T_lease'] = DEFAULT_T
    else:
        df['T_lease'] = DEFAULT_T
    df = df[df['T_lease'] >= 1]
    df['Inv'] = 0.0
    df['Balans_for_Depr'] = 0.0
    df['CAPEX_annual'] = 0.0
    return df.reset_index(drop=True)

# ---------- TCO ----------
def calculate_tco_own(df):
    Inv, Balans = df['Inv'].values, df['Balans_for_Depr'].values
    r, tax = DISCOUNT_RATE, TAX_RATE
    years = np.arange(1, DEFAULT_T + 1)
    disc = 1 / (1 + r) ** years
    VAT_Inv = Inv * VAT_RATE
    inv_component = Inv + VAT_Inv * (1 - tax)
    OPEX = Inv * OPEX_RATE
    CAPEX = Inv * CAPEX_RATE
    VAT_CAPEX = CAPEX * VAT_RATE
    D = Balans * DEPR_RATE
    annual_cf = OPEX + VAT_CAPEX * (1 - tax) + CAPEX - D * tax
    disc_sum = np.sum(annual_cf[:, np.newaxis] * disc, axis=1)
    FV_T = Inv * (1 + GROWTH_RATE) ** DEFAULT_T
    Cs_T = FV_T * CS_RATE
    term = Cs_T * (1 - tax) / (1 + r) ** DEFAULT_T
    TCO = np.clip(inv_component + disc_sum + term, MIN_COST, None)
    return TCO

def calculate_tco_rent(df):
    r, tax = DISCOUNT_RATE, TAX_RATE
    T_vec = df['T_lease'].values.astype(int)
    OPEX = df['OPEX'].values
    annual_net = OPEX * (1 - tax)
    max_T = int(T_vec.max())
    disc_all = np.array([1 / (1 + r)**t for t in range(1, max_T+1)])
    TCO = np.zeros(len(df))
    for i, T_i in enumerate(T_vec):
        TCO[i] = annual_net[i] * np.sum(disc_all[:T_i])
    return np.clip(TCO, MIN_COST, None)

# ---------- Признаки ----------
def create_tco_features(df):
    data = df.copy()
    if 'validfrom' in data.columns:
        data['validfrom'] = pd.to_datetime(data['validfrom'], errors='coerce')
        data['building_age'] = ((pd.Timestamp.now() - data['validfrom']).dt.days / 365.25).clip(0, 150).fillna(0)
    else:
        data['building_age'] = 0
    area_cols = ['obsh_plosh', 'osnov_plosh', 'vspom_plosh', 'proch_plosh']
    exist_areas = [c for c in area_cols if c in data.columns]
    data['total_area'] = data[exist_areas].fillna(0).sum(axis=1) if exist_areas else 0
    num_cols = ['obsh_plosh', 'osnov_plosh', 'vspom_plosh', 'proch_plosh',
                'plosh_v_ar', 'neis_plosh', 'osn_obsh', 'vspom_obsh',
                'org_pm', 'teh_pm', 'k4_koef', 'k4_koef_ch', 'irrate',
                'quan', 'building_age', 'total_area',
                'zz_lattitude', 'zz_longitude']
    num_cols = [c for c in num_cols if c in data.columns]
    for c in num_cols:
        data[c] = safe_numeric(data[c])
    X_num = data[num_cols].fillna(0).astype(float)
    cat_cols = ['tipob', 'priznas', 'region', 'segmentt', 'subsegmentt',
                'usgfunction', 'elorgst_text', 'statusesq', 'zz_priznas',
                'xmusgfunction', 'vidpr']
    cat_cols = [c for c in cat_cols if c in data.columns]
    X_cat = data[cat_cols].fillna('NA')
    for col in X_cat.columns:
        le = LabelEncoder()
        X_cat[col] = le.fit_transform(X_cat[col].astype(str))
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    for c in X.columns:
        if X[c].dtype == 'object':
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
    return X

def compute_distance_to_region_center(df):
    coords = df[['region', 'zz_lattitude', 'zz_longitude']].copy()
    for col in ['zz_lattitude', 'zz_longitude']:
        coords[col] = safe_numeric(coords[col])
        coords[col] = coords.groupby('region')[col].transform(lambda x: x.fillna(x.median()))
    coords['zz_lattitude'] = coords['zz_lattitude'].fillna(coords['zz_lattitude'].median())
    coords['zz_longitude'] = coords['zz_longitude'].fillna(coords['zz_longitude'].median())
    centers = coords.groupby('region')[['zz_lattitude', 'zz_longitude']].median()
    centers.columns = ['center_lat', 'center_lon']
    df_with_center = coords.join(centers, on='region')
    lat_diff = df_with_center['zz_lattitude'] - df_with_center['center_lat']
    lon_diff = df_with_center['zz_longitude'] - df_with_center['center_lon']
    return np.sqrt(lat_diff**2 + lon_diff**2)

def create_ranking_features(df):
    data = df.copy()
    if 'validfrom' in data.columns:
        data['validfrom'] = pd.to_datetime(data['validfrom'], errors='coerce')
        data['building_age'] = ((pd.Timestamp.now() - data['validfrom']).dt.days / 365.25).clip(0, 150).fillna(25)
    else:
        data['building_age'] = 25
    for col in ['irrate', 'k4_koef', 'quan', 'obsh_plosh']:
        data[col] = safe_numeric(data.get(col, 0)).fillna(0)
    data['first_line'] = data.get('first_line_house', pd.Series()).apply(lambda v: 1 if str(v).strip() == 'X' else 0)
    cat_cols_rank = ['segmentt', 'subsegmentt', 'xmusgfunction', 'butxt', 'bezei', 'statusesq', 'zz_peref', 'region']
    cat_cols_rank = [c for c in cat_cols_rank if c in data.columns]
    encoders = {}
    for col in cat_cols_rank:
        le = LabelEncoder()
        data[col + '_enc'] = le.fit_transform(data[col].fillna('NA').astype(str))
        encoders[col] = le
    num_rank_cols = ['obsh_plosh', 'building_age', 'irrate', 'k4_koef', 'quan', 'first_line']
    enc_rank_cols = [c + '_enc' for c in cat_cols_rank if c + '_enc' in data.columns]
    all_rank_cols = num_rank_cols + enc_rank_cols
    all_rank_cols = [c for c in all_rank_cols if c in data.columns]
    X_rank = data[all_rank_cols].fillna(0).astype(float)
    return X_rank, encoders, all_rank_cols, data[['region', 'segmentt'] + [c for c in ['obsh_plosh', 'building_age', 'quan', 'first_line'] if c in data.columns]].copy()

def build_relevance_scores(df_raw, tco_pred):
    area = safe_numeric(df_raw.get('obsh_plosh', 0)).fillna(0).clip(lower=0)
    log_area = np.log1p(area.values)
    area_max = log_area.max() if log_area.max() > 0 else 1.0
    size_norm = log_area / area_max
    dist = compute_distance_to_region_center(df_raw)
    max_dist = dist.max() if dist.max() > 0 else 1.0
    dist_norm = 1.0 - np.clip(dist.values / max_dist, 0.0, 1.0)
    tco_safe = np.clip(tco_pred, 1, None)
    tco_min, tco_max = tco_safe.min(), tco_safe.max()
    tco_norm = 1.0 - (tco_safe - tco_min) / (tco_max - tco_min + 1e-9)
    relevance = 0.35 * size_norm + 0.35 * dist_norm + 0.30 * tco_norm
    relevance = np.clip(relevance, 0.0, 1.0)
    df_raw['relevance'] = relevance
    return df_raw

# ---------- XGBoost ----------
def train_xgboost_tco(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    y_train_log = np.log1p(y_train)
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [5, 6, 7],
        'learning_rate': [0.05, 0.07],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'reg_alpha': [0.5, 1.0],
        'reg_lambda': [2.0, 5.0],
        'min_child_weight': [3, 5],
        'gamma': [0.1, 0.3],
    }
    base = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    grid = GridSearchCV(base, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid.fit(X_train_sc, y_train_log)
    best = grid.best_estimator_
    y_pred_log = best.predict(X_test_sc)
    y_pred = np.expm1(y_pred_log)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"XGBoost R²: {r2:.4f}") 
    #print(MAPE: {mape*100:.2f}%, MAE: {mae:.0f}"))
    return best, scaler, X_test, y_test, y_pred, X.columns.tolist()

# ---------- RF ранкер (оптимизирован по размеру) ----------
def train_rf_ranker(X_rank, relevance, feature_names):
    N = len(X_rank)
    idx_all = np.arange(N)
    train_idx, test_idx = train_test_split(idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train = X_rank.iloc[train_idx].reset_index(drop=True)
    y_train = relevance[train_idx]
    rf_param_grid = {
        'n_estimators': [50, 100, 150],       # сильно уменьшено
        'max_depth': [5, 8, 12],               # ограниченная глубина
        'min_samples_leaf': [3, 5],            # больше листьев = меньше размер
        'max_features': ['sqrt', 0.7],
    }
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    cv_r2 = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='r2')
    print(f"RF CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    best_rf.fit(X_train, y_train)
    return best_rf, test_idx

# ---------- Запрос и ранжирование ----------
def compute_query_similarity(row, query, weights):
    total_w = 0.0
    score = 0.0
    for param, desired in query.items():
        if param not in row.index:
            continue
        w = weights.get(param, 0.0)
        if w == 0.0:
            continue
        total_w += w
        val = row[param]
        if pd.isna(val):
            continue
        if isinstance(desired, tuple) and len(desired) == 2:
            lo, hi = float(desired[0]), float(desired[1])
            v = float(val)
            if lo <= v <= hi:
                s = 1.0
            else:
                dist_val = max(lo - v, v - hi)
                span = max(hi - lo, 1.0)
                s = float(np.exp(-5.0 * dist_val / span))
        else:
            s = 1.0 if str(val).strip() == str(desired).strip() else 0.0
        score += w * s
    return score / max(total_w, 1e-9)

def rank_objects(df_rank_features, rf_ranker, tco_pred, relevance_full,
                 feature_names, df_extra, query=DEMO_QUERY,
                 weights=QUERY_WEIGHTS, top_k=TOP_K, test_idx=None):
    rf_base = rf_ranker.predict(df_rank_features)
    rf_base_norm = (rf_base - rf_base.min()) / (rf_base.max() - rf_base.min() + 1e-9)
    df_out = df_rank_features.copy()
    df_out['tco_pred_ths'] = tco_pred
    df_out['rf_base_score'] = rf_base_norm
    for col in df_extra.columns:
        df_out[col] = df_extra[col].values
    df_out['query_sim'] = df_out.apply(lambda r: compute_query_similarity(r, query, weights), axis=1)
    df_out['final_score'] = 0.7 * df_out['query_sim'] + 0.3 * df_out['rf_base_score']
    df_out = df_out.sort_values('final_score', ascending=False)
    top = df_out.head(top_k).copy()
    if test_idx is not None and len(test_idx) >= 2:
        test_df = df_out.iloc[test_idx]
        test_df['relevance'] = relevance_full[test_idx]
        ndcg = ndcg_score(test_df['relevance'].values.reshape(1, -1), test_df['final_score'].values.reshape(1, -1), k=min(top_k, len(test_df)-1))
    else:
        ndcg = 1.0
    print(f"NDCG@{top_k}: {ndcg:.4f}")
    return top, ndcg, df_out

# ============ ГЛАВНЫЙ ПАЙПЛАЙН ============
def run_pipeline(mode='own', save_artifacts=False):
    if mode == 'own':
        filepath = SOBSTV_FILE
        prep_func = prepare_cost_data_own
        tco_func = calculate_tco_own
    elif mode == 'rent':
        filepath = ARENDA_FILE
        prep_func = prepare_lease_data
        tco_func = calculate_tco_rent
    else:
        raise ValueError("mode должен быть 'own' или 'rent'")

    df = load_and_clean(filepath)
    df = prep_func(df)
    tco = tco_func(df)
    X_tco = create_tco_features(df)
    xgb_model, scaler, X_test, y_test, y_pred_test, feat_names = train_xgboost_tco(X_tco, tco)
    X_all_sc = scaler.transform(X_tco)
    tco_pred_all = np.expm1(xgb_model.predict(X_all_sc))
    tco_pred_all = np.clip(tco_pred_all, MIN_COST, None)
    df = build_relevance_scores(df, tco_pred_all)
    X_rank, encoders, rank_feat_names, df_extra = create_ranking_features(df)
    relevance = df['relevance'].values
    rf_ranker, test_idx = train_rf_ranker(X_rank, relevance, rank_feat_names)
    top_objects, ndcg, df_ranked = rank_objects(
        X_rank, rf_ranker, tco_pred_all, relevance,
        rank_feat_names, df_extra, query=DEMO_QUERY, weights=QUERY_WEIGHTS,
        top_k=TOP_K, test_idx=test_idx)

    if save_artifacts:
        os.makedirs('models', exist_ok=True)
        # Сохраняем каталог (pkl)
        df_ranked.to_pickle(f'models/{mode}_catalog.pkl')
        # Сохраняем XGBoost и scaler
        joblib.dump(xgb_model, f'models/xgb_model_{mode}.joblib', compress=3)
        joblib.dump(scaler, f'models/scaler_{mode}.joblib', compress=3)
        # Сохраняем RF ранкер (сжатый)
        joblib.dump(rf_ranker, f'models/rf_ranker_{mode}.joblib', compress=3)
        # Сохраняем энкодеры и список признаков
        with open(f'models/encoders_{mode}.pkl', 'wb') as f:
            pickle.dump(encoders, f)
        with open(f'models/rank_feat_names_{mode}.pkl', 'wb') as f:
            pickle.dump(rank_feat_names, f)
        print(f"Артефакты для режима '{mode}' сохранены в models/")

    return {
        'df': df, 'tco': tco,
        'xgb_model': xgb_model, 'scaler': scaler,
        'rf_ranker': rf_ranker, 'rank_feat_names': rank_feat_names,
        'encoders': encoders, 'X_rank': X_rank,
        'tco_pred_all': tco_pred_all, 'df_ranked': df_ranked, 'df_extra': df_extra
    }

if __name__ == "__main__":
    # Для ручного запуска одного режима
    mode_input = input("Выберите режим (1 - покупка, 2 - аренда): ").strip()
    mode = 'own' if mode_input == '1' else 'rent' if mode_input == '2' else None
    if mode:
        run_pipeline(mode, save_artifacts=False)