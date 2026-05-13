"""
Скрипт коррекции прогнозов TCO с помощью GigaChat API.
Версия 8.0 — возврат к исходной логике:
- отдельные XGBoost-модели для собственности и аренды,
- обучение и оценка на полных выборках (как в самой первой версии),
- LLM-коррекция для всех объектов,
- сравнение MAPE до и после LLM.
"""
import os, re, time, pickle, warnings, requests
import numpy as np, pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.2)

# ================= НАСТРОЙКИ =================
SOBSTV_FILE   = r"D:\user\Рабочий стол\TCO_gboost\sobstvennost.csv"
ARENDA_FILE   = r"D:\user\Рабочий стол\TCO_gboost\arenda.csv"

TAX_RATE      = 0.20
DISCOUNT_RATE = 0.10
DEFAULT_T     = 10
VAT_RATE      = 0.20
OPEX_RATE     = 0.03
CAPEX_RATE    = 0.01
DEPR_RATE     = 0.02
CS_RATE       = 0.05
GROWTH_RATE   = 0.02

RANDOM_STATE  = 42
MIN_COST      = 100

# GigaChat
AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
CREDENTIALS = "MDE5ZTE3ODctMTE2Yy03MzUzLWJiMzYtZWY2MmJkMGUxMGFiOmE2NzQyN2MxLTFhNDItNGYyYS05NTNmLTE3NGJhODFhOGM2ZA=="
VERIFY_SSL = False
REQUEST_DELAY = 2.0
BATCH_SIZE = 4
MAX_ADJUSTMENT = 0.30

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================
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
    df = df.replace(['-', '--', 'null', 'NULL', 'None', 'nan', 'NaN', ''], np.nan)
    for col in df.columns:
        if df[col].dtype == 'object':
            s = (df[col].astype(str).str.strip()
                 .str.replace(' ', '', regex=False)
                 .str.replace(',', '.', regex=False))
            s = s.replace(['-', '--', 'null', 'NULL', 'None'], np.nan)
            converted = pd.to_numeric(s, errors='coerce')
            if converted.notna().sum() > len(df) * 0.1:
                df[col] = converted
    return df

def prepare_cost_data_own(df):
    cost_cols = ['balans_v', 'ostat_v', 'rinoch_v']
    for col in cost_cols:
        if col not in df.columns: df[col] = np.nan
        df[col] = safe_numeric(df[col])
    df['Inv'] = df['balans_v'].fillna(df['ostat_v'])
    if 'region' in df.columns and 'tipob' in df.columns:
        group_med = df.groupby(['region', 'tipob'])['Inv'].transform('median')
        df['Inv'] = df['Inv'].fillna(group_med)
    overall_med = df['Inv'].median()
    if pd.isna(overall_med) or overall_med <= 0: overall_med = 10_000
    df['Inv'] = df['Inv'].fillna(overall_med)
    df.loc[df['Inv'] <= 0, 'Inv'] = overall_med
    df['Balans_for_Depr'] = df['balans_v'].fillna(df['Inv'])
    df['Balans_for_Depr'] = df['Balans_for_Depr'].clip(lower=MIN_COST/10)
    return df

def prepare_lease_data(df):
    for col in ['arend_st', 'obsh_plosh', 'perem_ap']:
        if col in df.columns: df[col] = safe_numeric(df[col])
    df = df.dropna(subset=['arend_st', 'obsh_plosh'])
    df = df[(df['arend_st'] > 0) & (df['obsh_plosh'] > 0)].copy()
    if df.empty: raise ValueError("Нет объектов аренды.")
    df['OPEX'] = df['arend_st'] * 12 * df['obsh_plosh'] / (1 + VAT_RATE)
    if 'perem_ap' in df.columns:
        df['T_lease'] = pd.to_numeric(df['perem_ap'], errors='coerce').fillna(DEFAULT_T)
        df.loc[df['T_lease'] <= 0, 'T_lease'] = DEFAULT_T
    else:
        df['T_lease'] = DEFAULT_T
    df = df[df['T_lease'] >= 1]
    df['Inv'] = 0.0; df['Balans_for_Depr'] = 0.0
    return df.reset_index(drop=True)

def calculate_tco_own(df):
    Inv = df['Inv'].values; Balans = df['Balans_for_Depr'].values
    r, tax = DISCOUNT_RATE, TAX_RATE
    years = np.arange(1, DEFAULT_T+1)
    disc = 1 / (1 + r) ** years
    VAT_Inv = Inv * VAT_RATE
    inv_comp = Inv + VAT_Inv * (1 - tax)
    OPEX = Inv * OPEX_RATE; CAPEX = Inv * CAPEX_RATE
    VAT_CAPEX = CAPEX * VAT_RATE; D = Balans * DEPR_RATE
    annual_cf = OPEX + VAT_CAPEX * (1 - tax) + CAPEX - D * tax
    disc_sum = np.sum(annual_cf[:, np.newaxis] * disc, axis=1)
    FV_T = Inv * (1 + GROWTH_RATE) ** DEFAULT_T
    Cs_T = FV_T * CS_RATE
    term = Cs_T * (1 - tax) / (1 + r) ** DEFAULT_T
    return np.clip(inv_comp + disc_sum + term, MIN_COST, None)

def calculate_tco_rent(df):
    r, tax = DISCOUNT_RATE, TAX_RATE
    T_vec = df['T_lease'].values.astype(int)
    annual_net = df['OPEX'].values * (1 - tax)
    max_T = int(T_vec.max())
    disc_all = np.array([1 / (1 + r)**t for t in range(1, max_T+1)])
    TCO = np.zeros(len(df))
    for i, T_i in enumerate(T_vec):
        TCO[i] = annual_net[i] * np.sum(disc_all[:T_i])
    return np.clip(TCO, MIN_COST, None)

def create_tco_features(df):
    data = df.copy()
    if 'validfrom' in data.columns:
        data['validfrom'] = pd.to_datetime(data['validfrom'], errors='coerce')
        data['building_age'] = ((pd.Timestamp.now() - data['validfrom']).dt.days / 365.25).clip(0,150).fillna(0)
    else:
        data['building_age'] = 0
    area_cols = ['obsh_plosh','osnov_plosh','vspom_plosh','proch_plosh']
    exist = [c for c in area_cols if c in data.columns]
    data['total_area'] = data[exist].fillna(0).sum(axis=1) if exist else 0
    num_cols = ['obsh_plosh','osnov_plosh','vspom_plosh','proch_plosh',
               'plosh_v_ar','neis_plosh','osn_obsh','vspom_obsh',
               'org_pm','teh_pm','k4_koef','k4_koef_ch','irrate',
               'quan','building_age','total_area','zz_lattitude','zz_longitude']
    num_cols = [c for c in num_cols if c in data.columns]
    for c in num_cols: data[c] = safe_numeric(data[c])
    X_num = data[num_cols].fillna(0).astype(float)
    cat_cols = ['tipob','priznas','region','segmentt','subsegmentt',
               'usgfunction','elorgst_text','statusesq','zz_priznas','xmusgfunction','vidpr']
    cat_cols = [c for c in cat_cols if c in data.columns]
    X_cat = data[cat_cols].fillna('NA')
    for col in X_cat.columns:
        le = LabelEncoder()
        X_cat[col] = le.fit_transform(X_cat[col].astype(str))
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    for c in X.columns:
        if X[c].dtype == 'object': X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
    return X

def get_gigachat_token():
    headers = {"Authorization": f"Basic {CREDENTIALS}",
               "RqUID": "12345678-1234-1234-1234-123456789012",
               "Content-Type": "application/x-www-form-urlencoded"}
    data = {"scope": "GIGACHAT_API_PERS"}
    try:
        resp = requests.post(AUTH_URL, headers=headers, data=data, verify=VERIFY_SSL)
        if resp.status_code == 200: return resp.json()["access_token"]
        else: print(f"Ошибка аутентификации {resp.status_code}: {resp.text[:300]}")
    except Exception as e: print(f"Ошибка токена: {e}")
    return None

def call_gigachat_batch(prompt, token):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"model": "GigaChat-2", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": 150}
    try:
        resp = requests.post(CHAT_URL, headers=headers, json=payload, verify=VERIFY_SSL)
        if resp.status_code == 200:
            answer = resp.json()["choices"][0]["message"]["content"]
            numbers = re.findall(r"\d+\.?\d*", answer.replace(',', '.'))
            return [float(n) for n in numbers]
        elif resp.status_code == 400: print(f"\n400: {resp.text[:200]}")
        elif resp.status_code == 429: print("\n429 - ждём"); time.sleep(15)
    except Exception as e: print(f"\nСеть: {e}")
    return None

def correct_objects(df, pred_xgb, token, desc="LLM"):
    n = len(df)
    corrected = np.copy(pred_xgb)
    skipped = 0
    indices = list(range(n))
    batches = [indices[i:i+BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]
    for batch in tqdm(batches, desc=desc):
        prompts = []
        for idx in batch:
            row = df.iloc[idx]
            feat = (f"Пл:{row.get('obsh_plosh','?')}м2, Рег:{row.get('region','?')}, "
                    f"Сегм:{row.get('segmentt','?')}, Возр:{row.get('building_age','?')}л, "
                    f"1лин:{'да' if str(row.get('first_line_house','')).strip()=='X' else 'нет'}, "
                    f"XGB:{pred_xgb[idx]:.1f}т.р")
            prompts.append(f"Об{idx}: {feat} -> СкорTCO:")
        full_prompt = ("Ты оценщик недвижимости. Для каждого объекта дай уточнённый TCO в тыс.руб. "
                       "Только числа через запятую в том же порядке.\n" + "\n".join(prompts))
        nums = call_gigachat_batch(full_prompt, token)
        if nums and len(nums) == len(batch):
            for i, idx in enumerate(batch):
                lo = pred_xgb[idx] * (1 - MAX_ADJUSTMENT)
                hi = pred_xgb[idx] * (1 + MAX_ADJUSTMENT)
                corrected[idx] = np.clip(nums[i], lo, hi)
        else:
            skipped += len(batch)
        time.sleep(REQUEST_DELAY)
    print(f"   Пропущено объектов: {skipped}")
    return corrected

def main():
    print("Загрузка данных...")
    df_own = load_and_clean(SOBSTV_FILE)
    df_own = prepare_cost_data_own(df_own)
    tco_own = calculate_tco_own(df_own)
    X_own = create_tco_features(df_own)

    df_rent = load_and_clean(ARENDA_FILE)
    df_rent = prepare_lease_data(df_rent)
    tco_rent = calculate_tco_rent(df_rent)
    X_rent = create_tco_features(df_rent)

    # Обучение отдельных моделей на полных выборках
    scaler_own = StandardScaler()
    X_own_sc = scaler_own.fit_transform(X_own)
    y_own_log = np.log1p(tco_own)
    model_own = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    model_own.fit(X_own_sc, y_own_log)
    pred_own = np.expm1(model_own.predict(X_own_sc))

    scaler_rent = StandardScaler()
    X_rent_sc = scaler_rent.fit_transform(X_rent)
    y_rent_log = np.log1p(tco_rent)
    model_rent = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    model_rent.fit(X_rent_sc, y_rent_log)
    pred_rent = np.expm1(model_rent.predict(X_rent_sc))

    mape_own = mean_absolute_percentage_error(tco_own, pred_own)
    mape_rent = mean_absolute_percentage_error(tco_rent, pred_rent)
    print(f"\nMAPE XGBoost: собств. {mape_own*100:.2f}%, аренда {mape_rent*100:.2f}%")

    # LLM-коррекция
    token = get_gigachat_token()
    if not token: return

    df_all = pd.concat([df_own, df_rent], ignore_index=True)
    pred_all = np.concatenate([pred_own, pred_rent])
    n_own = len(df_own)

    print("\nLLM-коррекция для всех объектов...")
    corrected_all = correct_objects(df_all, pred_all, token)

    corrected_own = corrected_all[:n_own]
    corrected_rent = corrected_all[n_own:]

    mape_own_llm = mean_absolute_percentage_error(tco_own, corrected_own)
    mape_rent_llm = mean_absolute_percentage_error(tco_rent, corrected_rent)
    print(f"\nMAPE после LLM: собств. {mape_own_llm*100:.2f}%, аренда {mape_rent_llm*100:.2f}%")

    # Сохранение словаря скорректированных TCO
    result = {}
    for i in range(n_own):
        result[('own', i)] = corrected_own[i]
    for i in range(len(df_rent)):
        result[('rent', i)] = corrected_rent[i]
    with open("tco_llm_corrected.pkl", "wb") as f:
        pickle.dump(result, f)
    print("\nСловарь сохранён в tco_llm_corrected.pkl")

    # График сравнения
    categories = ['Собственность', 'Аренда']
    mape_before = [mape_own*100, mape_rent*100]
    mape_after = [mape_own_llm*100, mape_rent_llm*100]
    x = np.arange(len(categories))
    width = 0.3
    fig, ax = plt.subplots(figsize=(8,5))
    bars1 = ax.bar(x - width/2, mape_before, width, label='XGBoost', color='#4e79a7', edgecolor='k')
    bars2 = ax.bar(x + width/2, mape_after, width, label='XGBoost + LLM', color='#f28e2b', edgecolor='k')
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}%', ha='center')
    ax.set_ylabel('MAPE, %')
    ax.set_title('Сравнение ошибки TCO до и после LLM-коррекции')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig('mape_llm_comparison.png', dpi=120)
    plt.show()
    print("График сохранён: mape_llm_comparison.png")

if __name__ == "__main__":
    main()