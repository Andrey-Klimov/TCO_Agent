import os, pickle, joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def clean_region(series):
    s = series.astype(str).str.strip()
    s = s.str.replace(r'\.0+$', '', regex=True)
    return s

CATALOGS = {}
for mode in ['own', 'rent']:
    path = f'models/{mode}_catalog.pkl'
    if os.path.exists(path):
        CATALOGS[mode] = pd.read_pickle(path)
        if 'region' in CATALOGS[mode].columns:
            CATALOGS[mode]['region'] = clean_region(CATALOGS[mode]['region'])
        print(f"Загружен каталог для режима '{mode}': {len(CATALOGS[mode])} объектов")
    else:
        CATALOGS[mode] = pd.DataFrame()

QUERY_WEIGHTS = {
    'region': 0.25, 'obsh_plosh': 0.15, 'quan': 0.05,
    'building_age': 0.10, 'first_line': 0.02, 'tco_pred_ths': 0.18,
}

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
            s = 1.0 if str(val).strip().lower() == str(desired).strip().lower() else 0.0
        score += w * s
    return score / max(total_w, 1e-9)

@app.route('/predict', methods=['POST'])
def predict():
    # ... (без изменений, как в предыдущем ответе)
    # Убедитесь, что код функции predict взят из последней версии, где есть tco_min.
    # Здесь вставим тот же код, что был в ответе с tco_min.
    try:
        data = request.get_json()
        mode = data.get('mode', 'own')
        if mode not in CATALOGS or CATALOGS[mode].empty:
            return jsonify({'error': f'Режим {mode} недоступен'}), 400
        df = CATALOGS[mode].copy()

        region = data.get('region')
        area_min = float(data['obsh_plosh_min']) if data.get('obsh_plosh_min') is not None else None
        area_max = float(data['obsh_plosh_max']) if data.get('obsh_plosh_max') is not None else None
        quan_min = int(data['quan_min']) if data.get('quan_min') is not None else None
        quan_max = int(data['quan_max']) if data.get('quan_max') is not None else None
        age_min = float(data['building_age_min']) if data.get('building_age_min') is not None else None
        age_max = float(data['building_age_max']) if data.get('building_age_max') is not None else None
        first_line = data.get('first_line', 0)
        tco_min = float(data['tco_min']) if data.get('tco_min') is not None else None
        tco_max = float(data['tco_max']) if data.get('tco_max') is not None else None

        filters = []
        if region and str(region).strip():
            target = str(region).strip().lower()
            filters.append(df['region'].astype(str).str.strip().str.lower() == target)

        if area_min is not None or area_max is not None:
            if area_min is not None and area_max is not None:
                filters.append((df['obsh_plosh'] >= area_min) & (df['obsh_plosh'] <= area_max))
            elif area_min is not None:
                filters.append(df['obsh_plosh'] >= area_min)
            elif area_max is not None:
                filters.append(df['obsh_plosh'] <= area_max)

        if quan_min is not None or quan_max is not None:
            if quan_min is not None and quan_max is not None:
                filters.append((df['quan'] >= quan_min) & (df['quan'] <= quan_max))
            elif quan_min is not None:
                filters.append(df['quan'] >= quan_min)
            elif quan_max is not None:
                filters.append(df['quan'] <= quan_max)

        if 'building_age' in df.columns:
            if age_min is not None or age_max is not None:
                if age_min is not None and age_max is not None:
                    filters.append((df['building_age'] >= age_min) & (df['building_age'] <= age_max))
                elif age_min is not None:
                    filters.append(df['building_age'] >= age_min)
                elif age_max is not None:
                    filters.append(df['building_age'] <= age_max)

        if first_line == 1 or first_line is True:
            filters.append(df['first_line'] == 1)

        if tco_min is not None:
            filters.append(df['tco_pred_ths'] >= tco_min)
        if tco_max is not None:
            filters.append(df['tco_pred_ths'] <= tco_max)

        if filters:
            mask = filters[0]
            for f in filters[1:]:
                mask &= f
            df = df[mask].copy()

        if df.empty:
            return jsonify([])

        query = {}
        if region and str(region).strip():
            query['region'] = str(region).strip().lower()
        if area_min is not None or area_max is not None:
            if area_min is not None and area_max is not None:
                query['obsh_plosh'] = (area_min, area_max)
            elif area_min is not None:
                query['obsh_plosh'] = (area_min, float('inf'))
            elif area_max is not None:
                query['obsh_plosh'] = (0, area_max)
        if quan_min is not None or quan_max is not None:
            if quan_min is not None and quan_max is not None:
                query['quan'] = (quan_min, quan_max)
            elif quan_min is not None:
                query['quan'] = (quan_min, int(1e9))
            elif quan_max is not None:
                query['quan'] = (0, quan_max)
        if 'building_age' in df.columns:
            if age_min is not None or age_max is not None:
                if age_min is not None and age_max is not None:
                    query['building_age'] = (age_min, age_max)
                elif age_min is not None:
                    query['building_age'] = (age_min, 200)
                elif age_max is not None:
                    query['building_age'] = (0, age_max)
        if first_line in (1, True):
            query['first_line'] = (1, 1)
        if tco_min is not None or tco_max is not None:
            lo = tco_min if tco_min is not None else 0.0
            hi = tco_max if tco_max is not None else float('inf')
            query['tco_pred_ths'] = (lo, hi)

        if query:
            df['query_sim'] = df.apply(lambda row: compute_query_similarity(row, query, QUERY_WEIGHTS), axis=1)
        else:
            df['query_sim'] = 1.0

        df['final_score'] = 0.7 * df['query_sim'] + 0.3 * df['rf_base_score']
        top_k = min(int(data.get('top_k', 10)), len(df))
        top = df.sort_values('final_score', ascending=False).head(top_k)

        columns = ['region','segmentt','obsh_plosh','building_age','quan',
                   'first_line','tco_pred_ths','rf_base_score','query_sim','final_score']
        result = top[columns].to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/regions', methods=['GET'])
def get_regions():
    regions = set()
    for df in CATALOGS.values():
        if not df.empty and 'region' in df.columns:
            regions.update(df['region'].dropna().unique())
    cleaned = sorted([str(r) for r in regions], key=lambda x: int(x) if x.isdigit() else 0)
    return jsonify(cleaned)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)