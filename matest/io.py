import io
import pandas as pd

def read_table_with_type(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln for ln in f.readlines() if ln.strip() and not ln.strip().startswith('//')]
        csv_text = ''.join(lines)
    try:
        df0 = pd.read_csv(io.StringIO(csv_text), header=0)
        cols0 = [str(c).strip() for c in df0.columns]
    except Exception:
        df0 = None
        cols0 = []

    oct_keys = {'MeasurePosX','MeasurePosY','CorrPosX','CorrPosY'}
    if set([c.lower() for c in cols0]).issuperset({c.lower() for c in oct_keys}):
        df = df0.copy()
        df = df.rename(columns={
            'MeasurePosX':'MeasurePosX', 'MeasurePosY':'MeasurePosY',
            'CorrPosX':'CorrPosX', 'CorrPosY':'CorrPosY'
        })
        df_out = pd.DataFrame()
        df_out['Ideal_X'] = pd.to_numeric(df['MeasurePosX'], errors='coerce')
        df_out['Ideal_Y'] = pd.to_numeric(df['MeasurePosY'], errors='coerce')
        corrx = pd.to_numeric(df['CorrPosX'], errors='coerce')
        corry = pd.to_numeric(df['CorrPosY'], errors='coerce')
        df_out['Real_X'] = df_out['Ideal_X'] + corrx
        df_out['Real_Y'] = df_out['Ideal_Y'] + corry
        return df_out, 'OCT'

    try:
        dfn = pd.read_csv(io.StringIO(csv_text), header=None)
    except Exception:
        dfn = None

    legacy_names = {'Ideal_X','Ideal_Y','Real_X','Real_Y'}
    if df0 is not None and set([c.lower() for c in cols0]) & set([c.lower() for c in legacy_names]):
        df = df0
    else:
        df = dfn if dfn is not None else pd.DataFrame()

    if df.shape[1] == 4:
        df.columns = ['Ideal_X', 'Ideal_Y', 'Real_X', 'Real_Y']
    elif df.shape[1] == 2:
        df.columns = ['Ideal_X', 'Ideal_Y']
        df['Real_X'] = pd.NA
        df['Real_Y'] = pd.NA
    else:
        rename_map = {}
        for c in df.columns:
            lc = str(c).strip().lower()
            if 'ideal' in lc and ('x' in lc) and ('y' not in lc):
                rename_map[c] = 'Ideal_X'
            elif 'ideal' in lc and ('y' in lc):
                rename_map[c] = 'Ideal_Y'
            elif 'real' in lc and ('x' in lc) and ('y' not in lc):
                rename_map[c] = 'Real_X'
            elif 'real' in lc and ('y' in lc):
                rename_map[c] = 'Real_Y'
        df = df.rename(columns=rename_map)
        for k in ['Ideal_X','Ideal_Y','Real_X','Real_Y']:
            if k not in df.columns:
                df[k] = pd.NA
    df = df[['Ideal_X','Ideal_Y','Real_X','Real_Y']]
    return df, 'DIST'
