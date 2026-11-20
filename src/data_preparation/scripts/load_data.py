"""Read only declared features and one declared split; keeps audit columns out."""
import json
from pathlib import Path
import pandas as pd

def load_xy(root,protocol='R',split='train',extended=False,state=None):
    root=Path(root)
    config=json.loads((root/'config.json').read_text())
    spec=config[protocol]
    numeric=list(spec['numeric']);categorical=list(spec['categorical'])
    if extended:
        numeric+=config['optional_extended_features']['numeric']
        categorical+=config['optional_extended_features']['categorical']
    features=numeric+categorical
    columns=list(dict.fromkeys(['loan_id','state','report_date',spec['target'],spec['split']]+features))
    filters=[(spec['split'],'=',split)]
    if state:filters.append(('state','=',state))
    d=pd.read_parquet(root/'data'/spec['file'],columns=columns,filters=filters)
    X=d[features].copy()
    for c in numeric:X[c]=X[c].astype('float32')
    for c in categorical:X[c]=X[c].fillna('UNKNOWN')
    y=d[spec['target']].astype('int8')
    metadata=d[['loan_id','state','report_date',spec['split']]].copy()
    return X,y,metadata

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    X,y,meta=load_xy(root,protocol='E',split='calibration_select')
    print({'rows':len(X),'features':list(X),'positives':int(y.sum()),'regions':meta.state.nunique()})
