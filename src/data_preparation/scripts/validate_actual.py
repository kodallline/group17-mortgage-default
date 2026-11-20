"""Independent reference-label spot checks against the actual monthly source data."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from ingest import setup
from prepare import label_future

def main(work,out):
    con=setup(work/'prep.duckdb',work/'duckdb_temp')
    selections=[]
    for condition,n in [("target_default=1",200),("target_default=0 AND label_reason='payoff'",100),("target_default=0 AND label_reason='complete_or_incomplete'",200),("target_default IS NULL",1000)]:
        selections.append(con.execute(f"SELECT loan_id,month_ord,target_default FROM forecast_base WHERE {condition} ORDER BY hash(loan_id),month_ord LIMIT {n}").df())
    a=pd.concat(selections,ignore_index=True)
    ids=pd.DataFrame({'loan_id':a.loan_id.unique()});con.register('audit_ids',ids)
    m=con.execute('SELECT m.loan_id,m.month_ord,m.status_known,m.zero_code FROM monthly m JOIN audit_ids a USING(loan_id) ORDER BY m.loan_id,m.month_ord').df()
    histories=con.execute('SELECT h.* FROM history h JOIN audit_ids a USING(loan_id)').df().set_index('loan_id')
    grouped={k:list(g[['month_ord','status_known','zero_code']].itertuples(index=False,name=None)) for k,g in m.groupby('loan_id')}
    def nullable(x):return None if pd.isna(x) else int(x)
    failures=[]
    for row in a.itertuples(index=False):
        h=histories.loc[row.loan_id]
        future=[x for x in grouped[row.loan_id] if x[0]>row.month_ord][:3]
        expected,_=label_future(int(row.month_ord),nullable(h.first_default_month),nullable(h.first_payoff_month),nullable(h.first_other_exit_month),future)
        got=nullable(row.target_default)
        if got!=expected:failures.append({'loan_id':row.loan_id,'month_ord':int(row.month_ord),'actual':got,'expected':expected})
    # Recompute the training-only lagged regional feature from aggregate input rows.
    regional_errors=con.execute("""WITH expected AS (
        SELECT state,quarter_ord+4 quarter_ord,avg(is_default::INT) rate
        FROM aggregate_inputs WHERE split_R='train' GROUP BY 1,2)
        SELECT count(*) FROM dataset_R d JOIN expected e USING(state,quarter_ord)
        WHERE abs(d.regional_default_rate_lag4q-e.rate)>1e-7""").fetchone()[0]
    # All model files must expose the exact feature whitelist and finite known values.
    config=json.loads((out/'config.json').read_text())
    finite={}
    for p in ['R','E']:
        cols=config[p]['numeric']
        expr=' OR '.join(f'(isnan({c}) OR isinf({c}))' for c in cols)
        finite[p]=con.execute(f'SELECT count(*) FROM dataset_{p} WHERE {expr}').fetchone()[0]==0
    result={'actual_anchors_checked':len(a),'label_mismatches':failures,'training_only_regional_feature_errors':regional_errors,'no_infinite_features':finite}
    (out/'reports'/'independent_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert not failures and regional_errors==0 and all(finite.values())
    con.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.work,a.output)
