"""Create quarterly replication (R) and prospective three-month (E) datasets."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from ingest import setup,literal
from public_sources import FIPS

SEED=20261116
START=2013*12
END=2019*12-1
CORE_NUMERIC=['orig_dti_pct','current_rate_pct','current_ltv_proxy_pct','seniority_months','log_orig_property_value_proxy','gdp_growth_lag2y_pct','unemployment_change_lag2y_pp','hpi_growth_lag3m_pct','regional_default_rate_lag4q']
CORE_CATEGORICAL=['property_type']
OPTIONAL_NUMERIC=['fico','orig_ltv_pct','orig_cltv_pct','orig_rate_pct','orig_term_months','borrowers','units','seller_mean_ltv_lag4q','seller_mean_dti_lag4q','seller_mean_balance_change_lag4q']
OPTIONAL_CATEGORICAL=['occupancy','purpose','channel']

def split_loans(df,seed=SEED):
    """Stratified by state and any observed default through the analysis cutoff."""
    d=df.sort_values('loan_id').reset_index(drop=True).copy()
    d['stratum_default']=d.first_default_month.notna().astype('int8')
    d['split_R']=''
    rng=np.random.default_rng(seed)
    for _,ix in d.groupby(['state','stratum_default'],dropna=False).groups.items():
        ix=np.asarray(ix);ix=ix[rng.permutation(len(ix))]
        n=len(ix);a=int(n*.6);b=int(n*.8)
        d.loc[ix[:a],'split_R']='train';d.loc[ix[a:b],'split_R']='validation';d.loc[ix[b:],'split_R']='test'
    return d[['loan_id','state','stratum_default','split_R']]

def temporal_role(quarter):
    """Quarter ordinal = year*4 + zero-based quarter."""
    if 2013*4<=quarter<=2014*4+3:return 'train'
    if 2015*4+1<=quarter<=2015*4+2:return 'validation'
    if 2016*4<=quarter<=2016*4+1:return 'calibration_fit'
    if quarter==2016*4+3:return 'calibration_select'
    if 2017*4+1<=quarter<=2018*4+2:return 'test'
    return 'purged'

def label_future(current_month,first_default,first_payoff,first_other_exit,future):
    """Reference specification used in independent hand-checked boundary tests.

    future contains (month ordinal, known_status, zero_code) for the next records.
    The production SQL below implements the same declared event definition.
    """
    if first_default is not None and current_month<first_default<=current_month+3:return 1,'default'
    if first_other_exit is not None and current_month<first_other_exit<=current_month+3:return None,'other_exit'
    by_month={m:(known,z) for m,known,z in future}
    if first_payoff is not None and current_month<first_payoff<=current_month+3:
        needed=range(current_month+1,first_payoff+1)
        if all(m in by_month and (by_month[m][0] or (m==first_payoff and by_month[m][1]=='01')) for m in needed):return 0,'payoff'
    if all(m in by_month and by_month[m][0] for m in range(current_month+1,current_month+4)):return 0,'complete'
    return None,'incomplete_followup'

def make(out,work):
    t0=time.time()
    con=setup(work/'prep.duckdb',work/'duckdb_temp')
    data=out/'data';reports=out/'reports';ext=out/'external'
    states=','.join(literal(x) for x in FIPS)
    loans=con.execute(f'SELECT loan_id,state,first_default_month FROM loans WHERE state IN ({states})').df()
    split=split_loans(loans);split.to_parquet(data/'loan_splits_R.parquet',index=False)
    con.register('split_frame',split)
    con.execute('CREATE OR REPLACE TABLE splits AS SELECT * FROM split_frame')
    con.execute(f"CREATE OR REPLACE TABLE hpi AS SELECT * FROM read_parquet({literal(ext/'hpi_state_monthly.parquet')})")
    con.execute(f"CREATE OR REPLACE TABLE annual AS SELECT * FROM read_parquet({literal(ext/'state_annual.parquet')})")
    con.execute(f"""CREATE OR REPLACE TABLE risk_monthly AS
        SELECT m.*,l.state,l.seller,h.first_default_month,h.first_payoff_month,h.first_other_exit_month,
            (m.month_ord-h.first_report_month)::INTEGER observed_seasoning_months
        FROM monthly m JOIN loans l USING(loan_id) JOIN history h USING(loan_id)
        WHERE l.state IN ({states})
          AND (h.first_default_month IS NULL OR m.month_ord<=h.first_default_month)
          AND (h.first_payoff_month IS NULL OR m.month_ord<=h.first_payoff_month)
          AND (h.first_other_exit_month IS NULL OR m.month_ord<=h.first_other_exit_month)""")
    # First event in a quarter is retained even when it occurs before quarter-end.
    con.execute("""CREATE OR REPLACE TABLE quarter_snapshots AS
        SELECT * EXCLUDE(rn) FROM (
          SELECT *,row_number() OVER(PARTITION BY loan_id,quarter_ord ORDER BY month_ord DESC) rn
          FROM risk_monthly WHERE status_known AND (current_upb>0 OR is_default)
        ) WHERE rn=1""")
    # Historical summaries never include post-default updates or future quarters.
    con.execute("""CREATE OR REPLACE TABLE aggregate_inputs AS
        SELECT q.*,l.orig_ltv_pct,l.orig_dti_pct,s.split_R,
            CASE WHEN p.current_upb>0 THEN q.current_upb/p.current_upb-1 END balance_change_12m
        FROM quarter_snapshots q JOIN loans l USING(loan_id) JOIN splits s USING(loan_id)
        LEFT JOIN monthly p ON q.loan_id=p.loan_id AND p.month_ord=q.month_ord-12""")
    for role,condition in [('R',"split_R='train'"),('E','true')]:
        con.execute(f"""CREATE OR REPLACE TABLE regional_{role} AS
            SELECT state,quarter_ord+4 quarter_ord,avg(is_default::INTEGER)::FLOAT regional_default_rate_lag4q,
                   count(*)::INTEGER regional_history_exposure
            FROM aggregate_inputs WHERE {condition} GROUP BY state,quarter_ord""")
        con.execute(f"""CREATE OR REPLACE TABLE seller_{role} AS
            SELECT seller,quarter_ord+4 quarter_ord,avg(orig_ltv_pct)::FLOAT seller_mean_ltv_lag4q,
                avg(orig_dti_pct)::FLOAT seller_mean_dti_lag4q,
                avg(balance_change_12m)::FLOAT seller_mean_balance_change_lag4q,
                count(*)::INTEGER seller_history_exposure
            FROM aggregate_inputs WHERE {condition} AND seller<>'OTHER' GROUP BY seller,quarter_ord""")
    con.execute(f"""CREATE OR REPLACE TABLE replication_base AS
        SELECT q.*,q.is_default::TINYINT target_default,s.split_R
        FROM quarter_snapshots q JOIN splits s USING(loan_id)
        WHERE q.month_ord BETWEEN {START} AND {END}""")
    # Leads are evaluated over the entire observed monthly history, before filtering anchors.
    lead_expr=',\n'.join(f"lead({c},{k}) OVER w AS f{k}_{c}" for k in range(1,4) for c in ['month_ord','status_known','zero_code'])
    con.execute(f"""CREATE OR REPLACE TABLE forecast_base AS
        WITH future AS (SELECT *,{lead_expr} FROM risk_monthly WINDOW w AS(PARTITION BY loan_id ORDER BY month_ord)),
        anchors AS (
            SELECT * FROM future
            WHERE month_ord BETWEEN {START} AND {END-3} AND (month_ord+1)%3=0
                AND NOT is_default AND status_known AND current_upb>0 AND zero_code=''
        ), labeled AS (
            SELECT *,
              CASE
                WHEN first_default_month>month_ord AND first_default_month<=month_ord+3 THEN 1
                WHEN first_other_exit_month>month_ord AND first_other_exit_month<=month_ord+3 THEN NULL
                WHEN first_payoff_month=month_ord+1 AND f1_month_ord=month_ord+1 AND f1_zero_code='01' THEN 0
                WHEN first_payoff_month=month_ord+2 AND f2_month_ord=month_ord+2 AND f2_zero_code='01' AND f1_status_known THEN 0
                WHEN first_payoff_month=month_ord+3 AND f3_month_ord=month_ord+3 AND f3_zero_code='01' AND f1_status_known AND f2_status_known THEN 0
                WHEN f3_month_ord=month_ord+3 AND f1_status_known AND f2_status_known AND f3_status_known THEN 0
              END::TINYINT target_default,
              CASE
                WHEN first_default_month>month_ord AND first_default_month<=month_ord+3 THEN 'default'
                WHEN first_other_exit_month>month_ord AND first_other_exit_month<=month_ord+3 THEN 'other_exit'
                WHEN first_payoff_month BETWEEN month_ord+1 AND month_ord+3 THEN 'payoff'
                ELSE 'complete_or_incomplete'
              END label_reason
            FROM anchors
        ) SELECT * FROM labeled""")
    # Calendar roles are declared before model training; unused quarters purge label overlap.
    role_rows=[{'quarter_ord':q,'split_E':temporal_role(q)} for q in range(2013*4,2019*4)]
    con.register('roles_frame',pd.DataFrame(role_rows))
    con.execute('CREATE OR REPLACE TABLE temporal_roles AS SELECT * FROM roles_frame')
    pd.DataFrame(role_rows).to_csv(reports/'temporal_roles.csv',index=False)

    feature_select="""
        b.loan_id,b.state,b.vintage,b.quarter_ord,b.month_ord,
        last_day(make_date(b.report_ym//100,b.report_ym%100,1)) AS report_date,
        b.target_default,
        l.orig_dti_pct,
        CASE WHEN b.current_rate BETWEEN 0 AND 30 THEN b.current_rate::FLOAT END current_rate_pct,
        greatest(b.month_ord-l.first_payment_month+2,0)::SMALLINT seniority_months,
        CASE WHEN l.orig_ltv_pct>0 THEN ln(l.orig_upb/(l.orig_ltv_pct/100))::FLOAT END log_orig_property_value_proxy,
        (l.orig_ltv_pct*b.current_upb/l.orig_upb)::FLOAT amortization_ltv_proxy_pct,
        (l.orig_ltv_pct*b.current_upb/l.orig_upb *
            CASE WHEN b.month_ord-3>=l.first_payment_month-2 AND ho.hpi_nsa>0 AND ht.hpi_nsa>0
                THEN ho.hpi_nsa/ht.hpi_nsa ELSE 1 END)::FLOAT current_ltv_proxy_pct,
        (ht.hpi_nsa/hy.hpi_nsa-1)::FLOAT*100 hpi_growth_lag3m_pct,
        (b.month_ord-3)::INTEGER hpi_reference_month,
        (b.report_ym//100-2)::SMALLINT macro_reference_year,
        a.gdp_growth_pct::FLOAT gdp_growth_lag2y_pct,
        a.unemployment_change_pp::FLOAT unemployment_change_lag2y_pp,
        a.unemployment_rate_pct::FLOAT unemployment_rate_lag2y_pct,
        l.property_type,l.amortization,l.fico,l.orig_ltv_pct,l.orig_cltv_pct,l.orig_rate_pct,
        l.orig_term_months,l.borrowers,l.units,l.occupancy,l.purpose,l.channel,
        r.regional_default_rate_lag4q,r.regional_history_exposure,
        s.seller_mean_ltv_lag4q,s.seller_mean_dti_lag4q,s.seller_mean_balance_change_lag4q,s.seller_history_exposure
    """
    for protocol,base in [('R','replication_base'),('E','forecast_base')]:
        extra="b.split_R" if protocol=='R' else "t.split_E,b.label_reason,(b.month_ord+3)::INTEGER label_end_month,last_day(make_date((b.month_ord+3)//12,(b.month_ord+3)%12+1,1)) label_end_date"
        joins='' if protocol=='R' else 'JOIN temporal_roles t ON t.quarter_ord=b.quarter_ord'
        role_filter='' if protocol=='R' else "AND t.split_E<>'purged'"
        con.execute(f"""CREATE OR REPLACE TABLE dataset_{protocol} AS
            SELECT {feature_select},{extra}
            FROM {base} b JOIN loans l USING(loan_id)
            LEFT JOIN hpi ho ON b.state=ho.state AND ho.month_ord=l.first_payment_month-2
            LEFT JOIN hpi ht ON b.state=ht.state AND ht.month_ord=b.month_ord-3
            LEFT JOIN hpi hy ON b.state=hy.state AND hy.month_ord=b.month_ord-15
            LEFT JOIN annual a ON a.state=b.state AND a.year=b.report_ym//100-2
            LEFT JOIN regional_{protocol} r ON r.state=b.state AND r.quarter_ord=b.quarter_ord
            LEFT JOIN seller_{protocol} s ON s.seller=b.seller AND s.quarter_ord=b.quarter_ord
            {joins}
            WHERE b.target_default IS NOT NULL {role_filter}""")
        con.execute(f"COPY dataset_{protocol} TO {literal(data/('replication_R.parquet' if protocol=='R' else 'forecast_E.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD)")
    # Audit-only columns are kept separate and must never be fed to a classifier.
    con.execute(f"""COPY (SELECT q.loan_id,q.report_ym,q.first_default_month,h.first_90plus_month,h.first_reo_month,
          q.first_payoff_month,q.first_other_exit_month,q.delinquency_raw,q.zero_code,q.modification,q.disaster,q.assistance,
          CASE WHEN q.eltv BETWEEN 1 AND 998 THEN q.eltv END reported_eltv
        FROM risk_monthly q JOIN history h USING(loan_id)
        WHERE q.month_ord BETWEEN {START} AND {END}
        AND ((q.month_ord+1)%3=0 OR q.is_default)
    ) TO {literal(data/'audit_only.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)""")
    config={'seed':SEED,'region':'US state and DC; territories excluded','origination_vintages':list(range(2010,2019)),
        'observation_start':'2013-01','observation_end':'2018-12','horizon_months':3,
        'target_definition':'First observed numeric delinquency status >=3 (90+ days proxy) or RA (REO acquisition fallback).',
        'R':{'file':'replication_R.parquet','target':'target_default','split':'split_R','numeric':CORE_NUMERIC,'categorical':CORE_CATEGORICAL},
        'E':{'file':'forecast_E.parquet','target':'target_default','split':'split_E','numeric':CORE_NUMERIC,'categorical':CORE_CATEGORICAL,'monitoring_existing_loans':True,'exclude_roles':['purged']},
        'optional_extended_features':{'numeric':OPTIONAL_NUMERIC,'categorical':OPTIONAL_CATEGORICAL},
        'do_not_use_as_features':['loan_id','state (when fitting separate regional models)','vintage','quarter_ord','month_ord','report_date','target_default','split_R','split_E','label_reason','label_end_month','label_end_date','hpi_reference_month','macro_reference_year','regional_history_exposure','seller_history_exposure','amortization (constant FRM)','all columns in audit_only.parquet'],
        'availability_limit':'Revised historical releases. HPI lagged 3 months; annual GDP/unemployment lagged 2 calendar years. Not a fully vintage-correct real-time backtest.',
        'preprocessing':'Missing values preserved. Fit imputers, scaling, category encoders and optional tail trimming on each model training split only. No balancing of validation/calibration/test.'}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    # Independent dataset checks and finite differences.
    checks={}
    for p in ['R','E']:
        checks[f'{p}_unique_loan_quarter']=con.execute(f'SELECT count(*)=count(DISTINCT (loan_id,quarter_ord)) FROM dataset_{p}').fetchone()[0]
        checks[f'{p}_feature_period']=con.execute(f'SELECT min(month_ord)>={START} AND max(month_ord)<={END} FROM dataset_{p}').fetchone()[0]
        checks[f'{p}_macro_lag']=con.execute(f'SELECT bool_and(macro_reference_year<=year(report_date)-2 AND hpi_reference_month<=month_ord-3) FROM dataset_{p}').fetchone()[0]
        checks[f'{p}_hpi_coverage']=con.execute(f'SELECT count(*) FILTER(WHERE hpi_growth_lag3m_pct IS NULL)=0 FROM dataset_{p}').fetchone()[0]
        checks[f'{p}_gdp_coverage']=con.execute(f'SELECT count(*) FILTER(WHERE gdp_growth_lag2y_pct IS NULL)=0 FROM dataset_{p}').fetchone()[0]
    checks['R_no_cross_split_loan']=con.execute('SELECT count(*)=0 FROM (SELECT loan_id FROM dataset_R GROUP BY 1 HAVING count(DISTINCT split_R)>1)').fetchone()[0]
    checks['R_one_positive_per_loan']=con.execute('SELECT count(*)=0 FROM (SELECT loan_id FROM dataset_R GROUP BY 1 HAVING sum(target_default)>1)').fetchone()[0]
    checks['E_no_existing_default']=con.execute('SELECT count(*)=0 FROM dataset_E d JOIN history h USING(loan_id) WHERE h.first_default_month<=d.month_ord').fetchone()[0]
    checks['E_positive_horizon']=con.execute('SELECT count(*)=0 FROM dataset_E d JOIN history h USING(loan_id) WHERE d.target_default=1 AND NOT(h.first_default_month>d.month_ord AND h.first_default_month<=d.label_end_month)').fetchone()[0]
    checks['E_negative_no_default_in_horizon']=con.execute('SELECT count(*)=0 FROM dataset_E d JOIN history h USING(loan_id) WHERE d.target_default=0 AND h.first_default_month>d.month_ord AND h.first_default_month<=d.label_end_month').fetchone()[0]
    checks['E_outcomes_before_cutoff']=con.execute(f'SELECT bool_and(label_end_month<={END}) FROM dataset_E').fetchone()[0]
    stages=['train','validation','calibration_fit','calibration_select','test']
    bounds={}
    for role in stages:
        bounds[role]=con.execute(f'SELECT min(month_ord),max(label_end_month) FROM dataset_E WHERE split_E={literal(role)}').fetchone()
    checks['E_no_label_time_overlap']=all(bounds[stages[i]][1]<bounds[stages[i+1]][0] for i in range(len(stages)-1))
    (reports/'checks.json').write_text(json.dumps(checks,indent=2))
    if not all(checks.values()):raise ValueError(f'Dataset checks failed: {checks}')
    tables={
        'splits_R':"SELECT split_R,count(*) rows,count(DISTINCT loan_id) loans,sum(target_default) positive_rows,avg(target_default) positive_rate FROM dataset_R GROUP BY 1 ORDER BY 1",
        'splits_E':"SELECT split_E,count(*) rows,count(DISTINCT loan_id) loans,sum(target_default) positive_rows,avg(target_default) positive_rate,min(report_date) first_date,max(report_date) last_date,max(label_end_date) last_label_date FROM dataset_E GROUP BY 1 ORDER BY 1",
        'quarters_R':"SELECT quarter_ord,min(report_date) first_date,max(report_date) last_date,count(*) rows,sum(target_default) defaults,avg(target_default) default_rate FROM dataset_R GROUP BY 1 ORDER BY 1",
        'quarters_E':"SELECT quarter_ord,split_E,count(*) rows,sum(target_default) positive_rows,avg(target_default) positive_rate FROM dataset_E GROUP BY 1,2 ORDER BY 1",
        'states_R':"SELECT state,count(DISTINCT loan_id) loans,sum(target_default) defaults,sum(target_default) FILTER(WHERE split_R='train') train_defaults,sum(target_default) FILTER(WHERE split_R='validation') validation_defaults,sum(target_default) FILTER(WHERE split_R='test') test_defaults FROM dataset_R GROUP BY 1 ORDER BY train_defaults DESC",
        'states_E':"SELECT state,sum(target_default) FILTER(WHERE split_E='train') train_positives,sum(target_default) FILTER(WHERE split_E='calibration_fit') calibration_fit_positives,sum(target_default) FILTER(WHERE split_E='calibration_select') calibration_select_positives,sum(target_default) FILTER(WHERE split_E='test') test_positives FROM dataset_E GROUP BY 1 ORDER BY train_positives DESC",
        'censoring_E':"SELECT label_reason,target_default,count(*) rows FROM forecast_base GROUP BY 1,2 ORDER BY 1,2",
    }
    for name,sql in tables.items():con.execute(sql).df().to_csv(reports/f'{name}.csv',index=False)
    missing=[]
    for p in ['R','E']:
        total=con.execute(f'SELECT count(*) FROM dataset_{p}').fetchone()[0]
        for col in CORE_NUMERIC+CORE_CATEGORICAL+OPTIONAL_NUMERIC+OPTIONAL_CATEGORICAL:
            n,lo,hi=con.execute(f'SELECT count(*) FILTER(WHERE {col} IS NULL),min({col}),max({col}) FROM dataset_{p}').fetchone()
            missing.append({'protocol':p,'feature':col,'missing_rows':n,'missing_fraction':n/total,'min':lo,'max':hi})
    pd.DataFrame(missing).to_csv(reports/'feature_missingness.csv',index=False)
    summary={'R':con.execute('SELECT count(*) rows,count(DISTINCT loan_id) loans,sum(target_default) positives FROM dataset_R').df().to_dict('records')[0],
             'E':con.execute('SELECT count(*) rows,count(DISTINCT loan_id) loans,sum(target_default) positives FROM dataset_E').df().to_dict('records')[0],
             'E_censored_anchors':con.execute('SELECT count(*) FROM forecast_base WHERE target_default IS NULL').fetchone()[0],
             'E_purged_anchors':con.execute("SELECT count(*) FROM forecast_base f JOIN temporal_roles t USING(quarter_ord) WHERE target_default IS NOT NULL AND split_E='purged'").fetchone()[0],
             'elapsed_seconds':round(time.time()-t0,1),'checks_passed':len(checks)}
    (reports/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    con.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();make(a.output,a.work)
