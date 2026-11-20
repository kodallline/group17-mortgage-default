"""Read Freddie Mac Release 48 files without changing source folders."""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import duckdb

ORIG = ['fico','first_payment_ym','first_time_buyer','maturity_ym','msa','mi_pct','units','occupancy','orig_cltv','orig_dti','orig_upb','orig_ltv','orig_rate','channel','prepayment_penalty','amortization','state','property_type','zip3','loan_id','purpose','orig_term','borrowers','seller','super_conforming','pre_harp_id','program','harp','valuation_method','interest_only','vantage_score']
PERF = ['loan_id','report_ym','current_upb','delinquency','loan_age','remaining_term','defect_ym','modification','zero_code','zero_ym','current_rate','noninterest_upb','last_paid_ym','mi_recoveries','net_sales','nonmi_recoveries','total_expenses','legal_costs','maintenance','taxes_insurance','misc_expenses','actual_loss','cumulative_modification_cost','rate_step','deferral','eltv','zero_removal_upb','delinquent_interest','disaster','assistance','current_modification_cost','interest_bearing_upb','mi_cancellation','servicer','cramdown_cost']

def literal(s):
    return "'" + str(s).replace("'", "''") + "'"

def csv_scan(path, fields):
    cols = '{' + ','.join(f'{literal(x)}: \'VARCHAR\'' for x in fields) + '}'
    return f"read_csv({literal(path)}, delim='|', header=false, columns={cols}, nullstr='', strict_mode=true, parallel=true)"

def setup(db, temp):
    con = duckdb.connect(str(db))
    con.execute("SET threads=4")
    con.execute("SET memory_limit='2GB'")
    con.execute(f"SET temp_directory={literal(temp)}")
    con.execute("SET preserve_insertion_order=false")
    return con

def ingest(source_root, work, out):
    work.mkdir(parents=True, exist_ok=True)
    (out/'data').mkdir(parents=True, exist_ok=True)
    (out/'reports').mkdir(parents=True, exist_ok=True)
    con = setup(work/'prep.duckdb', work/'duckdb_temp')
    inventory=[]
    fingerprint_path=work/'cache_sources.json'
    fingerprints=json.loads(fingerprint_path.read_text()) if fingerprint_path.exists() else {}
    for year in range(2010,2019):
        t0=time.time()
        op=source_root/f'sample_{year}'/f'sample_orig_{year}.txt'
        pp=source_root/f'sample_{year}'/f'sample_perf_{year}.txt'
        for kind,p,ncols in [('orig',op,31),('perf',pp,35)]:
            with p.open() as f:
                first=f.readline().rstrip('\r\n').split('|')
            if len(first)!=ncols:
                raise ValueError(f'{p.name}: expected {ncols} columns, got {len(first)}')
            h=hashlib.sha256()
            with p.open('rb') as f:
                while b:=f.read(8*1024*1024): h.update(b)
            inventory.append(dict(file=p.name,year=year,kind=kind,bytes=p.stat().st_size,sha256=h.hexdigest(),columns=ncols))
        orig_out=work/f'orig_{year}.parquet'
        perf_out=work/f'perf_{year}.parquet'
        fingerprint={x['kind']:x['sha256'] for x in inventory if x['year']==year}
        changed=fingerprints.get(str(year))!=fingerprint
        if not orig_out.exists() or changed:
            con.execute(f"COPY (SELECT *, {year}::SMALLINT vintage FROM {csv_scan(op,ORIG)}) TO {literal(orig_out)} (FORMAT PARQUET, COMPRESSION ZSTD)")
        if not perf_out.exists() or changed:
            con.execute(f"""COPY (
                SELECT loan_id, try_cast(report_ym AS INTEGER) report_ym,
                    try_cast(current_upb AS DOUBLE) current_upb,
                    coalesce(upper(delinquency),'XX') delinquency_raw,
                    try_cast(delinquency AS SMALLINT) delinquency_months,
                    try_cast(loan_age AS SMALLINT) loan_age_reported,
                    try_cast(remaining_term AS SMALLINT) remaining_term,
                    coalesce(zero_code,'') zero_code,
                    try_cast(zero_ym AS INTEGER) zero_ym,
                    try_cast(current_rate AS DOUBLE) current_rate,
                    try_cast(noninterest_upb AS DOUBLE) noninterest_upb,
                    try_cast(eltv AS SMALLINT) eltv,
                    coalesce(modification,'') modification,
                    coalesce(disaster,'') disaster,
                    coalesce(assistance,'') assistance,
                    {year}::SMALLINT vintage
                FROM {csv_scan(pp,PERF)}
                WHERE try_cast(report_ym AS INTEGER)<=201812
            ) TO {literal(perf_out)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
        n=con.execute(f'SELECT count(*) FROM read_parquet({literal(orig_out)})').fetchone()[0]
        nperf,lo,mx=con.execute(f'SELECT count(*),min(report_ym),max(report_ym) FROM read_parquet({literal(perf_out)})').fetchone()
        print(json.dumps(dict(vintage=year,loans=n,monthly_rows_through_2018=nperf,min_month=lo,max_month=mx,seconds=round(time.time()-t0,1))),flush=True)
        fingerprints[str(year)]=fingerprint
    fingerprint_path.write_text(json.dumps(fingerprints,indent=2))
    (out/'reports'/'source_files.json').write_text(json.dumps(inventory,indent=2))
    con.execute(f"CREATE OR REPLACE TABLE orig_raw AS SELECT * FROM read_parquet({literal(work/'orig_*.parquet')})")
    if con.execute('SELECT count(*)-count(DISTINCT loan_id) FROM orig_raw').fetchone()[0]:
        raise ValueError('Origination loan identifiers are not unique')
    con.execute(f"CREATE OR REPLACE VIEW perf_raw AS SELECT * FROM read_parquet({literal(work/'perf_*.parquet')})")
    con.execute("""CREATE OR REPLACE TABLE monthly AS
        SELECT DISTINCT *, (report_ym//100*12+report_ym%100-1)::INTEGER month_ord,
            ((report_ym//100)*4+(report_ym%100-1)//3)::INTEGER quarter_ord,
            (coalesce(delinquency_months BETWEEN 3 AND 99,false) OR delinquency_raw='RA')::BOOLEAN is_default,
            (coalesce(delinquency_months BETWEEN 0 AND 99,false) OR delinquency_raw='RA')::BOOLEAN status_known
        FROM perf_raw WHERE report_ym%100 BETWEEN 1 AND 12""")
    conflicts=con.execute('SELECT count(*) FROM (SELECT loan_id,month_ord FROM monthly GROUP BY ALL HAVING count(*)>1)').fetchone()[0]
    if conflicts:
        raise ValueError(f'{conflicts} conflicting monthly keys; do not arbitrarily keep the last row')
    bad_ids=con.execute('SELECT count(*) FROM monthly m ANTI JOIN orig_raw o USING(loan_id)').fetchone()[0]
    if bad_ids: raise ValueError(f'{bad_ids} performance rows without origination data')
    print('Monthly unique rows:',con.execute('SELECT count(*) FROM monthly').fetchone()[0],flush=True)
    con.execute("""CREATE OR REPLACE TABLE history AS
        SELECT loan_id,min(month_ord) first_report_month,max(month_ord) last_report_month,
            min(month_ord) FILTER(WHERE is_default) first_default_month,
            min(month_ord) FILTER(WHERE delinquency_months BETWEEN 3 AND 99) first_90plus_month,
            min(month_ord) FILTER(WHERE delinquency_raw='RA') first_reo_month,
            min(month_ord) FILTER(WHERE zero_code='01') first_payoff_month,
            min(month_ord) FILTER(WHERE zero_code NOT IN ('','01')) first_other_exit_month,
            count(*) n_months_through_2018,
            count(*) FILTER(WHERE NOT status_known) unknown_status_months
        FROM monthly GROUP BY loan_id""")
    con.execute("""CREATE OR REPLACE TABLE loans AS
        SELECT o.loan_id,o.vintage,o.state,o.msa,o.zip3,
            try_strptime(o.first_payment_ym,'%Y%m')::DATE first_payment_date,
            (try_cast(left(first_payment_ym,4) AS INTEGER)*12+try_cast(right(first_payment_ym,2) AS INTEGER)-1)::INTEGER first_payment_month,
            CASE WHEN try_cast(fico AS INTEGER) BETWEEN 300 AND 850 THEN try_cast(fico AS SMALLINT) END fico,
            CASE WHEN try_cast(orig_dti AS INTEGER) BETWEEN 0 AND 65 THEN try_cast(orig_dti AS FLOAT) END orig_dti_pct,
            CASE WHEN try_cast(orig_ltv AS INTEGER) BETWEEN 1 AND 998 THEN try_cast(orig_ltv AS FLOAT) END orig_ltv_pct,
            CASE WHEN try_cast(orig_cltv AS INTEGER) BETWEEN 1 AND 998 THEN try_cast(orig_cltv AS FLOAT) END orig_cltv_pct,
            CASE WHEN try_cast(orig_upb AS DOUBLE)>0 THEN try_cast(orig_upb AS DOUBLE) END orig_upb,
            CASE WHEN try_cast(orig_rate AS DOUBLE) BETWEEN 0 AND 30 THEN try_cast(orig_rate AS FLOAT) END orig_rate_pct,
            CASE WHEN try_cast(orig_term AS INTEGER) BETWEEN 1 AND 600 THEN try_cast(orig_term AS SMALLINT) END orig_term_months,
            CASE WHEN try_cast(borrowers AS INTEGER) BETWEEN 1 AND 10 THEN try_cast(borrowers AS SMALLINT) END borrowers,
            CASE WHEN try_cast(units AS INTEGER) BETWEEN 1 AND 4 THEN try_cast(units AS SMALLINT) END units,
            CASE WHEN try_cast(mi_pct AS INTEGER) BETWEEN 0 AND 55 THEN try_cast(mi_pct AS FLOAT) END mi_pct,
            CASE WHEN property_type IN ('CP','CO','PU','SF','MH') THEN property_type ELSE 'UNKNOWN' END property_type,
            CASE WHEN occupancy IN ('P','I','S') THEN occupancy ELSE 'UNKNOWN' END occupancy,
            CASE WHEN purpose IN ('P','C','N','R') THEN purpose ELSE 'UNKNOWN' END purpose,
            amortization,channel,coalesce(seller,'OTHER') seller,harp,first_time_buyer,
            h.* EXCLUDE(loan_id)
        FROM orig_raw o LEFT JOIN history h USING(loan_id)""")
    con.execute(f"COPY loans TO {literal(out/'data'/'loans.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)")
    audit={
        'origination_loans':con.execute('SELECT count(*) FROM loans').fetchone()[0],
        'monthly_rows_through_2018':con.execute('SELECT count(*) FROM monthly').fetchone()[0],
        'monthly_rows_2013_2018':con.execute('SELECT count(*) FROM monthly WHERE report_ym BETWEEN 201301 AND 201812').fetchone()[0],
        'duplicate_rows_removed':con.execute('SELECT (SELECT count(*) FROM perf_raw)-(SELECT count(*) FROM monthly)').fetchone()[0],
        'defaults_before_2013':con.execute('SELECT count(*) FROM history WHERE first_default_month<2013*12').fetchone()[0],
        'defaults_2013_2018':con.execute('SELECT count(*) FROM history WHERE first_default_month BETWEEN 2013*12 AND 2019*12-1').fetchone()[0],
        'reo_without_prior_90plus_through_2018':con.execute('SELECT count(*) FROM history WHERE first_reo_month IS NOT NULL AND (first_90plus_month IS NULL OR first_reo_month<first_90plus_month)').fetchone()[0],
        'unknown_delinquency_codes':con.execute("SELECT delinquency_raw,count(*) n FROM monthly WHERE NOT status_known GROUP BY 1 ORDER BY n DESC").fetchall(),
        'state_counts':con.execute('SELECT state,count(*) FROM loans GROUP BY 1 ORDER BY 1').fetchall(),
    }
    (out/'reports'/'ingestion_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)
    con.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();ingest(a.source_root,a.work,a.output)
