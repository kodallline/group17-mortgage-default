"""Download/cache public regional series. No credit account credentials needed."""
from __future__ import annotations
import argparse, hashlib, json, shutil, zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request,urlopen
import pandas as pd

FIPS = dict(AL='01',AK='02',AZ='04',AR='05',CA='06',CO='08',CT='09',DE='10',DC='11',FL='12',GA='13',HI='15',ID='16',IL='17',IN='18',IA='19',KS='20',KY='21',LA='22',ME='23',MD='24',MA='25',MI='26',MN='27',MS='28',MO='29',MT='30',NE='31',NV='32',NH='33',NJ='34',NM='35',NY='36',NC='37',ND='38',OH='39',OK='40',OR='41',PA='42',RI='44',SC='45',SD='46',TN='47',TX='48',UT='49',VT='50',VA='51',WA='53',WV='54',WI='55',WY='56')

def fetch(url,path,data=None):
    if path.exists(): return path
    headers={'User-Agent':'Mozilla/5.0 (university mortgage research)'}
    if data is not None: headers['Content-Type']='application/json'
    request=Request(url,headers=headers,data=data)
    with urlopen(request,timeout=90) as r:
        b=r.read(100_000_000)
    if len(b)>=100_000_000: raise ValueError('Source exceeds download cap')
    path.write_bytes(b)
    return path

def prepare(out):
    raw=out/'external'/'raw';raw.mkdir(parents=True,exist_ok=True)
    ext=out/'external'
    hpi_url='https://www.freddiemac.com/fmac-resources/research/docs/fmhpi_master_file.csv'
    hpi_path=raw/'fmhpi_master_file.csv'
    # An already downloaded official snapshot can be reused offline.
    probe=Path('work/macro_probe/hpi.csv')
    if probe.exists() and not hpi_path.exists(): shutil.copyfile(probe,hpi_path)
    fetch(hpi_url,hpi_path)
    hpi=pd.read_csv(hpi_path,low_memory=False)
    hpi=hpi[(hpi.GEO_Type=='State') & hpi.GEO_Name.isin(FIPS)].copy()
    hpi['month_ord']=hpi.Year*12+hpi.Month-1
    hpi=hpi[hpi.Year.between(2008,2018)]
    hpi=hpi.rename(columns={'GEO_Name':'state','Index_NSA':'hpi_nsa','Index_SA':'hpi_sa'})
    hpi[['state','month_ord','hpi_nsa','hpi_sa']].to_parquet(ext/'hpi_state_monthly.parquet',index=False)
    print('HPI:',len(hpi),'rows;',hpi.state.nunique(),'states',flush=True)
    by_series={f'LAUST{f}0000000000003':s for s,f in FIPS.items()}
    items=list(by_series)
    def bls_chunk(i):
        ids=items[i:i+25]
        body={'seriesid':ids,'startyear':'2009','endyear':'2018','annualaverage':True}
        p=raw/f'bls_states_{i//25+1}.json'
        fetch('https://api.bls.gov/publicAPI/v2/timeseries/data/',p,json.dumps(body).encode())
        obj=json.loads(p.read_text())
        if obj.get('status')!='REQUEST_SUCCEEDED':raise ValueError(obj)
        rows=[]
        for series in obj['Results']['series']:
            for x in series['data']:
                if x['period']=='M13':
                    rows.append({'state':by_series[series['seriesID']],'year':int(x['year']),'unemployment_rate_pct':float(x['value'])})
        return rows
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for chunk in pool.map(bls_chunk,range(0,len(items),25)):rows.extend(chunk)
    u=pd.DataFrame(rows).sort_values(['state','year'])
    if len(u)!=51*10 or u.duplicated(['state','year']).any():raise ValueError(f'Incomplete BLS annual data: {len(u)} rows')
    u['unemployment_change_pp']=u.groupby('state').unemployment_rate_pct.diff()
    u.to_parquet(ext/'unemployment_state_annual.parquet',index=False)
    print('BLS:',len(u),'annual rows',flush=True)
    gdp_url='https://apps.bea.gov/regional/zip/SAGDP.zip'
    gdp_path=raw/'SAGDP.zip';fetch(gdp_url,gdp_path)
    if not zipfile.is_zipfile(gdp_path):raise ValueError('BEA returned non-ZIP content')
    with zipfile.ZipFile(gdp_path) as z:
        candidates=[n for n in z.namelist() if ('SAGDP9N__ALL_AREAS' in n or 'SAGDP9__ALL_AREAS' in n) and n.endswith('.csv')]
        if len(candidates)!=1:raise ValueError(f'Cannot choose real GDP table: {candidates}')
        member=candidates[0]
        with z.open(member) as f:g=pd.read_csv(f,dtype=str,encoding='utf-8-sig',low_memory=False)
    g['fips_clean']=g.GeoFIPS.str.replace('"','',regex=False).str.strip()
    rev={v+'000':k for k,v in FIPS.items()}
    g=g[(g.LineCode.str.strip()=='1') & g.fips_clean.isin(rev)].copy()
    year_cols=[c for c in g if c.isdigit() and 2009<=int(c)<=2018]
    g=g.melt(id_vars=['fips_clean'],value_vars=year_cols,var_name='year',value_name='real_gdp')
    g['state']=g.fips_clean.map(rev);g['year']=g.year.astype(int)
    g['real_gdp']=pd.to_numeric(g.real_gdp.str.replace(',','',regex=False),errors='coerce')
    g=g.sort_values(['state','year'])
    g['gdp_growth_pct']=g.groupby('state').real_gdp.pct_change(fill_method=None)*100
    if len(g)!=510 or g.real_gdp.isna().any():raise ValueError(f'Incomplete BEA GDP: {len(g)} rows')
    g[['state','year','real_gdp','gdp_growth_pct']].to_parquet(ext/'gdp_state_annual.parquet',index=False)
    print('BEA:',len(g),'annual rows; table',member,flush=True)
    (ext/'state_fips.json').write_text(json.dumps(FIPS,indent=2))
    sources=[]
    for p in sorted(raw.iterdir()):
        if p.is_file():sources.append({'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'hpi_url':hpi_url,'bls_url':'https://api.bls.gov/publicAPI/v2/timeseries/data/','bls_series':by_series,'bls_startyear':2009,'bls_endyear':2018,'bea_url':gdp_url,'bea_member':member,'files':sources,'vintage_note':'Current revised historical series, not historical releases. Calendar lags do not remove revision look-ahead.'}
    (ext/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    annual=u.merge(g[['state','year','gdp_growth_pct']],on=['state','year'],validate='one_to_one')
    annual.to_parquet(ext/'state_annual.parquet',index=False)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True)
    prepare(a.parse_args().output)
