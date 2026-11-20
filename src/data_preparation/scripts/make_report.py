"""Create the participant-1 data card, dictionary, figures and handoff report."""
from __future__ import annotations
import argparse,base64,html,json,os,tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='freddie_mpl_'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ingest import setup
from prepare import CORE_NUMERIC,CORE_CATEGORICAL,OPTIONAL_NUMERIC,OPTIONAL_CATEGORICAL

DESCRIPTIONS={
 'orig_dti_pct':('Отношение ежемесячных долговых платежей к ежемесячному доходу при выдаче','%','999 заменен пропуском. Не равен DTI в статье.'),
 'current_rate_pct':('Ставка в отчетном месяце','% годовых','В Standard Dataset преобладают фиксированные ставки.'),
 'current_ltv_proxy_pct':('Индексированный расчетный LTV','%','OLTV * текущий долг / исходный долг * HPI при выдаче / доступный HPI. HPI отстает на три месяца.'),
 'seniority_months':('Приближенный возраст кредита','месяцы','Месяц выдачи приближен первым платежом минус два месяца. Возраст ограничен снизу нулем.'),
 'log_orig_property_value_proxy':('Логарифм расчетной стоимости залога при выдаче','ln(USD)','Расчетная стоимость = исходный долг / (OLTV / 100). Не исходная оценка недвижимости.'),
 'gdp_growth_lag2y_pct':('Рост реального ВВП штата за год, отстающий на два календарных года','%','BEA SAGDP9, вся экономика. Современная пересмотренная версия.'),
 'unemployment_change_lag2y_pp':('Изменение среднегодовой безработицы штата за год, отстающий на два года','процентные пункты','BLS LAUS, M13, без сезонной корректировки.'),
 'hpi_growth_lag3m_pct':('Годовой рост индекса цен жилья штата, отстающий на три месяца','%','FMHPI, Index_NSA. Современная пересмотренная версия.'),
 'regional_default_rate_lag4q':('Доля первых дефолтов среди наблюдаемых кредитов штата четыре квартала назад','доля от 0 до 1','R: только обучающие кредиты. E: ранее наблюдаемая история портфеля.'),
 'property_type':('Тип недвижимости','категория','CO, CP, PU, SF, MH. Неизвестные значения обозначены UNKNOWN.'),
 'fico':('Classic FICO при выдаче','баллы','Только диапазон от 300 до 850. 9999 заменен пропуском.'),
 'orig_ltv_pct':('LTV при выдаче','%','999 заменен пропуском. Высокие значения HARP сохранены.'),
 'orig_cltv_pct':('Совокупный LTV при выдаче','%','999 заменен пропуском.'),
 'orig_rate_pct':('Процентная ставка при выдаче','% годовых','Дополнительный признак.'),
 'orig_term_months':('Срок кредита при выдаче','месяцы','Дополнительный признак.'),
 'borrowers':('Число заемщиков','человек','До второго квартала 2018 код 2 означает два и более. Нельзя интерпретировать весь столбец как точный счет.'),
 'units':('Число жилых единиц объекта','единицы','99 заменен пропуском.'),
 'seller_mean_ltv_lag4q':('Средний исходный LTV по кредитору четыре квартала назад','%','OTHER не объединен в псевдобанк. Пропуски сохранены.'),
 'seller_mean_dti_lag4q':('Средний исходный DTI по кредитору четыре квартала назад','%','R: обучение. E: прошлая история. Пропуски сохранены.'),
 'seller_mean_balance_change_lag4q':('Среднее годовое изменение долга по кредитору четыре квартала назад','доля','Требуется наблюдение того же кредита ровно 12 месяцев назад.'),
 'occupancy':('Назначение проживания','категория','P основное жилье, I инвестиционное, S второе жилье.'),
 'purpose':('Цель кредита','категория','P покупка, C рефинансирование с выдачей денег, N другое рефинансирование, R неуточненное.'),
 'channel':('Канал выдачи','категория','B, C, R, T по словарю источника.'),
}

def fmt(x):return f'{int(x):,}'.replace(',',' ')
def main(out,work):
    reports=out/'reports';fig=reports/'figures';fig.mkdir(exist_ok=True)
    con=setup(work/'prep.duckdb',work/'duckdb_temp')
    stats=json.loads((reports/'summary.json').read_text())
    ingestion=json.loads((reports/'ingestion_audit.json').read_text())
    r=pd.read_csv(reports/'splits_R.csv');e=pd.read_csv(reports/'splits_E.csv')
    sr=pd.read_csv(reports/'states_R.csv');se=pd.read_csv(reports/'states_E.csv')
    sr['eligible_R_train_30']=sr.train_defaults>=30
    se['eligible_E_regional_calibration']=(se.train_positives>=30)&(se.calibration_fit_positives>=10)&(se.calibration_select_positives>=5)
    sr.to_csv(reports/'regional_eligibility_R.csv',index=False)
    se.to_csv(reports/'regional_eligibility_E.csv',index=False)
    flow=[('Исходные кредиты',450000),('Кредиты в 50 штатах и DC',con.execute('SELECT count(*) FROM splits').fetchone()[0]),('Кредиты в квартальной выборке R',stats['R']['loans']),('Квартальные наблюдения R',stats['R']['rows']),('Кредиты в прогнозной выборке E',stats['E']['loans']),('Квартальные прогнозы E',stats['E']['rows']),('Исключенные окна E с неполной/прерванной историей',stats['E_censored_anchors']),('Дополнительно исключенные окна E между временными блоками',stats['E_purged_anchors'])]
    pd.DataFrame(flow,columns=['step','count']).to_csv(reports/'data_flow.csv',index=False)
    dq=[]
    for c in CORE_NUMERIC+CORE_CATEGORICAL+OPTIONAL_NUMERIC+OPTIONAL_CATEGORICAL:
        title,unit,note=DESCRIPTIONS[c]
        dq.append({'column':c,'role':'core' if c in CORE_NUMERIC+CORE_CATEGORICAL else 'optional_extended','description_ru':title,'unit':unit,'note_ru':note})
    for c,title in [('loan_id','Идентификатор кредита'),('state','Штат или DC'),('report_date','Конец отчетного месяца'),('quarter_ord','Код календарного квартала: год * 4 + номер квартала от 0'),('target_default','Исход, определение зависит от протокола'),('split_R','Группа кредита в R'),('split_E','Календарный блок E'),('label_end_date','Конец будущего трехмесячного окна E')]:
        dq.append({'column':c,'role':'metadata_or_target','description_ru':title,'unit':'','note_ru':'Не использовать как автоматически выбранный признак модели.'})
    pd.DataFrame(dq).to_csv(out/'data_dictionary.csv',index=False)
    mapping=[
        ('DTI','orig_dti_pct','Изменено определение: платежи к доходу вместо долга к годовому доходу'),
        ('Interest rate type','amortization','FRM, константа. Исключено из основного набора'),
        ('Current interest rate','current_rate_pct','Ставка за отчетный месяц'),
        ('Current LTV','current_ltv_proxy_pct','Расчетный показатель для общего периода. ELTV источника оставлен только в аудите'),
        ('Property type','property_type','Категории источника'),
        ('Seniority','seniority_months','Приближенный возраст в месяцах'),
        ('Log valuation','log_orig_property_value_proxy','Стоимость восстановлена из UPB и OLTV'),
        ('Borrower employment','Нет','Данные не предоставлены. Не имитируются'),
        ('Log income','Нет','Данные не предоставлены. Не имитируются'),
        ('Regional GDP growth','gdp_growth_lag2y_pct','Штат вместо NUTS2, лаг два года'),
        ('Regional unemployment change','unemployment_change_lag2y_pp','Штат, изменение в процентных пунктах, лаг два года'),
        ('Regional house price growth','hpi_growth_lag3m_pct','Штат вместо NUTS3, годовой рост с лагом три месяца'),
        ('Regional default rate','regional_default_rate_lag4q','Лаг четыре квартала, строго исторический расчет'),
        ('Originator controls','seller_mean_*_lag4q','Дополнительные признаки. Объединенная категория OTHER не считается отдельным банком'),
    ]
    pd.DataFrame(mapping,columns=['article_variable','dataset_column','difference_ru']).to_csv(out/'article_feature_mapping.csv',index=False)
    # All figures use aggregate data; no borrower-level output is plotted.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    q=pd.read_csv(reports/'quarters_R.csv')
    labels=[f'{int(x)//4} Q{int(x)%4+1}' for x in q.quarter_ord]
    f,ax=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    ax[0].plot(range(len(q)),q.rows/1000,color='#155f79',marker='o',markersize=3)
    ax[0].set_ylabel('Observed loans (thousands)');ax[0].set_title('Quarterly panel coverage and first observed defaults')
    ax[1].bar(range(len(q)),q.defaults,color='#d47e35');ax[1].set_ylabel('First defaults (loans)')
    ax[1].set_xticks(range(0,len(q),2),labels[::2],rotation=45,ha='right')
    f.savefig(fig/'quarterly_coverage.png',dpi=180);plt.close(f)
    top=sr.head(15).sort_values('train_defaults')
    f,ax=plt.subplots(figsize=(9,6),layout='constrained')
    ax.barh(top.state,top.train_defaults,color='#155f79');ax.set_xlabel('First defaults in R training loans')
    ax.set_title('Regional sample size: top 15 states by training events')
    f.savefig(fig/'regional_events.png',dpi=180);plt.close(f)
    f,ax=plt.subplots(figsize=(10,2.7),layout='constrained')
    names=['train','validation','calibration_fit','calibration_select','test']
    colors=['#155f79','#498998','#ce9b34','#dfbc69','#99593a']
    for i,(name,color) in enumerate(zip(names,colors)):
        row=e[e.split_E==name].iloc[0]
        start=pd.Timestamp(row.first_date);end=pd.Timestamp(row.last_date)
        x=start.year*4+(start.month-1)//3
        z=end.year*4+(end.month-1)//3
        ax.broken_barh([(x-2013*4-.4,z-x+.8)],(i-.3,.6),facecolors=color)
    ax.set_yticks(range(5),['Train','Validation','Calibration fit','Calibration selection','Final test'])
    ax.set_xticks([i*4 for i in range(6)],[str(2013+i) for i in range(6)])
    ax.set_xlim(-1,23);ax.set_title('Prospective protocol E: chronological blocks with purged gaps')
    ax.set_xlabel('Feature observation year; outcomes extend three months forward')
    f.savefig(fig/'temporal_splits.png',dpi=180);plt.close(f)
    missing=pd.read_csv(reports/'feature_missingness.csv')
    show=missing[(missing.protocol=='R') & (missing.missing_fraction>0)].sort_values('missing_fraction')
    f,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.barh(show.feature,show.missing_fraction*100,color='#155f79')
    ax.set_xlabel('Missing observations (%)');ax.set_title('Missing values preserved for training-only imputation')
    f.savefig(fig/'missingness.png',dpi=180);plt.close(f)
    # Histograms use training rows only.
    train=con.execute("SELECT current_ltv_proxy_pct,fico,target_default FROM dataset_R WHERE split_R='train' ORDER BY hash(loan_id,quarter_ord,42) LIMIT 100000").df()
    f,ax=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    ax[0].hist(train.current_ltv_proxy_pct.dropna(),bins=60,range=(0,160),color='#155f79');ax[0].set_xlabel('Indexed LTV proxy (%)');ax[0].set_ylabel('Training observations')
    ax[1].hist(train.fico.dropna(),bins=35,color='#d47e35');ax[1].set_xlabel('Origination Classic FICO')
    f.suptitle('Training feature distributions (100,000-row descriptive sample)')
    f.savefig(fig/'training_distributions.png',dpi=180);plt.close(f)
    # Source rows preserve events, including disaster-related flags, for audit only.
    event_disaster=con.execute("""SELECT l.state,m.report_ym,m.disaster,count(*) events
      FROM history h JOIN monthly m ON h.loan_id=m.loan_id AND h.first_default_month=m.month_ord
      JOIN loans l ON h.loan_id=l.loan_id
      WHERE m.report_ym BETWEEN 201301 AND 201812
      GROUP BY 1,2,3 ORDER BY 2,1""").df()
    event_disaster.to_csv(reports/'event_disaster_audit.csv',index=False)
    comparison='\n'.join(f'| {a} | {b} | {c} |' for a,b,c in mapping)
    def mdtable(df,cols):
        lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
        for _,row in df.iterrows():
            lines.append('| '+' | '.join(str(row[c]) for c in cols)+' |')
        return '\n'.join(lines)
    text=f'''# Подготовка данных: результат участника 1

Дата подготовки: 16 ноября 2026 года. Преподаватель разрешил аналогичные данные при реализации и сравнении методов на одной выборке.

## Полученные данные

Использованы девять официальных Sample выгрузок Freddie Mac за годы выдачи с 2010 по 2018. Это {fmt(450000)} кредитов. Истории до конца 2018 содержат {fmt(ingestion['monthly_rows_through_2018'])} ежемесячных записей; в основном окне с 2013 по 2018 год находятся {fmt(ingestion['monthly_rows_2013_2018'])} записей.

В основном окне во всех территориях найден {fmt(ingestion['defaults_2013_2018'])} первый наблюдаемый дефолт. После исключения территорий и подготовки квартальной панели R сохранен {fmt(stats['R']['positives'])} первый дефолт. Дефолтный исход определен как первый наблюдаемый статус 90+ дней, включая редкий RA как признак приобретения недвижимости в REO. Это приближение порога более 90 дней из статьи, а не буквальное равенство порогов. В полной истории найдено только три случая RA без предшествующего наблюдаемого числового статуса 90+.

Выгрузки одинакового размера по годам выдачи не являются случайной выборкой всего ипотечного рынка. Веса, восстанавливающие рыночную структуру, не использовались.

## Два подготовленных протокола

**R, воспроизведение подхода:** {fmt(stats['R']['rows'])} квартальных наблюдений по {fmt(stats['R']['loans'])} кредитам. В квартале сохраняется последнее допустимое наблюдение до выхода или первое событие дефолта. После первого дефолта строки кредита исключены. Цель соответствует событию в наблюдаемом квартале и не объявляется будущим трехмесячным прогнозом.

Кредиты случайно распределены в пропорции около 60/20/20 с учетом штата и наличия дефолта. Идентификаторы кредитов между группами не пересекаются. Доля дефолтных кредитов в R составляет {stats['R']['positives']/stats['R']['loans']*100:.3f}%. Это доля среди кредитов за весь период. Доля положительных квартальных строк составляет {stats['R']['positives']/stats['R']['rows']*100:.3f}%. Эти две величины нельзя путать.

{mdtable(r,['split_R','rows','loans','positive_rows'])}

**E, собственное расширение:** {fmt(stats['E']['rows'])} прогнозных наблюдений по {fmt(stats['E']['loans'])} кредитам, {fmt(stats['E']['positives'])} положительных исходов. В конце квартала активный кредит без прежнего дефолта получает метку первого дефолта в следующих трех месяцах. Наблюдения с неизвестным дальнейшим исходом исключены. Подтвержденное полное погашение без дефолта считается отрицательным исходом до завершения кредита. Продажа, передача и прочие выходы без наблюдаемого дефолта не подменяются благополучным погашением.

Между обучением, валидацией, обучением калибраторов, выбором калибровки и тестом оставлены кварталы, предотвращающие пересечение целевых окон. Исключено {fmt(stats['E_purged_anchors'])} таких промежуточных наблюдений. Еще {fmt(stats['E_censored_anchors'])} прогнозных окон исключены из-за прерванной или неполной истории. В E один кредит может встречаться в разных календарных блоках: это мониторинг обслуживаемого портфеля. Для утверждений о новых кредитах потребуется отдельная проверка.

{mdtable(e,['split_E','rows','positive_rows','first_date','last_date','last_label_date'])}

Последняя дата признаков E: 30 сентября 2018 года. Последние целевые исходы наблюдаются до декабря 2018 включительно. Промежуточные кварталы уже исключены из основного файла E.

## Признаки и обработка

Основной набор включает девять численных признаков и тип недвижимости. Дополнительный набор содержит Classic FICO, исходные характеристики кредита, признаки назначения и исторические агрегаты кредитора. Два набора перечислены в config.json, одинаковый набор должен использоваться у сравниваемых моделей.

Специальные коды отсутствующих значений заменены пропусками. Доход и занятость не имитируются. Пропуски не заполнялись по всей выборке. Заполнение, масштабирование, кодирование категорий, выбор признаков и возможное отсечение хвостов выполняются участником 2 только по обучению. Валидация, калибровка и тест сохраняют естественную частоту дефолта.

Расчетный текущий LTV использует исходный LTV, отношение текущего долга к исходному и изменение FMHPI. Месяц выдачи для индекса приближен первым платежом минус два месяца. Текущий индекс отстает на три месяца. Пока индекс месяца выдачи еще не мог быть доступен с этим лагом, используется только корректировка на остаток долга. Это расчетная величина, чувствительная к приближению даты и региональному индексу. Официальный ELTV, доступный лишь с апреля 2017, сохранен отдельно для аудита, чтобы не ограничивать весь анализ поздним периодом.

Годовые ВВП и безработица отстают на два календарных года. Это консервативный лаг относительно полного годового показателя, отличающийся от статьи. Лаги не устраняют последующие пересмотры. Использованы современные исправленные исторические выгрузки, поэтому эксперимент является проверкой временного переноса на пересмотренных данных, а не полностью восстановленным real-time тестом. Отчетный месяц Freddie Mac также не равен точной дате публикации: исторические циклы учета могли сдвигать доступность записи.

Региональные дефолты и агрегаты кредиторов отстают на четыре квартала. Для R они рассчитываются только из обучающих кредитов. Для E используются уже наблюдаемые прошлые кварталы портфеля. Категория OTHER не объединяется в искусственный банк. Границы штатов используются вместо NUTS.

VantageScore 4.0, признаки дефолта, будущие даты, убытки, коды выхода, информация о последующем обслуживании и аудиторские признаки не входят в основной набор. Коды бедствий и помощи сохранены для отдельного аудита, а не автоматически добавлены в модели.

## Соответствие статье

| Переменная статьи | Подготовленное поле | Отличие |
| --- | --- | --- |
{comparison}

Standard Dataset содержит фиксированную полностью амортизируемую ипотеку. Выводы авторов о плавающей ставке на этой выборке непосредственно не проверяются. Годы выдачи ограничены 2010 и последующими годами, тогда как у авторов включены более старые кредиты. Общий календарный период наблюдений сохранен, популяция отличается.

## Достаточность региональных данных

В R {int(sr.eligible_R_train_30.sum())} штатов имеют не менее 30 обучающих дефолтов. Для E {int(se.eligible_E_regional_calibration.sum())} штатов одновременно имеют не менее 30 событий в обучении, 10 в C1 и 5 в C2. Это ориентир достаточности, зафиксированный без использования результатов моделей на тесте. Полные таблицы приведены в regional_eligibility_R.csv и regional_eligibility_E.csv. Нельзя предполагать, что все 51 регион одинаково пригодны для отдельного обучения и калибровки.

В остальных регионах для собственного расширения разумно использовать общий калибратор или регуляризованные поправки, а не свободно оценивать много параметров по единичным дефолтам. Проверка этого решения относится к участнику 3.

## Что передано команде

data/replication_R.parquet: готовая квартальная выборка R.

data/forecast_E.parquet: готовая выборка E с пятью календарными блоками.

data/loan_splits_R.parquet: единое распределение кредитов R.

data/loans.parquet и data/audit_only.parquet: происхождение и диагностика. Эти файлы не являются готовыми матрицами признаков.

config.json: основной и дополнительный наборы признаков, целевые поля и ограничения.

data_dictionary.csv и article_feature_mapping.csv: словарь полей и соответствие статье.

scripts/load_data.py: загрузка одной группы с явным списком разрешенных признаков.

reports/: численность, пропуски, число событий, правила разбиений и проверки.

## Проверки и границы результата

Пройдены 17 проверок структуры и хронологии на полных данных. Отдельно выполнены тесты граничных случаев целевой переменной и независимая сверка выборки прогнозных окон с исходной месячной историей. Итог независимой проверки сохранен в independent_checks.json.

Работа участника 1 завершает подготовку данных и протокола. Обучение Logistic, GB и XGB, калибровка и сравнение моделей еще не проводились. Полученные количества дефолтов не являются результатами качества моделей.

## Источники

Freddie Mac Single-Family Loan-Level Dataset, Release 48, October 2026: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

Словарь Freddie Mac: https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf

FMHPI: https://www.freddiemac.com/research/indices/house-price-index

BLS LAUS, годовые средние M13: https://www.bls.gov/lau/ и https://api.bls.gov/publicAPI/v2/timeseries/data/

BEA SAGDP9: https://www.bea.gov/data/gdp/gdp-state и https://apps.bea.gov/regional/zip/SAGDP.zip

Исходная статья: https://doi.org/10.1093/jjfinec/nbab010

Файлы источников, размеры и SHA-256 сохранены в source_files.json и external/source_manifest.json. Официальные описания полей и текущая версия региональных рядов сохранены для воспроизводимости.
'''
    (reports/'data_report.md').write_text(text)
    # Self-contained HTML, so charts remain available after sharing the report file.
    def image(name):
        b=base64.b64encode((fig/name).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b}" alt="{html.escape(name)}">'
    sections=[]
    for paragraph in text.split('\n\n'):
        if paragraph.startswith('# '):sections.append('<h1>'+html.escape(paragraph[2:])+'</h1>')
        elif paragraph.startswith('## '):sections.append('<h2>'+html.escape(paragraph[3:])+'</h2>')
        elif paragraph.startswith('| '):
            rows=[x for x in paragraph.splitlines() if not x.startswith('| ---')]
            body=[]
            for i,row in enumerate(rows):
                cells=row.strip('|').split('|');tag='th' if i==0 else 'td'
                body.append('<tr>'+''.join(f'<{tag}>{html.escape(x.strip())}</{tag}>' for x in cells)+'</tr>')
            sections.append('<div class="table-wrap"><table>'+''.join(body)+'</table></div>')
        else:sections.append('<p>'+html.escape(paragraph).replace('**','')+'</p>')
    sections.insert(4,image('quarterly_coverage.png'))
    sections.append('<h2>Графики для презентации</h2>'+''.join(image(n) for n in ['temporal_splits.png','regional_events.png','missingness.png','training_distributions.png']))
    css='''body{font-family:Arial,sans-serif;color:#183340;background:#f2f5f6;margin:0;line-height:1.65}main{max-width:1080px;margin:32px auto;background:white;padding:44px 52px}h1{font-size:30px;line-height:1.2;color:#155f79}h2{font-size:21px;margin-top:35px;border-top:1px solid #dce5e9;padding-top:22px}p{font-size:15px}img{display:block;max-width:100%;height:auto;margin:28px auto}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #dce5e9;padding:9px 10px}th{background:#edf4f6}p{overflow-wrap:anywhere}@media(max-width:700px){main{padding:22px;margin:0}h1{font-size:25px}}@media print{body{background:white}main{margin:0;padding:15px}img,table{break-inside:avoid}}'''
    (reports/'data_report.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Подготовка ипотечных данных</title><style>'+css+'</style><main>'+''.join(sections)+'</main></html>')
    print(json.dumps({'regional_R_eligible':int(sr.eligible_R_train_30.sum()),'regional_E_eligible':int(se.eligible_E_regional_calibration.sum()),'report':'data_report.html','figures':5},ensure_ascii=False),flush=True)
    con.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();main(a.output,a.work)
