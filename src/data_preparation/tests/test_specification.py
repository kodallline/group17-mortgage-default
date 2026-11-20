"""Hand-specified target edge cases and split guarantees."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare import label_future,split_loans,temporal_role
import pandas as pd

def test_default_at_last_horizon_month():
    assert label_future(100,103,None,None,[(101,True,''),(102,True,''),(103,True,'')])==(1,'default')

def test_default_outside_horizon_is_not_positive():
    assert label_future(100,104,None,None,[(101,True,''),(102,True,''),(103,True,'')])==(0,'complete')

def test_missing_middle_month_is_censored():
    assert label_future(100,None,None,None,[(101,True,''),(103,True,''),(104,True,'')])[0] is None

def test_unknown_future_status_is_censored():
    assert label_future(100,None,None,None,[(101,True,''),(102,False,''),(103,True,'')])[0] is None

def test_confirmed_payoff_before_horizon_is_negative():
    assert label_future(100,None,102,None,[(101,True,''),(102,True,'01')])==(0,'payoff')

def test_payoff_with_missing_prior_month_is_censored():
    assert label_future(100,None,102,None,[(102,True,'01')])[0] is None

def test_other_exit_is_not_treated_as_payoff():
    assert label_future(100,None,None,102,[(101,True,''),(102,True,'15')])==(None,'other_exit')

def test_observed_default_is_retained_before_exit():
    assert label_future(100,101,None,102,[(101,True,''),(102,True,'03')])==(1,'default')

def test_right_censoring_is_not_a_negative():
    assert label_future(100,None,None,None,[(101,True,'')])[0] is None

def test_split_is_disjoint_and_reproducible():
    frame=pd.DataFrame({'loan_id':[f'L{i:03}' for i in range(100)],'state':['CA']*100,'first_default_month':[None]*50+[102]*50})
    a=split_loans(frame);b=split_loans(frame.sample(frac=1,random_state=7))
    assert a.equals(b)
    assert a.loan_id.is_unique
    assert a.split_R.value_counts().to_dict()=={'train':60,'validation':20,'test':20}

def test_calendar_roles_purge_intermediate_quarters():
    assert temporal_role(2015*4)=='purged'
    assert temporal_role(2017*4)=='purged'
    assert temporal_role(2018*4+2)=='test'
    assert temporal_role(2018*4+3)=='purged'
