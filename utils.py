import json
from pandas import DataFrame

def parse_sample(r):
    '''accepts sample response and parses it into a pandas dataframe'''
    return DataFrame(json.loads(r.json()))

def parse_post(string):
    '''accepts str of customer information and parses it into a pandas dataframe'''
    return DataFrame(json.loads(string))

def parse_metrics(metrics_r):
    '''accepts metrics response and returns column names and values as lists'''
    columns = [col for col in metrics_r.keys()]
    values = [round(val,2) for val in metrics_r.values()]
    return columns,values

def parse_sample_cols_vals(sample_r):
    sample_p = json.loads(sample_r)
    sample_cols = [col for col in sample_p.keys()]
    sample_vals = [list(val.values())[0] for val in sample_p.values()]
    return sample_cols,sample_vals
