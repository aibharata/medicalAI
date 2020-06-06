import pytest
import medicalai as ai
from medicalai import dataAnalyzer as DAZY
import pandas as pd

def test_dataAnalyzer():
    df1 = pd.DataFrame({'ID': [0, 1, 2]})
    df2 = pd.DataFrame({'ID': [2, 3, 4]})
    print(10*'-','Checking dataAnalyzer',10*'-')
    assert DAZY.check_dataset_leakage(df1, df2, 'ID')==True, "check_dataset_leakage Failed - when leakage present"

    df2 = pd.DataFrame({'ID': [5, 7, 4]})
    assert DAZY.check_dataset_leakage(df1, df2, 'ID')==False, "check_dataset_leakage Failed - when leakage absent"

