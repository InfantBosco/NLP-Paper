import pandas as pd
from typing import List, Dict, Any

def generate_summary_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generates comparison summary dataframe for metrics reporting."""
    return pd.DataFrame(results_list)
