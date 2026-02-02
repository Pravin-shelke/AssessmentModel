import pandas as pd
import json
import logging
from .config import Config

class DataLoader:
    def __init__(self):
        self.feature_columns = Config.FEATURE_COLUMNS

    def load_data(self, csv_file):
        """
        Load assessment data and parse nested JSON labelsAnswersMap.
        """
        try:
            logging.info(f"Loading data from {csv_file}")
            raw_df = pd.read_csv(csv_file, encoding='utf-8')
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

        records = []
        for idx, row in raw_df.iterrows():
            try:
                cell = row['labelsAnswersMap']
                label_list = json.loads(cell) if isinstance(cell, str) else []
            except Exception as e:
                logging.warning(f"Row {idx} JSON error: {e}")
                continue
                
            for label in label_list:
                indicator = label.get('soaId')
                answer = label.get('answer')
                if indicator:
                    rec = {col: row[col] for col in self.feature_columns}
                    rec['indicator'] = indicator
                    rec['answer'] = answer
                    records.append(rec)
                    
        if not records:
            raise ValueError("No valid records found in data file")
            
        long_df = pd.DataFrame(records)
        wide_df = long_df.pivot_table(
            index=self.feature_columns,
            columns='indicator',
            values='answer',
            aggfunc=lambda x: x.iloc[0] if len(x) else None
        ).reset_index()
        
        wide_df.columns.name = None
        logging.info(f"Data parsed successfully. Shape: {wide_df.shape}")
        return wide_df
