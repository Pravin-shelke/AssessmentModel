import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.feature_modes = {}
        
    def fit_transform(self, df, feature_columns):
        """
        Fit encoders and transform features, including interaction terms.
        """
        df_encoded = df.copy()
        
        # Numeric handling
        if 'area' in df_encoded.columns:
            df_encoded['area'] = pd.to_numeric(df_encoded['area'], errors='coerce').fillna(0)
            
        # Create Interaction Features (New Accuracy Improvement)
        if 'crop_name' in df_encoded.columns and 'country_code' in df_encoded.columns:
            df_encoded['crop_country'] = df_encoded['crop_name'].astype(str) + "_" + df_encoded['country_code'].astype(str)
            feature_columns.append('crop_country') # Add dynamically
            
        for col in feature_columns:
            if col == 'area' or col not in df_encoded.columns:
                continue
                
            le = LabelEncoder()
            values = df_encoded[col].astype(str).fillna('Unknown')
            classes = values.unique().tolist()
            if 'Unknown' not in classes:
                classes.append('Unknown')
            
            le.fit(classes)
            df_encoded[col + '_encoded'] = le.transform(values)
            self.label_encoders[col] = le
            self.feature_modes[col] = values.mode(dropna=True)[0]
            
        self.feature_cols = feature_columns # Update stored list
        return df_encoded
        
    def transform(self, input_data):
        """
        Transform single input dict for prediction.
        """
        # Create interaction term on fly
        if 'crop_name' in input_data and 'country_code' in input_data:
            input_data['crop_country'] = str(input_data['crop_name']) + "_" + str(input_data['country_code'])
            
        encoded_vector = []
        for col in self.feature_cols:
            if col == 'area':
                val = input_data.get(col, 0)
                encoded_vector.append(float(val) if val is not None else 0.0)
                continue
                
            if col in self.label_encoders:
                le = self.label_encoders[col]
                val = str(input_data.get(col, 'Unknown'))
                try:
                    enc_val = le.transform([val])[0]
                except ValueError:
                    fallback = self.feature_modes.get(col, le.classes_[0])
                    enc_val = le.transform([fallback])[0]
                encoded_vector.append(enc_val)
            else:
                 # Boolean/Numeric fallback
                val = input_data.get(col, 0)
                if isinstance(val, bool): val = int(val)
                encoded_vector.append(val)
                
        return encoded_vector

    def get_feature_names(self):
        # Return encoded names for model training
        return [c + '_encoded' if c in self.label_encoders else c for c in self.feature_cols]
