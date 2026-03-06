import pandas as pd
import json
import os
from sklearn.preprocessing import LabelEncoder
import logging

# Columns treated as plain numbers (no LabelEncoder)
NUMERIC_COLS = {'area', 'hired_workers', 'planYear'}


class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.feature_modes = {}
        self.feature_cols = []
        self._crop_synonyms = self._load_crop_synonyms()

    # ------------------------------------------------------------------
    # Crop normalization — maps regional/localized names to canonical form
    # ------------------------------------------------------------------
    def _load_crop_synonyms(self):
        """
        Loads ONLY actual synonym overrides from the optional crop_synonyms.json.
        Identity entries ("Wheat": "Wheat") are silently ignored — the crop list
        itself is derived from training data, not from this file.
        If the file doesn't exist, crop normalization simply passes values through.
        """
        try:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, 'config', 'crop_synonyms.json')
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # Keep only entries where key != value (real synonyms/translations)
            overrides = {k.strip(): v.strip() for k, v in raw.items() if k.strip() != v.strip()}
            if overrides:
                logging.info(f"Crop synonyms loaded: {len(overrides)} override(s) — "
                             f"{list(overrides.items())[:5]}")
            return overrides
        except FileNotFoundError:
            return {}  # file is optional
        except Exception as e:
            logging.warning(f"Could not load crop_synonyms.json: {e}")
            return {}

    def _normalize_crop(self, crop_val):
        """Return canonical crop name; falls back to original if not found."""
        val = str(crop_val).strip()
        return self._crop_synonyms.get(val, val)

    # ------------------------------------------------------------------
    def fit_transform(self, df, feature_columns):
        """Fit encoders and transform features, including interaction terms."""
        df_encoded = df.copy()

        # Normalize crop names
        if 'crop_name' in df_encoded.columns:
            df_encoded['crop_name'] = df_encoded['crop_name'].apply(
                lambda x: self._normalize_crop(x) if pd.notna(x) else 'Unknown'
            )

        # Numeric columns: coerce to float
        for num_col in NUMERIC_COLS:
            if num_col in df_encoded.columns:
                # hired_workers can be stored as True/False in CSV
                df_encoded[num_col] = df_encoded[num_col].apply(
                    lambda x: 1.0 if x is True or x == 'True'
                    else (0.0 if x is False or x == 'False'
                    else pd.to_numeric(x, errors='coerce'))
                ).fillna(0)

        # Interaction feature: crop_country captures regional crop behavior
        if 'crop_name' in df_encoded.columns and 'country_code' in df_encoded.columns:
            df_encoded['crop_country'] = (
                df_encoded['crop_name'].astype(str) + "_" +
                df_encoded['country_code'].astype(str)
            )
            if 'crop_country' not in feature_columns:
                feature_columns.append('crop_country')

        # Interaction feature: crop_partner — program-specific crop practices differ widely
        if 'crop_name' in df_encoded.columns and 'partner' in df_encoded.columns:
            df_encoded['crop_partner'] = (
                df_encoded['crop_name'].astype(str) + "_" +
                df_encoded['partner'].astype(str)
            )
            if 'crop_partner' not in feature_columns:
                feature_columns.append('crop_partner')

        # Interaction feature: country_partner — same partner behaves differently per country
        if 'country_code' in df_encoded.columns and 'partner' in df_encoded.columns:
            df_encoded['country_partner'] = (
                df_encoded['country_code'].astype(str) + "_" +
                df_encoded['partner'].astype(str)
            )
            if 'country_partner' not in feature_columns:
                feature_columns.append('country_partner')

        for col in feature_columns:
            if col in NUMERIC_COLS or col not in df_encoded.columns:
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

        self.feature_cols = feature_columns
        return df_encoded

    def transform(self, input_data):
        """Transform a single input dict for prediction."""
        # Normalize crop name at inference time
        if 'crop_name' in input_data:
            input_data['crop_name'] = self._normalize_crop(input_data.get('crop_name', ''))

        # Build interaction terms (must mirror fit_transform)
        if 'crop_name' in input_data and 'country_code' in input_data:
            input_data['crop_country'] = (
                str(input_data['crop_name']) + "_" + str(input_data['country_code'])
            )
        if 'crop_name' in input_data and 'partner' in input_data:
            input_data['crop_partner'] = (
                str(input_data['crop_name']) + "_" + str(input_data.get('partner', 'Unknown'))
            )
        if 'country_code' in input_data and 'partner' in input_data:
            input_data['country_partner'] = (
                str(input_data['country_code']) + "_" + str(input_data.get('partner', 'Unknown'))
            )

        encoded_vector = []
        for col in self.feature_cols:
            if col in NUMERIC_COLS:
                val = input_data.get(col, 0)
                if val is True or val == 'True':
                    val = 1.0
                elif val is False or val == 'False':
                    val = 0.0
                try:
                    encoded_vector.append(float(val))
                except (TypeError, ValueError):
                    encoded_vector.append(0.0)
                continue

            if col in self.label_encoders:
                le = self.label_encoders[col]
                val = str(input_data.get(col, 'Unknown'))
                try:
                    enc_val = le.transform([val])[0]
                except ValueError:
                    # Unseen label: use training mode as fallback
                    fallback = self.feature_modes.get(col, le.classes_[0])
                    enc_val = le.transform([str(fallback)])[0]
                    logging.debug(f"Unknown value '{val}' for '{col}', using fallback '{fallback}'")
                encoded_vector.append(int(enc_val))
            else:
                val = input_data.get(col, 0)
                if isinstance(val, bool):
                    val = int(val)
                try:
                    encoded_vector.append(float(val))
                except (TypeError, ValueError):
                    encoded_vector.append(0.0)

        return encoded_vector

    def get_feature_names(self):
        return [
            col if col in NUMERIC_COLS else (col + '_encoded' if col in self.label_encoders else col)
            for col in self.feature_cols
        ]
