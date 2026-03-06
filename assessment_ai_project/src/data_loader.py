import pandas as pd
import json
import logging
from .config import Config

# Maps soaId prefix → human-readable section name.
# This is a domain constant — it doesn't belong in a config file.
_SECTION_MAP = {
    "AS": "Animal Stewardship",
    "BH": "Biodiversity & Habitat",
    "BP": "Biodiversity Protection",
    "CC": "Carbon Credits",
    "CE": "Carbon Emissions",
    "CF": "Conservation Finance",
    "CM": "Crop Management",
    "CO": "Conservation",
    "CT": "Conservation Tillage",
    "FM": "Farm Management",
    "HA": "Habitat",
    "HH": "Habitat & Hydrology",
    "HP": "Habitat Protection",
    "HS": "Habitat Stewardship",
    "HW": "Habitat Water",
    "IO": "Integrated Organisms",
    "LM": "Land Management",
    "NM": "Nutrient Management",
    "OE": "Organic - Energy",
    "OF": "Organic Farming",
    "OM": "Organic Matter",
    "OP": "Organic Production",
    "OR": "Organic Resources",
    "OS": "Organic Stewardship",
    "OT": "Organic Tillage",
    "OW": "Organic Water",
    "PM": "Pest Management",
    "PS": "Pest Stewardship",
    "RI": "Risk Indicators",
    "RM": "Risk Management",
    "SC": "Soil Carbon",
    "SF": "Soil Fertility",
    "SM": "Soil Management",
    "SR": "Soil Resilience",
    "ST": "Seed Treatment",
    "SW": "Soil Water",
    "WI": "Water Irrigation",
    "WP": "Water Protection",
    "WQ": "Water Quality",
    "WS": "Water Stewardship",
}

# Answers that point to a numeric entry, regardless of the soaId
_NUMERIC_KEYWORDS = {"ha", "acres", "kg", "liters", "tonnes", "count", "number", "amount"}


def _section_for(soa_id):
    prefix = ''.join(c for c in soa_id if c.isalpha()).upper()
    return _SECTION_MAP.get(prefix, "General")


def _infer_type(unique_answers):
    """
    Infer question type and valid options purely from observed answer values.
    Returns (type_str, options_list).
    """
    clean = {str(a).strip() for a in unique_answers
             if a is not None and str(a).strip() not in ('', 'null', 'None')}

    if not clean:
        return "text", []

    # True/False boolean pattern (check before numeric so "1"/"0" maps to radio)
    if clean <= {"True", "False", "true", "false", "1", "0"}:
        return "radio", ["False", "True"]

    # All values are numeric (percentages, counts, amounts, etc.)
    if all(v.replace('.', '', 1).replace('-', '', 1).isdigit() for v in clean):
        # Few distinct numeric values → classifiable (e.g. percentage tiers: 0, 25, 50, 75, 100)
        # Threshold 75: allows nutrient-rate fields like Nitrogen (~61 distinct values) to be classified
        if len(clean) <= 75:
            return "radio", sorted(clean, key=lambda x: float(x))
        # Truly continuous / free-form number — too many unique values to classify
        return "number", []

    # Up to 50 distinct text values → classifiable (radio or checkbox)
    if len(clean) <= 50:
        # Long option strings or many options suggest checkbox/multi-select
        avg_len = sum(len(v) for v in clean) / len(clean)
        if len(clean) > 6 or avg_len > 40:
            return "checkbox", sorted(clean)
        return "radio", sorted(clean)

    # Many unique values → free-text
    return "text", []


class DataLoader:
    def __init__(self):
        self.feature_columns = Config.FEATURE_COLUMNS
        self.question_meta = {}  # populated by load_data()

    def load_data(self, csv_file):
        """
        Load assessment data and parse nested JSON labelsAnswersMap.
        Handles column name variations and missing optional columns gracefully.
        """
        try:
            logging.info(f"Loading data from {csv_file}")
            raw_df = pd.read_csv(csv_file, encoding='utf-8')
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

        # Normalize column names: strip whitespace, lowercase
        raw_df.columns = [c.strip() for c in raw_df.columns]

        # Map any alternate column names to canonical names
        col_aliases = {
            'Partner': 'partner',
            'Crop': 'crop_name',
            'Country': 'country_code',
            'PlanYear': 'planYear',
            'plan_year': 'planYear',
        }
        raw_df.rename(columns=col_aliases, inplace=True)

        # Determine which feature columns actually exist in this CSV
        available_features = [c for c in self.feature_columns if c in raw_df.columns]
        missing = [c for c in self.feature_columns if c not in raw_df.columns]
        if missing:
            logging.warning(f"Feature columns not found in CSV (will be set to 0/Unknown): {missing}")
            for m in missing:
                raw_df[m] = None
            available_features = self.feature_columns

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

                # Strip leading spaces from soaId (data quality issue found in real CSV)
                if indicator:
                    indicator = indicator.strip()

                # Skip records where both soaId and answer are missing
                if not indicator:
                    continue

                rec = {col: row.get(col) for col in available_features}
                rec['indicator'] = indicator
                rec['answer'] = answer
                records.append(rec)

        if not records:
            raise ValueError("No valid records found in data file")

        long_df = pd.DataFrame(records)
        wide_df = long_df.pivot_table(
            index=available_features,
            columns='indicator',
            values='answer',
            aggfunc=lambda x: x.iloc[0] if len(x) else None
        ).reset_index()

        wide_df.columns.name = None
        logging.info(f"Data loaded. Shape: {wide_df.shape} | Indicators: {wide_df.shape[1] - len(available_features)}")

        # Build question metadata dynamically from what was just parsed
        self.question_meta = self.build_question_meta(raw_df)
        logging.info(f"Question metadata built dynamically: {len(self.question_meta)} indicators")

        return wide_df

    def build_question_meta(self, raw_df):
        """
        Derives question metadata (type, options, section, labelName) entirely
        from the labelsAnswersMap column of the raw CSV.
        No static files, no hardcoded questions — data is the single source of truth.
        """
        # Collect per-soaId: all observed answers + first seen labelName
        soa_answers = {}   # soaId -> set of answer strings
        soa_labels  = {}   # soaId -> labelName string

        for cell in raw_df.get('labelsAnswersMap', []):
            try:
                items = json.loads(cell) if isinstance(cell, str) else []
            except Exception:
                continue

            for item in items:
                soa_id = item.get('soaId', '')
                if not soa_id:
                    continue
                soa_id = soa_id.strip()

                answer    = item.get('answer')
                label_name = item.get('labelName', '')

                if soa_id not in soa_answers:
                    soa_answers[soa_id] = set()
                    soa_labels[soa_id]  = ''

                if answer is not None and str(answer).strip() not in ('', 'null', 'None'):
                    soa_answers[soa_id].add(str(answer).strip())

                if label_name and not soa_labels[soa_id]:
                    soa_labels[soa_id] = str(label_name).strip()

        meta = {}
        for soa_id in sorted(soa_answers):
            q_type, options = _infer_type(soa_answers[soa_id])
            entry = {
                "type":       q_type,
                "section":    _section_for(soa_id),
                "labelName":  soa_labels.get(soa_id, ''),
                "conditional": False,
            }
            if options:
                entry["options"] = options
            if q_type == "checkbox":
                entry["multi"] = True
            meta[soa_id] = entry

        return meta
