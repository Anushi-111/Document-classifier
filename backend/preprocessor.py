import re
import pandas as pd

def preprocess_report(text: str) -> pd.DataFrame:
    text = " ".join(text.split())
    text = re.sub(r"Page \d+ of \d+.*?\d{2}:\d{2} [AP]M", "", text)
    text = re.sub(r"Patient Name.*?Report Status : Final Report", "", text)

    pattern = r"([A-Za-z\s\(\)\/]+)\s([\d\.]+)\s?([a-zA-Z%\/µ\^\d]+)?\s([\d\.\-<> ]+)?"
    matches = re.findall(pattern, text)

    rows = []
    for m in matches:
        test_name = m[0].strip()
        value = m[1].strip()
        unit = m[2].strip() if m[2] else ""
        ref_range = m[3].strip() if m[3] else ""
        rows.append([test_name, value, unit, ref_range])

    df = pd.DataFrame(rows, columns=["Test", "Value", "Unit", "Reference Range"])
    df = df[df["Test"] != ""].drop_duplicates()
    return df
