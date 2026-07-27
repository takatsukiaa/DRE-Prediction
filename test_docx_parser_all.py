import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from docx import Document

import docx_parser_all as parser


def add_rows(table, rows):
    for row_index, values in enumerate(rows):
        if row_index >= len(table.rows):
            table.add_row()
        row = table.rows[row_index]
        for column_index, value in enumerate(values):
            if column_index >= len(row.cells):
                break
            row.cells[column_index].text = str(value)


def build_complete_docx(path: Path, gender="1") -> None:
    doc = Document()

    pregnancy = doc.add_table(rows=1, cols=2)
    add_rows(pregnancy, [["Pregenacy", "Pregnant"]])

    baseline = doc.add_table(rows=1, cols=2)
    add_rows(
        baseline,
        [
            ["Baseline 基本資料", ""],
            ["Patient ID", "00123"],
            ["Gender", gender],
            ["Birthday", "1990-01-02"],
        ],
    )

    semiology = doc.add_table(rows=1, cols=2)
    add_rows(
        semiology,
        [
            ["Semiology", ""],
            [
                "Focal (F) Focal without generalized 1.1 (preserved), "
                "1.2 (impaired); Focal with generalized 1.3",
                "1.2 or 1.3 (clinical comment)",
            ],
            ["Focality", "9.9 from an unrelated legacy row"],
        ],
    )

    eeg_focality = doc.add_table(rows=1, cols=2)
    add_rows(
        eeg_focality,
        [
            ["EEG focality", ""],
            ["Temporal", "T3 T5"],
        ],
    )

    adverse = doc.add_table(rows=1, cols=2)
    add_rows(
        adverse,
        [
            ["Adverse for ASMs", "Yes"],
            ["ASM name", "Lamictal"],
            ["Description", "Dizziness"],
        ],
    )

    asm = doc.add_table(rows=1, cols=4)
    add_rows(
        asm,
        [
            ["ASM狀況", "", "", ""],
            ["原因 SOAP", "2020/01/15 seizure 2 times; 2020/03/01 no seizure", "", ""],
            ["Date", "Lamictal (Lamotrigine) 50mg/tab", "Keppra (Levetiracetam) 500mg/tab", "備註"],
            ["2020-01-15", "2 tab/BIDPC", "1 tab/PRN, 0.5 tab/UNKNOWN", ""],
            ["2020-03-10", "1.5 tab/HS", "", ""],
        ],
    )

    eeg = doc.add_table(rows=1, cols=2)
    add_rows(
        eeg,
        [
            ["EEG", ""],
            ["Date", "Report"],
            ["20200120", "Normal awake EEG."],
            ["2020-02-20", "Left temporal spikes."],
        ],
    )

    lab = doc.add_table(rows=1, cols=3)
    add_rows(
        lab,
        [
            ["Lab", "", ""],
            [parser.GENERAL_REPORT, "", ""],
            ["Date", "AST", "ALT"],
            ["2020-01-03", "10", "20"],
            [parser.ROUTINE_REPORT, "", ""],
            ["Date", "WBC", "Hb"],
            ["2020-01-04", "5.5", "13"],
            [parser.DRUG_REPORT, "", ""],
            ["Date", "VPA", ""],
            ["2020-01-05", "75", ""],
        ],
    )

    cognitive = doc.add_table(rows=1, cols=13)
    add_rows(
        cognitive,
        [
            ["Mental Retardation"] + [""] * 12,
            ["Date", "MMSE◉"] + [f"Q{i}" for i in range(1, 12)],
            ["2020-01-06", "28"] + [""] * 11,
        ],
    )

    brain = doc.add_table(rows=1, cols=2)
    add_rows(
        brain,
        [
            ["Brain Image", ""],
            ["MRI（核磁共振）", ""],
            ["Date", ""],
            ["2020-01-07", "Normal MRI study"],
            ["CT（電腦斷層）", ""],
            ["Date", ""],
            ["2020-01-08", "Normal CT study"],
            ["SPECT（核醫科影像）", ""],
            ["Date", ""],
            ["2020-01-09", "Normal SPECT study"],
        ],
    )
    doc.save(path)


PREGNANCY_ROWS = [
    ("正在懷孕", "0"),
    ("Abnormal at OBS routine check 產檢異常", "0"),
    ("Pregnancy number 先前+目前", "2"),
    ("Child birth number先前+目前", "1"),
    ("Inborn error of child 先天性疾病", "0"),
    ("Developmental delay of child", "1 (Spontaneous abortion 2016.12.12)"),
    ("Spontaneous Abortion", "1"),
    ("Artificial Abortion", "0"),
    ("Seizure frequency during pregnancy (T1, T2, T3 n/m)", "3"),
    ("Increased seizure frequency during pregnancy", "1"),
    ("Decreased seizure frequency during pregnancy", "0"),
]


def build_static_docx(
    path: Path,
    *,
    gender="0",
    first_seizure_age="11~12 (小兒神經)",
    systemic_disease="HTN",
    etiology="5, 6, 9(MTS, 頭部外傷, SAH)",
    pregnancy_rows=PREGNANCY_ROWS,
) -> None:
    doc = Document()
    baseline = doc.add_table(rows=1, cols=2)
    add_rows(
        baseline,
        [
            ["Baseline 基本資料", ""],
            ["Patient ID", "00456"],
            ["Gender", gender],
            ["1st Seizure Age", first_seizure_age],
            ["Systemic disease 內科疾病", systemic_disease],
        ],
    )
    etiology_table = doc.add_table(rows=1, cols=4)
    add_rows(etiology_table, [["Etiology of epilepsy", etiology, "Mental Retardation", "0"]])
    if pregnancy_rows is not None:
        pregnancy = doc.add_table(rows=1, cols=2)
        add_rows(pregnancy, [["Pregnancy 懷孕狀況", ""]] + list(pregnancy_rows))
    doc.save(path)


def asm_row(**overrides):
    row = {column: pd.NA for column in parser.ASM_COLUMNS}
    row.update(
        {
            "patient_id": "0001",
            "cohort": "DRE",
            "date": pd.Timestamp("2020-01-31"),
            "observed_date": pd.Timestamp("2020-01-15"),
            "record_type": "prescription",
            "drug_label": "Lamictal (Lamotrigine) 50mg/tab",
            "generic_name": "Lamotrigine",
            "unit_dose": 50.0,
            "unit_dose_unit": "mg/tab",
            "administered_dose": 0.5,
            "administration_unit": "tab",
            "frequency": "HS",
            "frequency_per_day": 1.0,
            "daily_dose_mg": 25.0,
            "ddd": 25.0 / 300.0,
            "raw_instruction": "0.5 tab/HS",
            "is_carried_forward": False,
        }
    )
    row.update(overrides)
    if row["record_type"] != "prescription" and "is_carried_forward" not in overrides:
        row["is_carried_forward"] = pd.NA
    return row


class ConsolidatedParserTests(unittest.TestCase):
    def test_complete_document_and_asm_detail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Pt123_DRE.docx"
            build_complete_docx(path)
            result = parser.parse_docx_files(path)

        self.assertEqual(set(result), set(parser.DATAFRAME_COLUMNS))
        for name, frame in result.items():
            if name != "parse_errors":
                self.assertNotIn("source_file", frame.columns)
        self.assertIn("source_file", result["parse_errors"].columns)

        static = result["static"].iloc[0]
        self.assertEqual(static["patient_id"], "00123")
        self.assertEqual(static["gender"], "male")
        self.assertTrue(bool(static["adverse_for_asms"]))
        self.assertEqual(static["adverse_for_asms_asm_name"], "Lamictal")
        self.assertEqual(static["adverse_for_asms_description"], "Dizziness")
        self.assertTrue(pd.isna(static["pregnancy"]))
        self.assertEqual(static["eeg_focality"], "T3, T5")
        self.assertEqual(static["focality"], "1.2, 1.3")

        asm = result["asm"]
        jan = asm[
            (asm["record_type"] == "prescription")
            & (asm["generic_name"] == "Lamotrigine")
            & (asm["date"] == pd.Timestamp("2020-01-31"))
        ].iloc[0]
        self.assertEqual(float(jan["unit_dose"]), 50.0)
        self.assertEqual(jan["unit_dose_unit"], "mg/tab")
        self.assertEqual(float(jan["administered_dose"]), 2.0)
        self.assertEqual(jan["administration_unit"], "tab")
        self.assertEqual(jan["frequency"], "BIDPC")
        self.assertEqual(float(jan["frequency_per_day"]), 2.0)
        self.assertAlmostEqual(float(jan["daily_dose_mg"]), 200.0)
        self.assertAlmostEqual(float(jan["ddd"]), 200 / 300)
        self.assertFalse(bool(jan["is_carried_forward"]))

        feb = asm[
            (asm["record_type"] == "prescription")
            & (asm["generic_name"] == "Lamotrigine")
            & (asm["date"] == pd.Timestamp("2020-02-29"))
        ].iloc[0]
        self.assertEqual(feb["observed_date"], pd.Timestamp("2020-01-15"))
        self.assertTrue(bool(feb["is_carried_forward"]))
        feb_summary = asm[
            (asm["record_type"] == "monthly_summary")
            & (asm["date"] == pd.Timestamp("2020-02-29"))
        ].iloc[0]
        self.assertTrue(bool(feb_summary["is_carried_forward"]))

        prn = asm[(asm["generic_name"] == "Levetiracetam") & (asm["record_type"] == "prescription")].iloc[0]
        self.assertEqual(prn["frequency"], "PRN")
        self.assertTrue(pd.isna(prn["frequency_per_day"]))
        self.assertTrue(pd.isna(prn["ddd"]))
        unknown = asm[
            (asm["generic_name"] == "Levetiracetam")
            & (asm["frequency"] == "UNKNOWN")
        ].iloc[0]
        self.assertEqual(float(unknown["administered_dose"]), 0.5)
        self.assertTrue(pd.isna(unknown["frequency_per_day"]))
        march = asm[
            (asm["record_type"] == "prescription")
            & (asm["generic_name"] == "Lamotrigine")
            & (asm["date"] == pd.Timestamp("2020-03-31"))
        ].iloc[0]
        self.assertEqual(float(march["administered_dose"]), 1.5)
        self.assertEqual(len(asm[asm["record_type"] == "monthly_summary"]), 3)

        self.assertEqual(len(result["eeg"]), 2)
        self.assertIn("Normal awake EEG.", result["eeg"]["report"].tolist())
        self.assertEqual(result["seizure"].iloc[0]["seizure_count_estimate"], 2)
        self.assertEqual(set(result["lab_general"]["test_name"]), {"AST", "ALT"})
        self.assertEqual(set(result["lab_routine"]["test_name"]), {"WBC", "Hb"})
        self.assertEqual(set(result["lab_drug"]["test_name"]), {"VPA"})
        self.assertEqual(set(result["cognitive"]["metric"]), set(parser.TEST_COLUMNS["MMSE"]))
        mmse_questions = result["cognitive"][result["cognitive"]["metric"].str.startswith("MMSE_Q")]
        self.assertEqual(len(mmse_questions), 11)
        self.assertTrue(mmse_questions["value"].isna().all())
        self.assertEqual(set(result["brain_image"]["field_name"]), {"MRI", "CT", "SPECT"})
        self.assertNotIn("record_number", result["brain_image"].columns)

    def test_brain_image_sections_skip_notes_and_report_review_errors(self):
        doc = Document()
        brain = doc.add_table(rows=1, cols=2)
        add_rows(
            brain,
            [
                ["Brain Image", ""],
                ["2024-01-01", "Report before a category"],
                ["MRI", ""],
                ["Date", ""],
                ["無", ""],
                ["住院期間有檢驗報告 20240102", ""],
                ["2024-01-03", "MRI report"],
                ["2024-01-03", "MRI report"],
                ["not-a-date", "MRI report with invalid date"],
                ["CT (電腦斷層)", ""],
                ["Date", ""],
                ["2024-01-04", "CT report"],
                ["SPECT", ""],
                ["Date", ""],
                ["2024-01-05", "SPECT report"],
            ],
        )
        errors = []
        rows = parser.parse_brain_image(
            doc, Path("Pt001.docx"), "0001", "DRE", errors
        )

        self.assertEqual(
            [row["field_name"] for row in rows],
            ["MRI", "MRI", "CT", "SPECT"],
        )
        self.assertEqual([row["value"] for row in rows[:2]], ["MRI report", "MRI report"])
        self.assertTrue(all("record_number" not in row for row in rows))
        self.assertEqual(
            {error["error_type"] for error in errors},
            {"missing_category", "unparseable_date"},
        )

    def test_lab_threshold_values_are_preserved_as_numeric_boundaries(self):
        threshold_cases = {
            ">=90": 90.0,
            "> 89.5": 89.5,
            "≥ 90": 90.0,
            "≧90": 90.0,
            "≧90 (107)": 90.0,
            "<=60": 60.0,
            "≤ 60.5": 60.5,
            "≦60": 60.0,
        }
        for raw, expected in threshold_cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(parser.clean_numeric_value(raw), expected)
        self.assertTrue(pd.isna(parser.clean_numeric_value("/")))
        self.assertTrue(pd.isna(parser.clean_numeric_value("Non-reactive")))

        doc = Document()
        lab = doc.add_table(rows=1, cols=3)
        add_rows(
            lab,
            [
                ["Lab", "", ""],
                [parser.ROUTINE_REPORT, "", ""],
                ["Date", "eGFR", "eGFR (CKD-EPI)"],
                ["2025-01-15", "≧90", "≧90 (107)"],
            ],
        )
        errors = []
        result = parser.parse_labs(doc, Path("threshold.docx"), "0001", "DRE", errors)
        routine = (
            pd.DataFrame(result["lab_routine"])
            .sort_values("test_name")
            .reset_index(drop=True)
        )

        self.assertEqual(errors, [])
        self.assertEqual(routine["test_name"].tolist(), ["eGFR", "eGFR (CKD-EPI)"])
        self.assertEqual(routine["value"].tolist(), [90.0, 90.0])
        self.assertTrue((routine["date"] == pd.Timestamp("2025-01-31")).all())

    def test_pregnancy_retained_for_non_male(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Pt456_NDRE.docx"
            build_complete_docx(path, gender="0")
            result = parser.parse_docx_files(path)
        self.assertEqual(result["static"].iloc[0]["gender"], "female")
        self.assertEqual(result["static"].iloc[0]["pregnancy"], "Pregnant")

    def test_static_normalizers(self):
        age_cases = {
            "11~12 (note)": "11~12",
            "16-17左右": "16-17",
            "10+": "10+",
            "0.5 y/o": "0.5 y/o",
            "3 months": "3 months",
            "4y9m": "4y9m",
            "early childhood": "early childhood",
            "國中": "國中",
            "12 syncope, 18GTC": "12",
        }
        for raw, expected in age_cases.items():
            self.assertEqual(parser.clean_first_seizure_age(raw), expected)
        self.assertTrue(pd.isna(parser.clean_first_seizure_age("未知 (old chart)")))
        self.assertTrue(pd.isna(parser.clean_first_seizure_age("2005")))

        self.assertEqual(parser.normalize_gender("0"), "female")
        self.assertEqual(parser.normalize_gender("female"), "female")
        self.assertEqual(parser.normalize_gender("1"), "male")
        self.assertEqual(parser.normalize_gender("男"), "male")
        self.assertTrue(pd.isna(parser.normalize_gender("2")))

        self.assertEqual(
            parser.clean_etiology_codes("5, 6, 9(MTS, 頭部外傷, SAH)"),
            "5, 6, 9",
        )
        self.assertEqual(parser.clean_etiology_codes("7、13"), "7, 13")
        self.assertTrue(pd.isna(parser.clean_etiology_codes("unknown")))

    def test_focality_code_normalizer(self):
        cases = {
            "1.2 or 1.3": ["1.2", "1.3"],
            "1.1 to 1.2, 1.1 to 1.3": ["1.1", "1.2", "1.3"],
            "1.2->1.3": ["1.2", "1.3"],
            "1.3or2.3(?)": ["1.3", "2.3"],
            "1.2, 1.3【semiology: convulsion for 1-2 mins】": ["1.2", "1.3"],
            "1.3 or 2.3 -> 1.2, 1.3": ["1.3", "2.3", "1.2"],
            "1.2 3.3": ["1.2", "3.3"],
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(parser.extract_focality_codes(raw), expected)
        self.assertEqual(parser.extract_focality_codes("clinical comment only"), [])

    def test_focality_uses_only_the_semiology_focal_row(self):
        labels = [
            "Focal (F) Focal without generalized 1.1; Focal with generalized 1.3",
            "Focal (F)Focal without generalized 1.1; Focal with generalized 1.3",
        ]
        for label in labels:
            with self.subTest(label=label):
                doc = Document()
                baseline = doc.add_table(rows=1, cols=2)
                add_rows(
                    baseline,
                    [
                        ["Baseline 基本資料", ""],
                        ["Patient ID", "001"],
                        ["Gender", "0"],
                        ["Focality", "3.3 must be ignored"],
                        ["EEG focality", "T3 T5"],
                    ],
                )
                semiology = doc.add_table(rows=1, cols=2)
                add_rows(
                    semiology,
                    [
                        ["Semiology", ""],
                        ["Focality", "2.2 must also be ignored"],
                        [label, "1.2 or 1.3 (comment)"],
                    ],
                )
                errors = []
                row = parser.parse_static(
                    doc,
                    Path("Pt001_DRE.docx"),
                    "001",
                    "DRE",
                    errors,
                )
                self.assertEqual(row["focality"], "1.2, 1.3")
                self.assertEqual(row["eeg_focality"], "T3, T5")
                self.assertFalse(
                    any(error["error_type"] == "unparseable_focality" for error in errors)
                )

    def test_focality_empty_and_unparseable_rows(self):
        for raw_value, expects_error in [("", False), ("clinical comment only", True)]:
            with self.subTest(raw_value=raw_value):
                doc = Document()
                baseline = doc.add_table(rows=1, cols=2)
                add_rows(
                    baseline,
                    [["Baseline", ""], ["Patient ID", "001"], ["Gender", "0"]],
                )
                semiology = doc.add_table(rows=1, cols=2)
                add_rows(
                    semiology,
                    [
                        ["Semiology", ""],
                        [
                            "Focal (F) Focal without generalized; Focal with generalized",
                            raw_value,
                        ],
                    ],
                )
                errors = []
                row = parser.parse_static(
                    doc,
                    Path("Pt001_DRE.docx"),
                    "001",
                    "DRE",
                    errors,
                )
                self.assertTrue(pd.isna(row["focality"]))
                focality_errors = [
                    error for error in errors
                    if error["error_type"] == "unparseable_focality"
                ]
                self.assertEqual(bool(focality_errors), expects_error)
                if expects_error:
                    self.assertEqual(focality_errors[0]["evidence"], raw_value)

    def test_bilingual_static_and_pregnancy_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Pt456_DRE.docx"
            build_static_docx(path)
            result = parser.parse_docx_files(path)

        static = result["static"].iloc[0]
        self.assertEqual(static["gender"], "female")
        self.assertEqual(static["first_seizure_age"], "11~12")
        self.assertEqual(static["systemic_disease"], "HTN")
        self.assertEqual(static["etiology"], "5, 6, 9")
        expected = {
            "pregnancy": "0",
            "pregnancy_abnormal_obstetric_check": "0",
            "pregnancy_count": "2",
            "childbirth_count": "1",
            "child_inborn_error": "0",
            "child_developmental_delay": "1 (Spontaneous abortion 2016.12.12)",
            "spontaneous_abortion": "1",
            "artificial_abortion": "0",
            "seizure_frequency_during_pregnancy": "3",
            "increased_seizure_frequency_during_pregnancy": "1",
            "decreased_seizure_frequency_during_pregnancy": "0",
        }
        for field, value in expected.items():
            self.assertEqual(static[field], value)

    def test_pregnancy_is_suppressed_for_male_and_over_50_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            male_path = root / "Pt101_DRE.docx"
            build_static_docx(male_path, gender="1")
            over_50_path = root / "Pt102_DRE.docx"
            over_50_rows = [
                (label, parser.PREGNANCY_NOT_APPLICABLE) for label, _value in PREGNANCY_ROWS
            ]
            build_static_docx(over_50_path, gender="0", pregnancy_rows=over_50_rows)
            result = parser.parse_docx_files(root)

        male = result["static"][result["static"]["gender"] == "male"].iloc[0]
        over_50 = result["static"][
            (result["static"]["gender"] == "female")
            & (result["static"]["patient_id"] == "00456")
        ].iloc[0]
        self.assertTrue(male[parser.PREGNANCY_COLUMNS].isna().all())
        self.assertTrue(over_50[parser.PREGNANCY_COLUMNS].isna().all())
        evidence = result["parse_errors"]["evidence"].fillna("")
        self.assertFalse(evidence.str.contains(parser.PREGNANCY_NOT_APPLICABLE, regex=False).any())

    def test_systemic_disease_preserves_zero_none_and_free_text(self):
        for systemic_disease in ["0", "無", "HTN and dyslipidemia"]:
            with self.subTest(systemic_disease=systemic_disease):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "Pt456_DRE.docx"
                    build_static_docx(
                        path,
                        systemic_disease=systemic_disease,
                        pregnancy_rows=None,
                    )
                    result = parser.parse_docx_files(path)
                self.assertEqual(
                    result["static"].iloc[0]["systemic_disease"],
                    systemic_disease,
                )

    def test_invalid_populated_static_values_create_review_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Pt456_DRE.docx"
            build_static_docx(
                path,
                gender="2",
                first_seizure_age="2005",
                etiology="uncoded",
                pregnancy_rows=None,
            )
            result = parser.parse_docx_files(path)

        static = result["static"].iloc[0]
        self.assertTrue(pd.isna(static["gender"]))
        self.assertTrue(pd.isna(static["first_seizure_age"]))
        self.assertTrue(pd.isna(static["etiology"]))
        static_errors = result["parse_errors"][
            result["parse_errors"]["category"] == "static"
        ]
        self.assertEqual(
            set(static_errors["error_type"]),
            {
                "unrecognized_gender",
                "unparseable_first_seizure_age",
                "unparseable_etiology",
            },
        )

    def test_ambiguous_adverse_description_is_nullable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Pt789.docx"
            doc = Document()
            baseline = doc.add_table(rows=1, cols=2)
            add_rows(baseline, [["Baseline 基本資料", ""], ["Patient ID", "789"], ["Gender", "2"]])
            adverse = doc.add_table(rows=1, cols=2)
            add_rows(
                adverse,
                [
                    ["Adverse of ASM", ""],
                    ["ASM name", "Lamictal"],
                    ["Adverse Response", "Rash"],
                ],
            )
            doc.save(path)
            result = parser.parse_docx_files(path)
        self.assertTrue(pd.isna(result["static"].iloc[0]["adverse_for_asms"]))
        self.assertEqual(result["static"].iloc[0]["adverse_for_asms_asm_name"], "Lamictal")
        self.assertEqual(result["static"].iloc[0]["adverse_for_asms_description"], "Rash")
        self.assertIn("ambiguous_adverse_flag", result["parse_errors"]["error_type"].tolist())

    def test_adverse_template_uses_checkbox_name_and_response_only(self):
        doc = Document()
        no_adverse = doc.add_table(rows=1, cols=2)
        add_rows(
            no_adverse,
            [
                ["Adverse of ASM", ""],
                ["Yes", "0"],
                ["ASM name", "stale drug text"],
                ["ASM dose", "stale dose text"],
                ["Adverse Response", "stale response text"],
            ],
        )
        flag, asm_name, description, ambiguous = parser.parse_adverse_table(no_adverse)
        self.assertFalse(flag)
        self.assertEqual(asm_name, "")
        self.assertEqual(description, "")
        self.assertFalse(ambiguous)

        has_adverse = doc.add_table(rows=1, cols=2)
        add_rows(
            has_adverse,
            [
                ["Adverse of ASM", ""],
                ["Yes", "1"],
                ["ASM name", "Fycompa"],
                ["ASM dose", "2 mg/HS"],
                ["Adverse Response", "Dizziness"],
            ],
        )
        flag, asm_name, description, ambiguous = parser.parse_adverse_table(has_adverse)
        self.assertTrue(flag)
        self.assertEqual(asm_name, "Fycompa")
        self.assertEqual(description, "Dizziness")
        self.assertFalse(ambiguous)

        has_multiple_adverse = doc.add_table(rows=1, cols=2)
        add_rows(
            has_multiple_adverse,
            [
                ["Adverse of ASM", ""],
                ["Yes", "3"],
                ["ASM name", "Zonisamide, Topiramate, Perampanel"],
                ["Adverse Response", "Three documented adverse responses"],
            ],
        )
        flag, asm_name, description, ambiguous = parser.parse_adverse_table(has_multiple_adverse)
        self.assertTrue(flag)
        self.assertEqual(asm_name, "Zonisamide, Topiramate, Perampanel")
        self.assertEqual(description, "Three documented adverse responses")
        self.assertFalse(ambiguous)

    def test_unit_dose_fallbacks_and_explicit_strength_priority(self):
        self.assertEqual(parser.unit_dose_from_label("Fycompa 2mg (Perampanel)"), (2.0, "mg"))
        self.assertEqual(parser.unit_dose_from_label("Valproate (Depakine)"), (500.0, "mg/tab"))
        self.assertEqual(parser.unit_dose_from_label("Lamictal 5mg/tab (Lamotrigine)"), (5.0, "mg/tab"))

    def test_mmse_q1_to_q11_values_are_parsed(self):
        doc = Document()
        table = doc.add_table(rows=1, cols=13)
        add_rows(
            table,
            [
                ["Mental Retardation"] + [""] * 12,
                ["Date", "MMSE◉"] + [f"Q {index}" for index in range(1, 12)],
                ["2024-01-02", "30"] + [str(index) for index in range(1, 12)],
            ],
        )
        errors = []
        rows = parser.parse_cognitive(doc, Path("Pt001.docx"), "001", "DRE", errors)
        self.assertEqual(errors, [])
        by_metric = {row["metric"]: row["value"] for row in rows}
        self.assertEqual(set(by_metric), set(parser.TEST_COLUMNS["MMSE"]))
        self.assertEqual(by_metric["MMSE"], "30")
        for index in range(1, 12):
            self.assertEqual(by_metric[f"MMSE_Q{index}"], str(index))

    def test_patient_specific_asm_frames_join_and_sum_instructions(self):
        rows = [
            asm_row(),
            asm_row(
                administered_dose=1.0,
                frequency="QD",
                daily_dose_mg=50.0,
                ddd=50.0 / 300.0,
                raw_instruction="1 tab/QD",
            ),
            asm_row(
                record_type="monthly_summary",
                drug_label=pd.NA,
                generic_name=pd.NA,
                unit_dose=pd.NA,
                unit_dose_unit=pd.NA,
                administered_dose=pd.NA,
                administration_unit=pd.NA,
                frequency=pd.NA,
                frequency_per_day=pd.NA,
                daily_dose_mg=pd.NA,
                ddd=pd.NA,
                total_ddd=0.25,
                number_of_drugs=1,
                raw_instruction=pd.NA,
            ),
            asm_row(
                date=pd.Timestamp("2020-02-29"),
                is_carried_forward=True,
            ),
            asm_row(
                date=pd.Timestamp("2020-02-29"),
                administered_dose=1.0,
                frequency="QD",
                daily_dose_mg=50.0,
                ddd=50.0 / 300.0,
                raw_instruction="1 tab/QD",
                is_carried_forward=True,
            ),
            asm_row(
                date=pd.Timestamp("2020-02-29"),
                record_type="monthly_summary",
                drug_label=pd.NA,
                generic_name=pd.NA,
                unit_dose=pd.NA,
                unit_dose_unit=pd.NA,
                administered_dose=pd.NA,
                administration_unit=pd.NA,
                frequency=pd.NA,
                frequency_per_day=pd.NA,
                daily_dose_mg=pd.NA,
                ddd=pd.NA,
                total_ddd=0.25,
                number_of_drugs=1,
                raw_instruction=pd.NA,
            ),
            asm_row(
                patient_id="0002",
                cohort="NDRE",
                drug_label="Keppra (Levetiracetam) 500mg/tab",
                generic_name="Levetiracetam",
                unit_dose=500.0,
                administered_dose=1.0,
                frequency="BID",
                frequency_per_day=2.0,
                daily_dose_mg=1000.0,
                ddd=2.0,
                raw_instruction="1 tab/BID",
            ),
            asm_row(
                patient_id="0002",
                cohort="NDRE",
                record_type="monthly_summary",
                drug_label=pd.NA,
                generic_name=pd.NA,
                unit_dose=pd.NA,
                unit_dose_unit=pd.NA,
                administered_dose=pd.NA,
                administration_unit=pd.NA,
                frequency=pd.NA,
                frequency_per_day=pd.NA,
                daily_dose_mg=pd.NA,
                ddd=pd.NA,
                total_ddd=2.0,
                number_of_drugs=1,
                raw_instruction=pd.NA,
            ),
        ]
        asm = parser._build_dataframe("asm", rows)
        frames = parser.build_patient_asm_dataframes(asm)

        self.assertEqual(set(frames), {"0001_DRE", "0002_NDRE"})
        patient = frames["0001_DRE"]
        self.assertTrue(all(patient["patient_id"] == "0001"))
        self.assertIn("lamotrigine__administered_dose", patient.columns)
        self.assertNotIn("levetiracetam__administered_dose", patient.columns)
        january = patient.iloc[0]
        self.assertEqual(january["lamotrigine__administered_dose"], "0.5; 1")
        self.assertEqual(january["lamotrigine__frequency"], "HS; QD")
        self.assertEqual(january["lamotrigine__raw_instruction"], "0.5 tab/HS; 1 tab/QD")
        self.assertAlmostEqual(float(january["lamotrigine__daily_dose_mg"]), 75.0)
        self.assertAlmostEqual(float(january["lamotrigine__ddd"]), 0.25)
        self.assertAlmostEqual(float(january["total_ddd"]), 0.25)
        self.assertEqual(int(january["number_of_drugs"]), 1)
        self.assertFalse(bool(january["is_carried_forward"]))
        february = patient.iloc[1]
        self.assertEqual(february["lamotrigine__observed_date"], pd.Timestamp("2020-01-15"))
        self.assertTrue(bool(february["is_carried_forward"]))
        self.assertFalse(any(column.endswith("__is_carried_forward") for column in patient.columns))
        self.assertNotIn("source_file", patient.columns)

    def test_patient_specific_asm_export_manifest_empty_patient_and_errors(self):
        rows = [
            asm_row(),
            asm_row(
                record_type="monthly_summary",
                drug_label=pd.NA,
                generic_name=pd.NA,
                unit_dose=pd.NA,
                unit_dose_unit=pd.NA,
                administered_dose=pd.NA,
                administration_unit=pd.NA,
                frequency=pd.NA,
                frequency_per_day=pd.NA,
                daily_dose_mg=pd.NA,
                ddd=pd.NA,
                total_ddd=25.0 / 300.0,
                number_of_drugs=1,
                raw_instruction=pd.NA,
            ),
        ]
        dataframes = {
            "asm": parser._build_dataframe("asm", rows),
            "static": pd.DataFrame(
                [
                    {"patient_id": "0001", "cohort": "DRE"},
                    {"patient_id": "0003", "cohort": "DRE"},
                ]
            ),
            "parse_errors": parser._build_dataframe(
                "parse_errors",
                [
                    {
                        "patient_id": "0001",
                        "cohort": "DRE",
                        "source_file": "Pt0001.docx",
                        "category": "asm",
                        "error_type": "unknown_frequency",
                        "row": "1 tab/XYZ",
                        "message": "Unknown frequency",
                    },
                    {
                        "patient_id": "0001",
                        "cohort": "DRE",
                        "source_file": "Pt0001.docx",
                        "category": "eeg",
                        "error_type": "test_error",
                        "message": "Not an ASM error",
                    },
                ],
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            manifest = parser.export_asm_per_patient(dataframes, output)
            self.assertTrue((output / "0001_DRE_asm.csv").exists())
            self.assertTrue((output / "0003_DRE_asm.csv").exists())
            self.assertTrue((output / "manifest.csv").exists())
            self.assertTrue((output / "asm_parse_errors.csv").exists())
            empty = pd.read_csv(output / "0003_DRE_asm.csv", dtype={"patient_id": "string"})
            errors = pd.read_csv(output / "asm_parse_errors.csv", dtype={"patient_id": "string"})

        self.assertEqual(manifest["patient_id"].tolist(), ["0001", "0003"])
        no_asm = manifest[manifest["patient_id"] == "0003"].iloc[0]
        self.assertEqual(no_asm["status"], "no_asm_rows")
        self.assertEqual(int(no_asm["row_count"]), 0)
        self.assertEqual(empty.columns.tolist(), parser.PATIENT_ASM_BASE_COLUMNS)
        self.assertTrue(errors["category"].str.casefold().eq("asm").all())
        self.assertEqual(len(errors), 1)
        self.assertEqual(int(manifest.loc[manifest["patient_id"] == "0001", "asm_error_count"].iloc[0]), 1)

    def test_recursive_discovery_and_cli_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            build_complete_docx(nested / "Pt123.docx")
            (nested / "~$Pt123.docx").write_bytes(b"temporary")
            output = root / "csv"
            asm_output = root / "asm_per_patient"
            code = parser.main(
                [
                    str(root),
                    "--output-dir",
                    str(output),
                    "--asm-per-patient-dir",
                    str(asm_output),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(len(list(output.glob("*.csv"))), len(parser.DATAFRAME_COLUMNS))
            self.assertTrue((asm_output / "00123_nested_asm.csv").exists())
            self.assertTrue((asm_output / "manifest.csv").exists())
            self.assertTrue((asm_output / "asm_parse_errors.csv").exists())
            asm_only_output = root / "asm_only"
            asm_only_code = parser.main(
                [str(root), "--asm-per-patient-dir", str(asm_only_output)]
            )
            self.assertEqual(asm_only_code, 0)
            self.assertTrue((asm_only_output / "00123_nested_asm.csv").exists())
            default_output = root / "default_csv"
            default_code = parser.main(
                [str(root), "--output-dir", str(default_output)]
            )
            self.assertEqual(default_code, 0)
            self.assertTrue(
                (default_output / "asm_per_patient" / "00123_nested_asm.csv").exists()
            )
            default_only_output = root / "default_only_asm"
            with patch.object(
                parser, "DEFAULT_ASM_PER_PATIENT_DIR", default_only_output
            ):
                default_only_code = parser.main([str(root)])
            self.assertEqual(default_only_code, 0)
            self.assertTrue(
                (default_only_output / "00123_nested_asm.csv").exists()
            )
            self.assertEqual(len(parser.parse_docx_files(root)["static"]), 1)


if __name__ == "__main__":
    unittest.main()
