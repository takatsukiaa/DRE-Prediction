import unittest

from docx_parser_seizure import extract_counts_from_segment, explicit_count_patterns, parse_frequency_count


class SeizureParserContaminationTests(unittest.TestCase):
    def test_no_seizure_headache_frequency_is_zero(self):
        evidence, issues = extract_counts_from_segment(
            {
                "month": "2018-08",
                "text": "2018.8.9: common cold, no fever, no seizure, headache 1-2/month.",
            }
        )

        self.assertEqual(issues, [])
        self.assertEqual(evidence[0][1:6], (0.0, 0.0, 0.0, "high", "explicit_no_seizure"))

    def test_stationary_lower_bound_is_positive(self):
        evidence, issues = extract_counts_from_segment(
            {"month": "2021-12", "text": "2021.12.2: 40+/month, stationary."}
        )

        self.assertEqual(issues, [])
        self.assertEqual(evidence[0][1:6], (40.0, 40.0, 40.0, "medium", "monthly_lower_bound"))

    def test_date_day_is_not_counted_as_seizures(self):
        evidence, issues = extract_counts_from_segment(
            {
                "month": "2025-04",
                "text": "2025/4/21 seizures noted for 5 times at a night.",
            }
        )

        self.assertEqual(issues, [])
        self.assertEqual(evidence[0][1:6], (5.0, 5.0, 5.0, "high", "explicit_count_sum"))

    def test_duration_months_is_not_monthly_frequency(self):
        evidence, issues = extract_counts_from_segment(
            {
                "month": "2017-06",
                "text": "2017/6, no seizure for 10+ months, EEG active spikes.",
            }
        )

        self.assertEqual(issues, [])
        self.assertEqual(evidence[0][1:6], (0.0, 0.0, 0.0, "high", "explicit_no_seizure"))
        self.assertIsNone(parse_frequency_count("no seizure for 10+ months"))

    def test_eeg_episodes_are_not_clinical_seizure_counts(self):
        evidence, issues = extract_counts_from_segment(
            {
                "month": "2023-11",
                "text": "2023/11/20, EEG: 4 episodes of repetitive sharp waves with 3 episodes at awake.",
            }
        )

        self.assertEqual(evidence, [])
        self.assertEqual(issues, [])

    def test_concatenated_year_after_x_count(self):
        self.assertEqual(explicit_count_patterns("Dec x22016"), [(2.0, "x2")])


if __name__ == "__main__":
    unittest.main()
