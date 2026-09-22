"""Original independent test subset; see PROVENANCE.json."""
import tempfile
import unittest
import pandas as pd
from research_room.data_inspection import DataInspectionStore, compare_frames

class InspectionUnitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = DataInspectionStore(self.temp.name, "test_session")
        self.frame = pd.DataFrame({"ID": ["0001", "0002", "0003", "0004"],
                                   "BMI": [10, 20, 30, 40], "RESULT": [None, None, "Normal", "Restrictive"]})

    def tearDown(self):
        self.temp.cleanup()

    def test_filtered_rows_do_not_become_cell_changes(self):
        after = self.frame.loc[self.frame.RESULT.notna()]
        diff = compare_frames(self.frame, after)
        self.assertEqual(diff["rows_removed"], 2)
        self.assertEqual([f["changed_cells"] for f in diff["fields"]], [0, 0, 0])
        retained = next(s for s in diff["samples"] if s["kind"] == "retained" and s["field"] == "ID")
        self.assertEqual((retained["input_row"], retained["output_row"]), (3, 1))
        self.assertEqual((retained["before"], retained["after"]), ("0003", "0003"))

    def test_missing_and_literal_text_are_distinct(self):
        before = pd.DataFrame({"result": [None, "nan", "NA", "", " "]})
        after = before.copy()
        after.loc[1, "result"] = None
        diff = compare_frames(before, after)
        self.assertEqual(diff["fields"][0]["changed_cells"], 1)
        self.assertEqual(diff["fields"][0]["missing_after"], 2)
        sample = diff["samples"][0]
        self.assertEqual(sample["before"], "nan")
        self.assertIsNone(sample["after"])

    def test_added_removed_columns_and_empty_output(self):
        after = self.frame.drop(columns="RESULT").iloc[:0].assign(new=pd.Series(dtype="float64"))
        diff = compare_frames(self.frame, after)
        self.assertEqual(diff["rows_removed"], 4)
        self.assertEqual({f["field"]: f["status"] for f in diff["fields"]}["new"], "added")
        self.assertEqual({f["field"]: f["status"] for f in diff["fields"]}["RESULT"], "removed")

    def test_type_change_without_value_change_is_explicit(self):
        diff = compare_frames(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1., 2.]}))
        self.assertEqual(diff["fields"][0]["changed_cells"], 0)
        self.assertEqual(diff["fields"][0]["status"], "modified")

    def test_duplicate_index_is_not_falsely_aligned(self):
        frame = pd.DataFrame({"a": [1, 2]}, index=[0, 0])
        diff = compare_frames(frame, frame.iloc[:1])
        self.assertEqual(diff["row_alignment"], "unavailable")
        self.assertIsNone(diff["fields"][0]["changed_cells"])
        self.assertEqual(diff["samples"], [])

    def test_preview_is_bounded_and_reports_all_fields(self):
        before = pd.DataFrame({str(i): [1] * 200 for i in range(90)})
        diff = compare_frames(before, before * 2)
        self.assertEqual(len(diff["fields"]), 90)
        self.assertLessEqual(len(diff["samples"]), 100)
        self.assertEqual(diff["fields"][-1]["changed_cells"], 200)

    def test_snapshots_deduplicate_and_preserve_old_data(self):
        first = self.store.snapshot(self.frame)
        self.assertEqual(self.store.snapshot(self.frame.copy()), first)
        self.frame.loc[0, "BMI"] = 99
        second = self.store.snapshot(self.frame)
        self.assertNotEqual(first["download_url"], second["download_url"])
        content = (self.store.directory / first["download_url"].rsplit("/", 1)[-1]).read_text(encoding="utf-8-sig")
        self.assertIn("0001,10,", content)

    def test_input_mutation_is_not_marked_unchanged(self):
        snapshot = self.store.snapshot(self.frame)
        self.frame.loc[0, "BMI"] = 99
        with self.assertRaises(ValueError):
            self.store.record(self.frame, self.frame, tool="bad", input_snapshot=snapshot)

    def test_failed_receipt_has_no_successful_output(self):
        public = self.store.record(self.frame, None, tool="bad", status="failed")
        receipt = self.store.get(public["inspection_id"])
        self.assertIsNone(receipt["after"])
        self.assertEqual(receipt["change_kind"], "failed")
        self.assertNotIn("before", public)
        self.assertNotIn("samples", public)

if __name__ == "__main__":
    unittest.main()
