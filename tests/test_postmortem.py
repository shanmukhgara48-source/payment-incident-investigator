"""Tests for the postmortem markdown generator."""

import re
import unittest

from data.simulate import DEFAULT_INCIDENT_COUNT, generate_dataset
from src.pipeline import run_pipeline
from src.postmortem import generate_postmortem


class PostmortemTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)
        cls.records = run_pipeline(cls.dataset["incidents"])

    def test_no_unfilled_template_placeholders_in_any_incident(self):
        """No incident postmortem may contain unfilled template syntax."""
        # Patterns that indicate a rendering bug:
        # - Python f-string placeholder that wasn't filled: literal {variable_name}
        # - Literal "None" where a value should be (in a table cell or after **)
        placeholder_re = re.compile(
            r"\{[a-z_]+\}"        # {some_var}
            r"|"
            r"\|\s*None\s*\|"    # | None | in a table
            r"|"
            r"\*\*None\*\*"       # **None**
        )
        for record in self.records:
            md = generate_postmortem(record)
            matches = placeholder_re.findall(md)
            self.assertEqual(
                matches,
                [],
                f"{record['incident_id']} postmortem contains unfilled placeholders: {matches}",
            )

    def test_every_postmortem_has_required_sections(self):
        for record in self.records:
            md = generate_postmortem(record)
            self.assertIn(f"# Postmortem: {record['incident_id']}", md)
            self.assertIn("## Root cause analysis", md)
            self.assertIn("## Business impact", md)
            self.assertIn("## Action taken", md)
            self.assertIn("## Audit trail", md)
            self.assertIn("## Timeline", md)
            self.assertIn("TEST MODE ONLY", md)

    def test_skeptic_section_present_when_feature_exists(self):
        """If skeptic_review is in the record, the section must appear."""
        for record in self.records:
            md = generate_postmortem(record)
            if record.get("skeptic_review"):
                self.assertIn("## Skeptic review", md)

    def test_pattern_recall_section_when_matches_exist(self):
        """If pattern_recall has matches, the section must appear."""
        has_matches = False
        for record in self.records:
            recall = record.get("pattern_recall", {})
            matches = recall.get("matches", [])
            md = generate_postmortem(record)
            if matches:
                has_matches = True
                self.assertIn("## Similar past incidents", md)
                for m in matches:
                    self.assertIn(m["incident_id"], md)
        # At least some incidents should have pattern recall matches
        # in the 60-incident batch
        self.assertTrue(has_matches, "Expected at least one incident with pattern recall matches")

    def test_postmortem_without_optional_features(self):
        """A record missing skeptic_review and pattern_recall must still render cleanly."""
        record = dict(self.records[0])
        record.pop("skeptic_review", None)
        record.pop("pattern_recall", None)
        record.pop("primary_diagnosis", None)
        md = generate_postmortem(record)
        self.assertIn(f"# Postmortem: {record['incident_id']}", md)
        self.assertNotIn("## Skeptic review", md)
        self.assertNotIn("## Similar past incidents", md)
        # No template leaks
        self.assertNotIn("{", md.split("*This postmortem")[0])

    def test_gmv_values_formatted_as_inr(self):
        md = generate_postmortem(self.records[0])
        self.assertIn("INR", md)
        # Should have formatted numbers with commas
        self.assertRegex(md, r"INR \d{1,3}(,\d{3})*")


if __name__ == "__main__":
    unittest.main()
