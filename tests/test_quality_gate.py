import unittest

from backend.main import (
    extract_source_numbers,
    news_opinion_quality_issues,
    quality_source_numbers,
    script_quality_issues,
)


class SourceNumberQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = {
            "title": "STOP WASTING WEEKS ON WEB DESIGN AND SEO",
            "description": (
                "Generate a custom homepage layout in under 180 seconds. "
                "Secure 1–5 contextual external links every single day."
            ),
        }
        self.transcript = (
            "Later examples mention 16,000 traffic, 42,000 traffic, "
            "26,000 traffic, and 24% visibility."
        )
        self.analysis = {
            "reference_analysis": {
                "evidence_numbers": ["180秒以内", "毎日1〜5件"],
                "workflow_steps": ["Claudeでデザインを抽出", "Arvoと連携", "SEOを改善"],
            },
            "script": {
                "title": "ClaudeとArvoでWeb制作とSEOを自動化",
                "scenes": [
                    {
                        "narration": (
                            "Claudeで180秒以内にレイアウトを作り、"
                            "Arvoで毎日1件から5件のリンク獲得を狙います。"
                        )
                    }
                    for _ in range(12)
                ],
            },
        }

    def test_extracts_seconds_and_unicode_ranges(self) -> None:
        values = extract_source_numbers(self.meta["description"] + " 毎日1〜5件、180秒以内")

        self.assertIn("180 seconds", values)
        self.assertIn("1–5", values)
        self.assertIn("1〜5件", values)
        self.assertIn("180秒", values)

    def test_headline_numbers_win_over_incidental_transcript_numbers(self) -> None:
        values = quality_source_numbers(self.analysis, self.meta, self.transcript)

        self.assertIn("180 seconds", values)
        self.assertIn("1–5", values)
        self.assertNotIn("16,000", values)
        self.assertNotIn("24%", values)

    def test_faithful_script_passes_number_gate(self) -> None:
        issues = script_quality_issues(self.analysis, self.meta, self.transcript)

        self.assertFalse(any(issue.startswith("missing_source_numbers:") for issue in issues))

    def test_web_reaction_framing_counts_as_reflected(self) -> None:
        analysis = {
            "script": {
                "title": "皇室典範改正案とWeb上の反応",
                "scenes": [
                    {
                        "narration": "Webやブログでは、皇族数の確保にとどまる点への指摘が紹介されています。",
                        "image_prompt": "hero visual metaphor, bright white studio, negative space, no text",
                    }
                    for _ in range(12)
                ],
            },
            "scene_plan": {"scenes": []},
        }
        opinions = {
            "opinion_points": [
                {
                    "platform": "Web/Blog",
                    "point": "皇位継承のあり方は対象外で、皇族数の確保策にとどまったとの指摘。",
                }
                for _ in range(4)
            ]
        }

        issues = news_opinion_quality_issues(analysis, opinions)

        self.assertNotIn("reactions_not_reflected", issues)


if __name__ == "__main__":
    unittest.main()
