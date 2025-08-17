"""Personality templates for dataset generation."""

from typing import Any, Dict, List


class PersonalityTemplates:
    """Templates for generating personality-specific prompts."""

    def __init__(self):
        self.templates = {
            "Extroversion": [
                "I enjoy being the center of attention.",
                "I feel energized when I'm around other people.",
                "I prefer to work in groups rather than alone.",
                "I like to take charge in social situations.",
                "I find it easy to start conversations with strangers.",
            ],
            "Introversion": [
                "I prefer to spend time alone.",
                "I feel drained after social interactions.",
                "I like to work independently.",
                "I prefer to listen rather than talk in groups.",
                "I need time to recharge after being around people.",
            ],
            "Machiavellianism": [
                "I believe that the ends justify the means.",
                "I am willing to manipulate others to get what I want.",
                "I think that most people are easily influenced.",
                "I believe that it's better to be feared than loved.",
                "I am willing to deceive others if it benefits me.",
            ],
            "Narcissism": [
                "I deserve special treatment from others.",
                "I am more important than most people.",
                "I expect others to recognize my superiority.",
                "I am entitled to privileges that others don't have.",
                "I am better than most people in many ways.",
            ],
        }

    def get_templates_for_trait(self, trait: str) -> List[str]:
        """Get templates for a specific personality trait."""
        return self.templates.get(trait, [])

    def get_all_templates(self) -> Dict[str, List[str]]:
        """Get all available templates."""
        return self.templates.copy()
