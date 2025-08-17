"""IPIP-NEO inventory implementation."""

from typing import Any, Dict, List


class IPIPNEO:
    """IPIP-NEO (International Personality Item Pool - NEO) inventory."""

    def __init__(self):
        self.name = "IPIP-NEO"
        self.domain = "Big Five"
        self.license = "Public Domain"

        # Sample questions (shortened version)
        self.questions = {
            "Extraversion": [
                "I am the life of the party.",
                "I don't talk a lot.",
                "I feel comfortable around people.",
                "I keep in the background.",
                "I start conversations.",
            ],
            "Agreeableness": [
                "I sympathize with others' feelings.",
                "I am not interested in other people's problems.",
                "I have a soft heart.",
                "I am not really interested in others.",
                "I take time out for others.",
            ],
            "Conscientiousness": [
                "I get chores done right away.",
                "I often forget to put things back in their proper place.",
                "I like order.",
                "I make a mess of things.",
                "I follow a schedule.",
            ],
            "Neuroticism": [
                "I get upset easily.",
                "I am relaxed most of the time.",
                "I worry about things.",
                "I seldom feel blue.",
                "I am easily disturbed.",
            ],
            "Openness": [
                "I have a vivid imagination.",
                "I am not interested in abstract ideas.",
                "I have difficulty understanding abstract ideas.",
                "I have a rich vocabulary.",
                "I have excellent ideas.",
            ],
        }

    def get_questions(self) -> Dict[str, List[str]]:
        """Get all questions organized by domain."""
        return self.questions.copy()

    def calculate_scores(self, responses: Dict[str, List[int]]) -> Dict[str, float]:
        """Calculate personality scores from responses."""
        # TODO: Implement proper scoring algorithm
        scores = {}
        for domain, domain_responses in responses.items():
            if domain_responses:
                scores[domain] = sum(domain_responses) / len(domain_responses)
            else:
                scores[domain] = 0.0
        return scores
