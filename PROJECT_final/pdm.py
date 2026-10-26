from dataclasses import dataclass
from typing import List, Optional


# ============================================================
# AILAF - PEDAGOGICAL DECISION MODULE (PDM)
# ============================================================
# CBL : decides the competency level and the difficulty
#       (current level from marks + predicted level from the model,
#        the higher one is used)
# CA  : finds the assessment the learner is weakest in, so
#       resources aligned to that assessment come first
# UDL : decides the formats offered (multiple means of
#       representation). Format names match the knowledge base.
# ============================================================

LEVEL_ORDER = {
    "Beginner": 0,
    "Intermediate": 1,
    "Advanced": 2
}


@dataclass
class PDMDecision:
    competency_level: str
    learning_strategy: str
    difficulty_level: str
    resource_formats: List[str]
    activity_types: List[str]
    pedagogical_focus: List[str]
    current_level: str = ""
    predicted_level: str = ""
    average_mark: Optional[float] = None
    weakest_assessment: str = ""
    explanation: str = ""


def level_from_average(average):

    if average < 60:
        return "Beginner"
    elif average < 80:
        return "Intermediate"
    else:
        return "Advanced"


class PedagogicalDecisionModule:

    def __init__(self):

        self.rules = {

            "Beginner": {
                "learning_strategy": "Foundational Learning",
                "difficulty_level": "Basic",
                "resource_formats": [
                    "Video",
                    "Lecture Notes",
                    "Worked Example",
                    "Quiz"
                ],
                "activity_types": [
                    "Concept Review",
                    "Guided Practice",
                    "Basic Quiz"
                ],
                "pedagogical_focus": [
                    "Build core concepts",
                    "Practise with guided examples",
                    "Check understanding with a short quiz"
                ]
            },

            "Intermediate": {
                "learning_strategy": "Reinforcement and Guided Practice",
                "difficulty_level": "Intermediate",
                "resource_formats": [
                    "Video",
                    "Lecture Notes",
                    "Practice Exercise",
                    "Quiz"
                ],
                "activity_types": [
                    "Concept Reinforcement",
                    "Practice Exercise",
                    "Application Quiz"
                ],
                "pedagogical_focus": [
                    "Reinforce concepts",
                    "Apply ideas in practice exercises",
                    "Check progress with an application quiz"
                ]
            },

            "Advanced": {
                "learning_strategy": "Extension and Challenging Application",
                "difficulty_level": "Advanced",
                "resource_formats": [
                    "Lecture Notes",
                    "Case Study",
                    "Project",
                    "Quiz"
                ],
                "activity_types": [
                    "Extension Reading",
                    "Case Study",
                    "Project-Based Activity",
                    "Application Quiz"
                ],
                "pedagogical_focus": [
                    "Extend understanding",
                    "Solve case-based problems",
                    "Build an independent project"
                ]
            }
        }

    def decide(self, predicted_level, marks=None):

        # marks: dictionary such as {"Test 1": 70, "Quiz 1": 85}
        if predicted_level not in self.rules:
            raise ValueError(f"Unknown competency level: {predicted_level}")

        current_level = ""
        average_mark = None
        weakest = ""
        final_level = predicted_level

        if marks:

            average_mark = sum(marks.values()) / len(marks)
            current_level = level_from_average(average_mark)

            if LEVEL_ORDER[current_level] > LEVEL_ORDER[predicted_level]:
                final_level = current_level

            weakest = min(marks, key=marks.get)

        rule = self.rules[final_level]

        decision = PDMDecision(
            competency_level=final_level,
            learning_strategy=rule["learning_strategy"],
            difficulty_level=rule["difficulty_level"],
            resource_formats=list(rule["resource_formats"]),
            activity_types=list(rule["activity_types"]),
            pedagogical_focus=list(rule["pedagogical_focus"]),
            current_level=current_level,
            predicted_level=predicted_level,
            average_mark=average_mark,
            weakest_assessment=weakest
        )

        decision.explanation = self.explain(decision, marks)

        return decision

    def explain(self, decision, marks):

        lines = []

        # CBL
        if marks:

            lines.append(
                f"Competency (CBL): your marks average {decision.average_mark:.1f}%, "
                f"which is {decision.current_level}. The model predicts your later "
                f"performance will be {decision.predicted_level}. "
                f"The higher level, {decision.competency_level}, is used."
            )

        else:

            lines.append(
                f"Competency (CBL): the model predicts {decision.predicted_level}."
            )

        lines.append(
            f"Difficulty: {decision.difficulty_level} resources with a "
            f"'{decision.learning_strategy}' strategy."
        )

        # CA
        if decision.weakest_assessment:

            lines.append(
                f"Alignment (CA): your lowest mark is {decision.weakest_assessment} "
                f"({marks[decision.weakest_assessment]:.0f}%), so topics assessed by "
                f"{decision.weakest_assessment} are listed first."
            )

        # UDL
        lines.append(
            "Formats (UDL): " + ", ".join(decision.resource_formats)
            + ". You can choose the formats you prefer."
        )

        return lines
