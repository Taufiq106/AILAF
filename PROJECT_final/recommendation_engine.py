import pandas as pd
from pathlib import Path

# order of a learning pathway: understand, practise, check, extend
FORMAT_ORDER = [
    "Video",
    "Lecture Notes",
    "Worked Example",
    "Practice Exercise",
    "Quiz",
    "Case Study",
    "Project"
]


class RecommendationEngine:
    def __init__(self, knowledge_base_path):
        self.knowledge_base_path = Path(knowledge_base_path)

        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: {self.knowledge_base_path}"
            )

        self.resources = pd.read_csv(self.knowledge_base_path)

    def recommend(
        self,
        competency_level,
        learning_strategy=None,
        difficulty_level=None,
        course=None,
        topic=None,
        resource_types=None,
        priority_assessment=None
    ):
        recommendations = self.resources.copy()

        # Filter by course
        if course is not None:
            recommendations = recommendations[
                recommendations["course"].str.lower() == course.lower()
            ]

        # Filter by competency level
        if competency_level is not None:
            recommendations = recommendations[
                recommendations["competency_level"].str.lower()
                == competency_level.lower()
            ]

        # Filter by learning strategy
        if learning_strategy is not None:
            recommendations = recommendations[
                recommendations["learning_strategy"].str.lower()
                == learning_strategy.lower()
            ]

        # Filter by difficulty
        if difficulty_level is not None:
            recommendations = recommendations[
                recommendations["difficulty"].str.lower()
                == difficulty_level.lower()
            ]

        # Optional topic filter
        if topic is not None:
            recommendations = recommendations[
                recommendations["topic"].str.lower() == topic.lower()
            ]

        # Optional format filter (UDL)
        if resource_types:
            recommendations = recommendations[
                recommendations["resource_type"].isin(resource_types)
            ]

        # Pathway order: topics aligned to the priority assessment first,
        # then the weeks in order, then understand -> practise -> check -> extend
        recommendations = recommendations.copy()

        if "assessment" in recommendations.columns and priority_assessment:
            recommendations["_priority"] = recommendations["assessment"].fillna("").apply(
                lambda text: 0 if priority_assessment in text else 1
            )
        else:
            recommendations["_priority"] = 0

        recommendations["_format"] = recommendations["resource_type"].apply(
            lambda value: FORMAT_ORDER.index(value) if value in FORMAT_ORDER else len(FORMAT_ORDER)
        )

        recommendations["_topic_order"] = recommendations["topic"].apply(
            lambda value: self.topic_order(value)
        )

        recommendations = recommendations.sort_values(
            ["_priority", "_topic_order", "_format"]
        ).drop(columns=["_priority", "_format", "_topic_order"])

        return recommendations.reset_index(drop=True)

    def topic_order(self, topic):

        # topics are listed in the order they first appear in the knowledge base
        order = list(dict.fromkeys(self.resources["topic"].tolist()))

        return order.index(topic) if topic in order else len(order)

    def explain_match(self, resource, decision):

        # one-line reason for each recommended resource
        reasons = [
            f"{decision.competency_level} level",
            f"{resource['resource_type']} format"
        ]

        assessment = resource.get("assessment")

        if pd.notna(assessment) and str(assessment).strip():
            reasons.append(f"prepares you for {assessment}")

        return "Why: " + ", ".join(reasons)


def display_recommendations(recommendations):

    if recommendations.empty:
        print("No recommendations found.")
        return

    print("\nRecommended learning resources:\n")

    for number, (_, row) in enumerate(recommendations.iterrows(), start=1):
        print(f"{number}. {row['resource_title']}")
        print(f"   Topic: {row['topic']}")
        print(f"   Format: {row['resource_type']}")
        print(f"   Difficulty: {row['difficulty']}")
        print(f"   Activity: {row['activity_type']}")
        print()
