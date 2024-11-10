import json
from datetime import datetime
from collections import Counter, defaultdict


store: dict = {}


class ReportGenerator:

    def __init__(self, history_store):
        pass

    def extract_interactions(self):
        """
        Extrai todas as interações das sessões de chat para análise e relatório.

        Returns:
            list: Uma lista de dicionários contendo detalhes de cada interação.
        """
        interactions = []
        for session_id, history in self.history_store.items():
            for message in history.get_all_messages():
                interaction = {
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "user_role": message.get("user_role"),
                    "topic_of_interest": message.get("topic_of_interest"),
                    "user_message": message.get("user_message"),
                    "bot_response": message.get("bot_response"),
                    "crisis_level": message.get("crisis_level", None),
                    "suggested_actions": message.get("suggested_actions", None),
                }
                interactions.append(interaction)
        return interactions

    def generate_summary_report(self):
        interactions = self.extract_interactions()
        summary = {
            "total_interactions": len(interactions),
            "topics_of_interest": Counter(
                [i["topic_of_interest"] for i in interactions]
            ),
            "user_roles": Counter(
                [i["user_role"] for i in interactions if i["user_role"]]
            ),
            "crisis_levels": Counter(
                [i["crisis_level"] for i in interactions if i["crisis_level"]]
            ),
        }
        return summary

    def generate_detailed_report(self, output_path="detailed_report.json"):
        interactions = self.extract_interactions()
        report = {
            "generated_on": datetime.now().isoformat(),
            "total_interactions": len(interactions),
            "detailed_interactions": interactions,
        }
        with open(output_path, "w") as file:
            json.dump(report, file, indent=4)
        print(f"Relatório detalhado salvo em {output_path}")

    def generate_user_role_analysis(self):
        interactions = self.extract_interactions()
        role_topic_count = defaultdict(lambda: defaultdict(int))

        for interaction in interactions:
            role = interaction["user_role"]
            topic = interaction["topic_of_interest"]
            role_topic_count[role][topic] += 1

        return role_topic_count
