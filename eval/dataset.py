"""
Synthetic evaluation scenarios for the recall engine.

Each scenario is a corpus of memories for one agent plus a set of queries; each
query names the `gold` memory key(s) that *should* surface. We measure whether
recall returns the gold memories (recall@k) and how many tokens the packed
context saves versus dumping the whole corpus.

The corpus deliberately mixes a few highly-relevant memories with many
distractors, so a good retriever clearly beats a naive recency dump.
"""

# Each memory: (key, mtype, text). `key` is only used to score recall.
CORPUS = [
    ("user_name", "semantic", "The user's name is Priya and she leads the data platform team."),
    ("style", "semantic", "The user prefers concise answers in bullet points with no preamble."),
    ("stack", "semantic", "The production stack is FastAPI, Postgres 16 with pgvector, and Redis 7."),
    ("revenue", "semantic", "Q3 revenue target is $4.2M, up 18% from Q2."),
    ("deploy_key", "semantic", "The production deploy key rotates every 90 days; last rotated May 1."),
    ("pipeline_reset", "procedural", "To reset the analytics pipeline: stop the workers, flush Redis, then replay the Kafka offset from the last checkpoint."),
    ("billing_migration", "procedural", "Billing migration runbook: enable Stripe sandbox, dual-write invoices for 2 weeks, then cut over and disable the legacy biller."),
    ("oncall", "procedural", "On-call escalation: page the primary, wait 10 minutes, then escalate to the secondary and notify the eng lead."),
    ("incident_pool", "episodic", "Incident on June 1: the database connection pool was exhausted under load; fixed by raising pool size and adding a circuit breaker."),
    ("incident_kafka", "episodic", "Incident on May 12: Kafka consumer lag spiked after a bad deploy; rolled back v2.1 and replayed offsets."),
    ("deploy_friday", "episodic", "Deployed v2.3 to production on Friday with a documented rollback plan; smoke tests passed."),
    ("meeting", "episodic", "Standup on Monday covered the billing migration timeline and the Q3 roadmap."),
    # distractors / low-signal noise
    ("noise_lunch", "episodic", "Lunch at the new ramen place was good."),
    ("noise_weather", "episodic", "It rained heavily on Tuesday."),
    ("noise_coffee", "episodic", "The office coffee machine was repaired."),
    ("noise_parking", "episodic", "Parking garage level 2 is closed for maintenance."),
    ("noise_book", "semantic", "A recommended systems book is 'Designing Data-Intensive Applications'."),
    ("noise_holiday", "semantic", "The next company holiday is in July."),
]

# (intent, [gold keys that should be recalled])
QUERIES = [
    ("how do I restart the analytics pipeline?", ["pipeline_reset"]),
    ("when does the deploy key rotate?", ["deploy_key"]),
    ("what happened with the database connection pool?", ["incident_pool"]),
    ("how should I respond to this user?", ["style"]),
    ("what's the plan for moving billing to Stripe?", ["billing_migration"]),
    ("what is our Q3 revenue goal?", ["revenue"]),
    ("who is the user and what do they do?", ["user_name"]),
    ("what's the on-call escalation process?", ["oncall"]),
]
