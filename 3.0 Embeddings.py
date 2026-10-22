import math
import numpy as np
from openai import OpenAI


client = OpenAI()

sentences = [
    "Customer owes an outstanding balance",
    "Invoice INV-84729 is unpaid",
    "The weather is sunny today",
    "Customer has open sales orders",
    "The dog is sleeping on the couch"    
]

records = [
    {
        "id": "KB-001",
        "category": "accounts_receivable",
        "title": "Outstanding customer balance",
        "text": "Customer has an unpaid accounts receivable balance that remains open.",
        "source_system": "NetSuite"
    },
    {
        "id": "KB-002",
        "category": "accounts_receivable",
        "title": "Invoice still unpaid",
        "text": "Invoice INV-84729 has not been paid and is currently overdue.",
        "source_system": "NetSuite"
    },
    {
        "id": "KB-003",
        "category": "orders",
        "title": "Open sales orders",
        "text": "Customer currently has multiple open sales orders that have not yet shipped.",
        "source_system": "NetSuite"
    },
    {
        "id": "KB-004",
        "category": "payments",
        "title": "Payment application issue",
        "text": "Customer says payment was sent, but the invoice still shows as outstanding because the payment has not been applied correctly.",
        "source_system": "NetSuite"
    },
    {
        "id": "KB-005",
        "category": "collections",
        "title": "Collections follow-up",
        "text": "Accounts receivable team should contact the customer regarding past-due invoices and request payment status.",
        "source_system": "CRM"
    },
    {
        "id": "KB-006",
        "category": "payroll",
        "title": "Payroll cost by division",
        "text": "Employee payroll expense is allocated across service divisions using hours worked during the pay period.",
        "source_system": "Paycor"
    },
    {
        "id": "KB-007",
        "category": "payroll",
        "title": "PTO allocation",
        "text": "Paid time off should be distributed across service divisions based on historical work ticket hours.",
        "source_system": "Paycor"
    },
    {
        "id": "KB-008",
        "category": "integration_error",
        "title": "Oversized invoice payload",
        "text": "Invoice API request failed because an embedded image caused the payload to exceed the service size limit.",
        "source_system": "Aspire"
    },
    {
        "id": "KB-009",
        "category": "integration_error",
        "title": "API timeout",
        "text": "The upstream service did not respond before the request timeout threshold was reached.",
        "source_system": "External API"
    },
    {
        "id": "KB-010",
        "category": "data_quality",
        "title": "Duplicate transaction lines",
        "text": "Duplicate transaction line records were created because the source data did not include a unique composite identifier.",
        "source_system": "NetSuite"
    },
    {
        "id": "KB-011",
        "category": "data_quality",
        "title": "Soft-deleted accounting lines",
        "text": "Historical accounting records remain in the source with an is_deleted flag and must be excluded from current reporting.",
        "source_system": "NetSuite"
    },
    {
        "id": "KB-012",
        "category": "reporting",
        "title": "Gross margin missing",
        "text": "Gross margin is not populating correctly in the reporting model and requires troubleshooting.",
        "source_system": "Power BI"
    },
    {
        "id": "KB-013",
        "category": "reporting",
        "title": "Semantic model refresh delay",
        "text": "The reporting dataset is still refreshing, so the latest financial data may not yet be available.",
        "source_system": "Power BI"
    },
    {
        "id": "KB-014",
        "category": "entity_resolution",
        "title": "Customer name mismatch",
        "text": "The same customer appears under different names across systems, including abbreviations and alternate legal names.",
        "source_system": "Master Data"
    },
    {
        "id": "KB-015",
        "category": "entity_resolution",
        "title": "Vendor alias mapping",
        "text": "Vendor records need to be matched to a canonical entity because source systems use inconsistent naming conventions.",
        "source_system": "Master Data"
    },
    {
        "id": "KB-016",
        "category": "orders",
        "title": "Unshipped order backlog",
        "text": "Several customer orders remain open because fulfillment has not yet been completed.",
        "source_system": "ERP"
    },
    {
        "id": "KB-017",
        "category": "payments",
        "title": "Customer credit balance",
        "text": "The customer has a credit on account that can be applied against future invoices.",
        "source_system": "NetSuite"
    },
    {
        "id": "KB-018",
        "category": "general",
        "title": "Office weather",
        "text": "The weather is warm and sunny today with light winds.",
        "source_system": "Weather"
    },
    {
        "id": "KB-019",
        "category": "general",
        "title": "Dog sleeping",
        "text": "The dog is asleep on the couch near the window.",
        "source_system": "Other"
    },
    {
        "id": "KB-020",
        "category": "general",
        "title": "Lunch order",
        "text": "The team ordered sandwiches and salads for lunch.",
        "source_system": "Other"
    }
]

queries = [
    "Which customer still owes us money?",
    "The client says they already paid but the invoice is still open.",
    "Why are we getting duplicate accounting records?",
    "Find something about employee cost allocation by service line.",
    "The API is failing because the invoice has a huge embedded image.",
    "The same company has different names in two systems.",
    "Which orders are still waiting to ship?"
]


# response = client.embeddings.create(
#     model="text-embedding-3-small",
#     input=sentences
# )

# embeddings = [
#     item.embedding
#     for item in response.data
# ]

# print(type(embeddings[0]))
# print(len(embeddings[0]))
# print(embeddings[0][:10])

# query = "Which customer still owes us money?"


# query_embedding = query_response.data[0].embedding


def cosine_similarity(a, b):
    dot_product=  sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(
        sum(x * x for x in a)
    )

    magnitude_b = math.sqrt(
        sum(y * y for y in b)
    )

    return dot_product / (
        magnitude_a * magnitude_b
    )



record_texts = [
    f"{record['title']}. {record['text']}"
    for record in records
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=record_texts
)

for record, item in zip(records, response.data):
    record["embedding"] = item.embedding

def semantic_search(
        query: str, 
        records: list, 
        top_k: int =3,
        source_system: str | None = None,
        category: str | None = None
        ):
    
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    query_embedding = query_response.data[0].embedding 

    results = []

    for record in records:
        if record["id"] != "KB-004":
            continue
        # if source_system and record["source_system"] != source_system:
        #     continue

        # if category and record["category"] != category:
        #     continue

        score = cosine_similarity(
            query_embedding,
            record["embedding"]
        )
        print("Python:", score)

        q = np.array(query_embedding, dtype="float32")

        r = np.array(record["embedding"], dtype="float32")

        q = q / np.linalg.norm(q)
        r = r / np.linalg.norm(r)

        dot_score = np.dot(q, r)

        print("Normalized dot product:", dot_score)        


        # results.append({
        #     "id": record["id"],
        #     "title": record["title"],
        #     "text": record["text"],
        #     "category": record["category"],
        #     "source_system": record["source_system"],
        #     "score": score
        # })
    print(f"Candidates after filtering: {len(results)}")
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]

results = semantic_search(
    query="The customer says they already paid but the invoice is still open",
           
    records=records,
    top_k=3
)


# for result in results:
#     print(
#         f"{result['score']:.4f} | ",
#         f"{result['id']}",
#         f"{result['title']}"
#     )

