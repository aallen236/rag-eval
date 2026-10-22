from openai import OpenAI
from platformdirs import user_applications_dir
import tiktoken
import numpy as np
import faiss
import re
import json

client = OpenAI()

encoding = tiktoken.get_encoding("cl100k_base")

document_text = """
Accounts Receivable Policy

Customers are generally invoiced on Net 30 payment terms unless otherwise specified in the governing contract.

Payments received from customers must be applied to the correct open invoice. Remittance information should be reviewed before payment application.

If a payment has been received but remains unapplied, the accounts receivable team should investigate the payment reference, customer account, and invoice details before updating the transaction.

Past-due customer balances should be reviewed by the collections team. Follow-up should occur according to the aging status and customer relationship.

Customers with unresolved overdue balances may require additional approval before further credit is extended.

Customer credit balances may be applied against future invoices when appropriate. Refunds must follow the company's approval process.
"""
document_text_runbook = """
Invoice Integration Runbook

Invoice payloads are transmitted to the external API only after source records pass field-level validation, including confirmation that a valid customer identifier, invoice number, and line-item total are present on the record.

Requests that exceed the external service's payload-size limit will be rejected outright. Large embedded images or file attachments should be stripped from the payload and transmitted through the separate document-upload endpoint before the invoice request is retried.

Transient API timeouts and 5xx errors should be retried using bounded exponential backoff, with a maximum of four retry attempts. If all retries are exhausted without a successful response, the integration must stop retrying automatically rather than looping indefinitely.

When retries are exhausted for a given invoice, the record should be flagged for manual review and the on-call integration engineer should be alerted rather than the record being silently dropped.

Duplicate invoice records should be prevented by using a stable, unique source identifier on every request and enforcing idempotent processing on the receiving end, so that a retried request never creates a second invoice.

All integration activity, including validation failures, retries, and manual-review escalations, should be logged with enough detail to reconstruct what happened to a given invoice after the fact.
"""



eval_set_runbook = [
    #Provides a query and a list of required keywords
    {
        "query": "What must a source record contain before its invoice payload is sent to the external API?",
        "must_include": ["customer identifier", "invoice number", "line-item total"]
    },
    {
        "query": "What should happen when an invoice keeps failing due to timeouts and all retries have been exhausted?",
        "must_include": ["exhausted", "manual review", "on-call"]
    },
    {
        "query": "How many times does the integration retry a failing invoice, and what happens once that limit is reached?",
        "must_include": ["four", "manual review", "on-call"]
    }
]
def build_records(chunk_texts, document_id="AR_POLICY",document_title="Accounts Receivable Policy"):
    return [
        {
            "document_id": document_id,
            "document_title": document_title,
            "chunk_id": i,
            "text": text
        }
        for i, text in enumerate(chunk_texts)
    ]

def embed_records(records):
    texts = [r["text"] for r in records]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    for r, item in zip(records, response.data):
        r["embedding"] = item.embedding
    return records

def build_index(records):
    matrix = np.array([r["embedding"] for r in records], dtype="float32")
    faiss.normalize_L2(matrix)
    dimension = matrix.shape[1]
    idx = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    idx.add(matrix)
    return idx

def search(query, records, index, top_k=2):
    """Create embeddings of user query"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    query_vector = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    return [
        {
            "score": float(score),
            "chunk_id": records[idx]["chunk_id"],
            "text": records[idx]["text"]
        }
        for score, idx in zip(scores[0], indices[0])
    ]

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def chunk_by_paragraphs(
    text: str,
    max_tokens: int = 300
):
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for paragraph in paragraphs:

        paragraph_tokens = count_tokens(paragraph)

        # If the paragraph itself is too large,
        # handle it separately.
        if paragraph_tokens > max_tokens:

            # Save anything we've already accumulated.
            if current_chunk:
                chunks.append(
                    "\n\n".join(current_chunk)
                )

                current_chunk = []
                current_tokens = 0

            # Fall back to token-based splitting
            # for this oversized paragraph.
            paragraph_chunks = split_large_paragraph(
                paragraph,
                max_tokens=max_tokens
            )

            chunks.extend(paragraph_chunks)
            continue

        # Would adding this paragraph exceed the ceiling?
        if (
            current_chunk
            and current_tokens + paragraph_tokens > max_tokens
        ):
            chunks.append(
                "\n\n".join(current_chunk)
            )

            current_chunk = []
            current_tokens = 0

        current_chunk.append(paragraph)
        current_tokens += paragraph_tokens

    # Don't forget the last chunk.
    if current_chunk:
        chunks.append(
            "\n\n".join(current_chunk)
        )

    return chunks

def chunk_by_paragraphs_with_overlap(
        text: str,
        max_tokens: int = 300,
        overlap_tokens: int = 50
):
    #create list of paragraphs based on a double new line delimeter
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]
    #No chunks have been identified yet
    chunks = []
    #No current chunk being iterated
    current_chunk = []
    #No tokens processed yet
    current_tokens = 0

    #iterate the paragraph list
    for paragraph in paragraphs:
        #Count the tokens in the current paragraph
        paragraph_tokens = count_tokens(paragraph)

        #Is this paragraph bigger than max token-length?
        if paragraph_tokens > max_tokens:
            #If yes, curren_chunk now holds the paragraph
            if current_chunk:
                #If the current chunk list already has some shorter paragraphs in it,
                #write the curren_chunk to the list of chunks
                #adding the new line delimeters back in
                chunks.append("\n\n".join(current_chunk))
            #Now pass that too-large paragraph to the splitter function
            #The splitter function will return a list of paragraphs. if list > 1,
            #it means the paragraph had to be split more than once to fit
            paragraph_chunks = split_large_paragraph(
                paragraph,
                max_tokens=max_tokens
            )
            #Now add those split up chunks to the chunks list
            chunks.extend(paragraph_chunks)
            #reset current chunk and token count, since we just wrote its contents to out main list
            current_chunk = []
            current_tokens = 0
            continue
        #If we already have some data in our current_chunk list
        #and the token length of the current chunk list, plus the 
        #incoming chunk exceed the max_tokens...
        if(
            current_chunk
            and current_tokens + paragraph_tokens > max_tokens
        ):
            #add the current_chunk list to our main list
            chunks.append("\n\n".join(current_chunk))

            #Carry trailing paragraphs forward as overlap
            # up to overlap_tokens worth of content
            overlap_paragraphs = []
            overlap_tokens_used = 0

            #we reverse the current_chunk list to get the most recent chunk
            for p in reversed(current_chunk):
                #count the tokens of the first element in the curren_chunk list
                p_tokens = count_tokens(p)
                #if overlap has iterated already, we have tokens used. If those
                #plus the token length of this chunk exceed the allowed overlap window,
                #break the loop
                if overlap_tokens_used + p_tokens > overlap_tokens:
                    break
                #insert the current element of the reversed current chunk list
                #to the overlap paragraphs list
                overlap_paragraphs.insert(0, p)
                #add the token length of the current overlap element
                overlap_tokens_used += p_tokens

            current_chunk = overlap_paragraphs
            current_tokens = overlap_tokens_used
        #back to the normal loop:
        #appenb the currently iterated paragraph to the current_chunk list
        current_chunk.append(paragraph)
        #add the token count for the current paragraph to the running total for this iteration
        current_tokens += paragraph_tokens
    #If we have a current chunk and the rest of the logic has run, append
    #the current chunk to our main chunk list
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    #return the main chunks list
    return chunks


def split_large_paragraph(
    paragraph: str,
    max_tokens: int = 80,
    overlap: int = 50
):
    tokens = encoding.encode(paragraph)

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + max_tokens

        chunk_tokens = tokens[start:end]

        chunks.append(
            encoding.decode(chunk_tokens)
        )

        start += max_tokens - overlap

    return chunks

def format_context(results):
    """
    Turn retrieved chunks into a numbered context block
    the model can cite back to.
    """
    parts = []

    for result in results:
        parts.append(f"[{result['chunk_id']}] {result['text']}")

    return "\n\n".join(parts)

def rag_answer(query:str, records, index, top_k: int=1):
    #Create the vector search for our user query and document records
    results =search(query, records, index, top_k=top_k)
    context= format_context(results)

    system_prompt = (
        "Answer the user's question using only the information in the "
        "provided context. Cite the sources using the bracketed "
        "numbers exactly as they appear in the context (for example [3] not[1] "
        "for the first item shown). If the context doesn't contain enough "
        "information to answer, say so explicitly rather than guessing."        
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    answer = response.choices[0].message.content
    return {
        "query": query,
        "answer": answer,
        "sources": results,
        "cited_sources": check_citations(answer, results)
    }
    

def check_citations(answer: str, results):
    valid_ids = {result["chunk_id"] for result in results}
    cited_ids = {int(n) for n in re.findall(r"\[(\d+)]", answer)}
    return {
        "cited": cited_ids,
        "valid": valid_ids,
        "ungrounded": cited_ids - valid_ids,
        "unused": valid_ids - cited_ids
    }

def keyword_score(answer: str, must_include: list[str]) -> dict:
    answer_lower = answer.lower()
    hits = [term for term in must_include if term.lower() in answer_lower]

    return {
        "hits": hits,
        "misses": [t for t in must_include if t not in hits],
        "coverage": len(hits) / len(must_include)
    }

def run_eval(eval_set, records, index, top_k=1):
    results = []

    for case in eval_set:
        outcome = rag_answer(case["query"], records, index, top_k=top_k)
        kw_score = keyword_score(outcome["answer"], case["must_include"])
        judge_score = llm_judge(case["query"], outcome["answer"], case["must_include"])
        results.append({
            "query": case["query"],
            "coverage": kw_score["coverage"],
            "judge_verdict": judge_score["verdict"],
            "fact_checks": judge_score["fact_checks"],
            "misses": kw_score["misses"],
            "ungrounded": outcome["cited_sources"]["ungrounded"],
            "answer": outcome["answer"]
        })

    return results

def llm_judge(query: str, answer: str, must_include: list[str] ) -> dict:
    facts_block = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(must_include))
    judge_prompt = f"""You are evaluating whether an answer correctly and completely addresses a question, given a list of facts a complete answer should cover
Question: {query}

Answer to evaluate:
{answer}
For EACH fact listed, decide if the answer addresses it. You must quote the exact phrase from the answer that supports "found: true". If you cannot find supporting text, mark it "found: false" and write "NOT FOUND" as the evidence -- do not paraphrase or infer.

Facts:
{facts_block}

Respond ONLY with JSON structure:
{{"fact_checks": [{{"fact": "<fact text>", "found": true/false, "evidence": "<exact quote or NOT FOUND>"}}], "verdict": "pass/partial/fail"}}"""

# Score the answer on:
# - "complete": does it cover ALL the facts, or does it address some while explicitly saying it cannot address others? A partial answer that explicitly flags its own gap is NOT complete, even if it's honest about it.
# - "correct": are the facts it DOES state accurate, and does it avoid asserting things not supported by an answer's own content?
# - "verdict": one of "pass", "partial", "fail"
#   - "pass": addresses all facts accurately
#   - "partial": addresses some facts accurately but is missing others (including cases where it explicitly says information is unavailable)
#   - "fail": inaccurate, or fails to address the question's core intent
# Respond with ONLY a JSON object, no other text:
# {{"complete": true/false, "correct": true/false, "verdict": "pass/partial/fail", "reasoning": "one sentence explaining the verdict"}}"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "fact_checks": [], 
            "verdict": "error", 
            #"reasoning": f"unparseable judge output: {raw}"
            }

#Chunk the runbook records
#No overlap has hard boundary of 60 tokens
no_overlap_runbook = chunk_by_paragraphs(document_text_runbook, max_tokens=60)
#With overlap has boundary of 60 tokens and overlap boundary of 55 tokens
with_overlap_runbook = chunk_by_paragraphs_with_overlap(document_text_runbook, max_tokens=60, overlap_tokens=55)

no_overlap_runbook_records = embed_records(build_records(no_overlap_runbook, document_id="INTEGRATION_RUNBOOK", document_title="Invoice Integration Runbook"))

with_overlap_runbook_records = embed_records(build_records(with_overlap_runbook, document_id="INTEGRATION_RUNBOOK", document_title="Invoice Integration Runbook"))

no_overlap_runbook_index = build_index(no_overlap_runbook_records)
with_overlap_runbook_index = build_index(with_overlap_runbook_records)

no_overlap_index = build_index(no_overlap_runbook_records)
with_overlap_index = build_index(with_overlap_runbook_records)
print("No overlap chunk")
print(no_overlap_runbook_records[3]["text"])
print("---")
print("With overlap chunk")
print(with_overlap_runbook_records[3]["text"])

for _ in range(4):
    outcome = rag_answer(
        "How many times does the integration retry a failing invoice, and what happens once that limit is reached?",
        with_overlap_runbook_records, with_overlap_runbook_index, top_k=1
    )
    print(outcome["answer"])
    print("---")
#run the RAG pipeline on the two cunking strategy versions and 
#get the answer, LLM judge's ruling, and keyword coverage
# for label, records, index in [
#     ("NO OVERLAP", no_overlap_runbook_records, no_overlap_runbook_index),
#     ("WITH OVERLAP", no_overlap_runbook_records, no_overlap_runbook_index),
# ]:
#     print(f"\n==== {label} ====")
#     for r in run_eval(eval_set_runbook, records, index, top_k=1):
#         print(f" {r['query']}")
#         print(f"LLM Answer: {r['answer']}")
#         print(f"coverage={r['coverage']:.2f} | verdict={r['judge_verdict']}")
        

# distractor_query = "What is the maximum allowed payload size in kilobytes for an invoice request?"
# print("\nRunning distractor query\n")
# #Run the RAG pipeline for a question with no answer in the eval set
# for label, records, index in [
#     ("NO OVERLAP", no_overlap_runbook_records, no_overlap_runbook_index),
#     ("WITH OVERLAP", with_overlap_runbook_records, with_overlap_runbook_index),
# ]:
#     outcome = rag_answer(distractor_query, records, index, top_k=1)
#     print(f"\n[{label}]\n{outcome['answer']}")

