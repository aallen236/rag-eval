from collections import Counter
from openai import OpenAI
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
    # --- Controls: everything lives in ONE paragraph ---
    {
        "id": "ctrl_validation", "type": "control", "top_k": 1,
        "query": "What must a source record contain before its invoice payload is sent to the external API?",
        "must_include": ["customer identifier", "invoice number", "line-item total"],
    },
    {
        "id": "ctrl_exhausted", "type": "control", "top_k": 1,
        "query": "What should happen when an invoice keeps failing due to timeouts and all retries have been exhausted?",
        "must_include": ["exhausted", "manual review", "on-call"],
    },
    {
        "id": "ctrl_attachments", "type": "control", "top_k": 1,
        "query": "How should large attachments on an invoice request be handled?",
        "must_include": ["attachments", "document-upload", "retried"],
    },

    # --- Boundary: one fact in each of two ADJACENT paragraphs ---
    {
        "id": "bnd_validation_size", "type": "boundary", "top_k": 1,
        "query": "What does an invoice request need to satisfy to be accepted: what must be on the record, and what happens if the request is too large?",
        "must_include": ["customer identifier", "rejected", "document-upload"],
    },
    {
        "id": "bnd_size_retry", "type": "boundary", "top_k": 1,
        "query": "How does the integration handle a request that fails because it is too large versus one that fails because of a timeout?",
        "must_include": ["document-upload", "exponential backoff", "four"],
    },
    {
        "id": "bnd_retry_escalation", "type": "boundary", "top_k": 1,
        "query": "How many times does the integration retry a failing invoice, and what happens once that limit is reached?",
        "must_include": ["four", "manual review", "on-call"],
    },
    {
        "id": "bnd_escalation_dupes", "type": "boundary", "top_k": 1,
        "query": "What safeguards keep a failing invoice from being either lost or duplicated?",
        "must_include": ["manual review", "idempotent", "unique"],
    },
    {
        "id": "bnd_dupes_logging", "type": "boundary", "top_k": 1,
        "query": "How does the integration prevent duplicate invoices, and how is activity recorded so problems can be investigated later?",
        "must_include": ["idempotent", "unique", "logged"],
    },

    # --- Multi-hop: three NON-adjacent paragraphs, same question at two top_k values ---
    {
        "id": "hop_k1", "type": "multihop", "top_k": 1,
        "query": "Describe the full path of an invoice that keeps failing: what is checked before it is sent, how many retries are allowed, and what happens at the end?",
        "must_include": ["customer identifier", "four", "manual review"],
    },
    {
        "id": "hop_k3", "type": "multihop", "top_k": 3,
        "query": "Describe the full path of an invoice that keeps failing: what is checked before it is sent, how many retries are allowed, and what happens at the end?",
        "must_include": ["customer identifier", "four", "manual review"],
    },
]
def build_records(chunk_texts, document_id="AR_POLICY",document_title="Accounts Receivable Policy"):
    """Use the referenced document text to build a dictionary to use for encoding
    Chunk texts are the result of the chunking function. Paragraph, fixed window, etc."""
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
    """Use the embeddings model to create embedding arrays for the given records. Takes all records from the source document
    and creates a numeric representation of their content"""
    texts = [r["text"] for r in records]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    for r, item in zip(records, response.data):
        r["embedding"] = item.embedding
    return records

def build_index(records):
    """Uses the embedded records to create indexes. First create a normalized (values > 0 && < 1 ) matrix, then
    use inner product to """
    matrix = np.array([r["embedding"] for r in records], dtype="float32")
    faiss.normalize_L2(matrix)
    dimension = matrix.shape[1]
    idx = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    idx.add(matrix)
    return idx

def search(query, records, index, top_k=2):
    """Create embeddings of user query and pass to search function, retrieving top_k results
    and return an object with score, chunk ID and the original document text"""
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
    """Counts the token length of a given text encoding"""
    return len(encoding.encode(text))

def chunk_by_paragraphs(
    text: str,
    max_tokens: int = 300
):
    """Takes given text and splits by text blocks, applying token-length limits and chunking accordingly"""
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
    """Splits given text by discrete sections and adds neighboring text up to overlap_tokens 
    length to each chunk"""
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
                #write the current_chunk to the list of chunks
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
    """Helper function used with chunking. Applies max_token and overlap rules to generate a list of document chunks"""
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
    """Retrieve the search results of the user's query against document chunks, give that to the competions API alongside the 
    system prompt and user's prompt, and return the original query, LLM response, sources used, and list of cited sources, and result
    of citation check"""
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
    """Use the sources in answer to check validity of source indicies"""
    valid_ids = {result["chunk_id"] for result in results}
    cited_ids = {int(n) for n in re.findall(r"\[(\d+)]", answer)}
    return {
        "cited": cited_ids,
        "valid": valid_ids,
        "ungrounded": cited_ids - valid_ids,
        "unused": valid_ids - cited_ids
    }

def keyword_score(answer: str, must_include: list[str]) -> dict:
    """Validation function to check for presence of required keywords in each answer"""
    answer_lower = answer.lower()
    hits = [term for term in must_include if term.lower() in answer_lower]

    return {
        "hits": hits,
        "misses": [t for t in must_include if t not in hits],
        "coverage": len(hits) / len(must_include)
    }

def run_eval(eval_set, records, index, top_k=1):
    """Main evaluation function to check keyword score, judge score, rag answer"""
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
    """Uses LLM to check whether a rag answer addresses all required facts based on  must_include list."""
    facts_block = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(must_include))
    #Judge prompt asks llm to return a structured JSON object with its true/false verdict, the example of a requred fact, and evidence for its judgement 
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
        temperature=0,
        response_format={"type": "json_object"}
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

def single_chunk_covers(records, must_include):
    """Checks whether a chunk includes all required keywords"""
    for r in records:
        text = r["text"].lower()
        if all(term.lower() in text for term in must_include):
            return True
    return False

def retrieval_covers(case, records, index):
    results = search(case["query"], records, index, top_k=case["top_k"])
    joined = " ".join(r["text"] for r in results).lower()
    return all(t.lower() in joined for t in case["must_include"])

flagged_claims = []

def run_eval_suite(eval_set, records, index, n=3):
    """Main eval function for eval_set of records. This function is the more comprehensive version of run_eval and 
    returns the case type, count of verdicts, average keyword coverages for each case type, and IDs of retrieved chunks"""
    rows = []
    
    for case in eval_set:
        verdicts, coverages, f_scores, f_labels = [], [], [], Counter()
        abstained = 0
        for _ in range(n):
            outcome = rag_answer(case["query"], records, index, top_k=case["top_k"])
            kw = keyword_score(outcome["answer"], case["must_include"])
            judge = llm_judge(case["query"], outcome["answer"], case["must_include"])
            judge_verdict = judge.get("verdict","error")
            verdicts.append(judge_verdict)
            coverages.append(kw["coverage"])
            faithfulness_test = faithfulness(outcome["answer"], outcome["sources"], case["query"])
            if faithfulness_test["score"] is not None:
                f_scores.append(faithfulness_test["score"])
            for c in faithfulness_test["checks"]:
                f_labels[c.get("label")] += 1
            score = faithfulness_test["score"]

            #print(f"Faithfulness ratio: {score}")
            for c in faithfulness_test["checks"]:
                if c.get('label') != "supported":
                    flagged_claims.append({
                        "case": case["id"],
                        "label": c.get("label"),
                        "judge_label": c.get("judge_label"),
                        "claim": c.get("claim"),
                        "evidence": c.get("evidence"),
                    })
            abstained += faithfulness_test["abstained"]
                    

        rows.append({
            "id": case["id"],
            "type": case["type"],
            "verdicts": Counter(verdicts),
            "avg_coverage": sum(coverages) / n,
            "retrieved": [s["chunk_id"] for s in outcome["sources"]],
            "faithfulness": sum(f_scores) / len(f_scores) if f_scores else None,
            "claim_labels": f_labels,
            "abstained": abstained
        })
    return rows

def print_suite(label, rows):
    """Utility to format printing of results"""
    print(f"\n==== {label} ====")

    for r in rows:
        f = r["faithfulness"]
        f_str = f"{f:.2f}" if f is not None else "n/a"
        v = ", ".join(f"{k}:{c}" for k, c in r["verdicts"].items())
        print(f"{r['id']:<22} {r['type']:<10} cov={r['avg_coverage']:.2f} {v:<22} chunks={str(r['retrieved']):<10} faithfulness scores: {f_str}, claim labels: {dict(r['claim_labels'])}, abstained = {r['abstained']}")

def normalize(text:str) -> str:
    text = re.sub(r"\[\d+\]", " ", text)
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[^\w\s]", " ", text.lower()) 
    return " ".join(text.split())

def extract_claims(answer: str):
    """Use LLM to identify and isolate factual claims in the RAG answer"""
    prompt = f"""Break the answer below into atomic factual claims. Each claim must be a single, self-contained statement that can be checked on its own. Ignore citation markers like [3] and statements about what the context does or doesn't contain.

    Answer:
    {answer}

Respond ONLY with JSON: {{"claims": ["...", "..."]}}"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
        )
    try:
        return json.loads(response.choices[0].message.content)["claims"]
    except (json.JSONDecodeError, KeyError):
        return None

def check_claim_support(claim: str, context: str, query: str) -> dict:
    """Use LLM to analyze context (the quoted document section) against claim (the RAG answer) and return
    supported, unsupported, contradicted, or unverified_evidence"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content": build_support_prompt(claim, context, query)}],
        temperature=0,
        response_format={"type": "json_object"},

    )
    try:
        result = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"label":"error", "evidence": "NONE"}

    result["judge_label"] = result.get("label")

    if result.get("label") == "supported":
        raw_ev = str(result.get("evidence", ""))
        ev = normalize(raw_ev)
        # print(f"Supported Evidence: {raw_ev}")
        # print(f"Normalized Evidence: {ev}")
        if not ev or ev not in normalize(context):
            result["label"] = "unverified_evidence"
    return result
META_PREFIXES = ("the context", "the provided", "the answer", "the author")
ABSTAIN_RE = re.compile(r"\b(does not|do not|cannot|can't|not (?:addressed|specified|provided|stated))\b", re.I)
REF_RE = re.compile(r"\b(context|answer|information)\b", re.I)

def is_meta(claim: str) -> bool:
    c = claim.strip().lower()
    return c.startswith(META_PREFIXES) or bool(ABSTAIN_RE.search(c) and REF_RE.search(c))


def faithfulness(answer: str, results, query: str) -> dict:
    context = format_context(results)
    claims = extract_claims(answer)
    if not claims:
        return {"score": None, "checks": [], "abstained": 0}
    meta = [c for c in claims if is_meta(c)]
       substantive = [c for c in claims if not is_meta(c)]
    if not substantive:
        return {"score": None, "checks": [], "abstained": len(meta)}
    checks = [{"claim": c, **check_claim_support(c, context, query)} for c in substantive]
    supported = sum(1 for c in checks if c["label"] == "supported")
    return {"score": supported / len(checks), "checks": checks, "abstained": len(meta)}

def build_support_prompt(claim, context, query):    
    prompt = f"""Context:
{context}

Claim:
{claim}

Question:
{query}

Is the claim directly supported by the context?
- "supported": the context states it, or it is a trivial restatement.
- "unsupported": the claim adds a detail, number, or condition the context does not state.
- "contradicted": the context says something different.
- "restates_question": the claim does not appear directly in the context but is implied by the user's question.
For "supported", quote the exact span from the context as evidence. Otherwise use "NONE". For "restates_question" include the restated claim.

Respond ONLY with JSON: {{"label": "supported/unsupported/contradicted/restates_question", "evidence": "<exact span or NONE>"}}"""
    return prompt

def debug_evidence(claim, context, query, n=10):
    for i in range(n):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": build_support_prompt(claim, context, query)}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        ev = normalize(str(result.get("evidence", "")))
        if not ev or ev not in normalize(context):
            result["label"] = "unverified_evidence"
        print(f"run {i}: judge label = {result.get('label')}")
        print(f"    evidence: {ev!r}")
        print(f"    in context (raw):   {ev in context}")
        print(f"    in context (normalized): {normalize(ev) in normalize(context)}")
        print(f"    in query (normalized): {normalize(ev) in normalize(query)}")
        
claim = 'Retries can be exhausted when an invoice fails.'

query = "What should happen when an invoice keeps failing due to timeouts and all retries have been exhausted?"


def flip_rate(claim, context, query, n=10):
    labels = [check_claim_support(claim, context, query)["label"] for _ in range(n)]
    return Counter(labels)

def run_debug(records, index):
    outcome = rag_answer(query, records, index, top_k=1)
    context = format_context(outcome["sources"])
    print(flip_rate(claim, context, query))
    for _ in range(3):
        r = check_claim_support(claim, context, query)
        print("judge_label:", r.get("judge_label"), "| label:", r.get("label"))
        print("evidence:   ", repr(r.get("evidence")))
    debug_evidence(claim, context, query, n=3)

def make_arm(chunks):
    records = embed_records(build_records(
        chunks,
        document_id="INTEGRATION_RUNBOOK",
        document_title="Invoice Integration Runbook"
    ))
    return records, build_index(records)
if __name__ == "__main__":

    RUN_PREFLIGHT = False
    RUN_SUITE = True
    RUN_DEBUG = False

    no_overlap_runbook = chunk_by_paragraphs(document_text_runbook, max_tokens=60)
    with_overlap_runbook = chunk_by_paragraphs_with_overlap(document_text_runbook, max_tokens=60, overlap_tokens=55)
    size_matched = chunk_by_paragraphs(document_text_runbook, max_tokens=95)

    print("===== TOKEN COUNTS =====")
    print("no overlap: ", [count_tokens(c) for c in no_overlap_runbook])
    print("with overlap: ", [count_tokens(c) for c in with_overlap_runbook])
    print("size_matched:", [count_tokens(c) for c in size_matched])    
    arms = {
        "no_overlap": make_arm(no_overlap_runbook),
        "with_overlap": make_arm(with_overlap_runbook),
        "size_matched": make_arm(size_matched),
    }

    if RUN_PREFLIGHT:
        print("\n== single_chunk_covers: could ANY one chunk answer this? ==")
        print(f"{'case':<24}" + "".join(f"{name:<14}" for name in arms))
        for case in eval_set_runbook:
            row = f"{case['id']:<24}"
            for records, _ in arms.values():
                row += f"{str(single_chunk_covers(records, case['must_include'])):<14}"
            print(row)

        print("\n=== retrieval_covers: did the retrieved chunks contain everything? ==")
        print(f"{'case':<24}" + "".join(f"{name:<14}" for name in arms))
        for case in eval_set_runbook:
            row = f"{case['id']:<24}"
            for records, index in arms.values():
                row += f"{str(retrieval_covers(case, records, index)):<14}"
            print(row)
    if RUN_SUITE:
        for label, (records, index) in arms.items():
            print_suite(label, run_eval_suite(eval_set_runbook, records, index, n=3))
            print("\n== flagged claims: label breakdown ==")
            print(Counter((f["label"], f["judge_label"]) for f in flagged_claims))

            print("\n== unique flagged claims (excluding restates_question) ==")
            seen = set()
            for f in flagged_claims:
                key = (f["label"], f["claim"])
                if f["label"] != "restates_question" and key not in seen:
                    seen.add(key)
                    print(f"[{f['label']}/{f['judge_label']}] {f['case']}: {f['claim']}")
    if RUN_DEBUG:
        run_debug(*arms["no_overlap"])