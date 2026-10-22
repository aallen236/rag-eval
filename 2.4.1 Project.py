from json import tool
from multiprocessing import Value
from turtle import pen
from unittest import result

from pydantic import BaseModel
from openai import OpenAI
from datetime import date
import json
import os
from rapidfuzz import process, fuzz
import time

#api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI()

#Tool Dictionary
tools = [
    {
        "type": "function",
        "name": "get_customer_balance",
        "description": "Get the outstanding balance for a customer",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer ID."
                }
            },
            "required": ["customer_id"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "get_open_orders",
        "description": "Get the currently open sales orders for a customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer's unique ID."
                }
            },
            "required": ["customer_id"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "get_invoice",
        "description": "Get the invoice details for a specific invoice number.",
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_number": {
                    "type": "string",
                    "description": "The invoice number."
                }
            }
        },
        "required": ["invoice_number"],
        "additionalProperties": False
    },
    {
        "type": "function",
        "name": "get_customer_id",
        "description": "Get the id for a customer",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "The customer's name"
                }
            }
        },
        "required": ["customer_name"],
        "additionalProperties": False
    },
    {
        "type": "function",
        "name": "create_customer_note",
        "description": "Create a note on a customer record.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string"
                },
                "note": {
                    "type": "string"
                }
            },
            "required": ["customer_id", "note"],
            "additionalProperties": False
        }        
    }
]

READ_TOOLS = {
    "get_customer_id",
    "get_customer_balance",
    "get_open_orders",
    "get_invoice",
}

WRITE_TOOLS = {
    "create_customer_note",
}


#Validation status helpers

def tool_success(data):
    return {
            "status": "success",
            "error": None,
            "data": data
    }

def tool_not_found(message):
    return {
            "status": "not_found",
            "error": {
                "message": message
            },
            "data": None        
    }

def tool_validation_error(errors, data=None):
    return{
            "status": "validation_error",
            "data": data,
            "error": {
                    "messages": errors
            },
    }

def tool_error(error_type, message):
        return {
            "status": "error",
            "data": None,
            "error": {
                "type": error_type,
                "message": message
            }
        }

#Tool functions

def get_customer_balance(customer_id: str):
    if customer_id == "9999":
        raise ConnectionError("Customer balance service is unreachable")
    if customer_id == "8888":
        raise TimeoutError("Simulated database timeout")
    balances = {
        "1001": 4281.17,
        "1002": 995.42,
        "1003": 18750.00,
        "1499": 12000.00
    }

    return {
        "customer_id": customer_id,
        "balance": balances.get(customer_id)
    }

def get_open_orders(customer_id: str):
    orders = {
        "1001": [
            {"order_id": "SO-101", "amount": 1200.00},
            {"order_id": "SO-102", "amount": 850.00},
        ],
        "1002": [],
        "1003": [
            {"order_id": "SO-200", "amount": 6400.00}
        ],        
    }
    return {
        "customer_id": customer_id,
        "orders": orders.get(customer_id)
    }
def get_invoice(invoice_number: str):
    invoices = {
        "INV-84729": {
            "invoice_number": "INV-84729",
            "customer_id": "1002",
            "total": 4821.17,
        },
        "INV-90001": {
            "invoice_number": "INV-90001",
            "customer_id": "1003",
            "total": 12750.00,
        },
        "INV-BAD": {
            "invoice_number": "INV-BAD",
            "customer_id": "1003",
            "total": -99999.00,
            "due_date": "1515-06-15"            
        }        
    }
    return invoices.get(invoice_number)

def validate_invoice(invoice: dict):
    """Helper function to ensure invoice integrity"""
    errors =[]
    if invoice is None:
        return tool_not_found("Invoice does not exist")
    if invoice["total"] < 0:
        return errors.append("Invoice total cannot be negative")
    
    try:
        due_date = date.fromisoformat(invoice["due_date"])

        if due_date.year < 2000:
            errors.append(":Invoice due date is implausibly old")
    except ValueError:
        errors.append("Invoice due date is not a valid ISO date")

    if errors:
        return tool_validation_error(
            errors=errors, 
            data=invoice
        )
    
    return tool_success(invoice)

def normalize_name(name: str) -> str:
    return(
        name.lower()
        .replace(".", "")
        .replace(",","")
        .replace("corporation", "corp")
        .replace("limited liability company", "llc")
        .strip()
    )


def get_customer_id(customer_name: str) -> str:

    customers = {
        "Acme corp":"1001",
        "Acme Industrial": "1005",
        "Widgets Inc.":"1002",
        "Corporate Death Machine": 1003,
        "Globex LLC": "1499",
        "Prestonwood Country Club": "1777"
    }

    normalized_query = normalize_name(customer_name)

    matches = []

    
    normalized_customers = {
        normalize_name(name): {
            "customer_id": customer_id,
            "customer_name": name
        }
        for name, customer_id in customers.items()
    }

    if normalized_query in normalized_customers:
            return tool_success(
                normalized_customers[normalized_query]
            )
    candidates = process.extract(
        normalized_query,
        normalized_customers.keys(),
        scorer=fuzz.WRatio,
        limit=3
    )

    strong_matches = [
        {
            **normalized_customers[match_name],
            "score": score
        }
        for match_name, score, _ in candidates
        if score >= 75
    ]
    if len(strong_matches) == 1:
        return tool_success(strong_matches[0])

    if len(strong_matches) > 1:
        return {
            "status" : "ambiguous",
            "data": {"matches": strong_matches},
            "error": {
                "message": "Multiple plausible customer matches found"
            }
        }
    return tool_not_found(
        f"No customer matched '{customer_name}'"
    )
    # if len(matches) == 1:
    #     return {
    #         "status": "success",
    #         "data": matches[0],
    #         "error": None
    #     }
    # if len(matches) == 0:
    #     return {
    #         "status": "not_found",
    #         "data": None,
    #         "error": {
    #             "message": f"No customner matched '{customer_name}'"
    #         }
    #     }
    
    # return {
    #     "status": "ambiguous",
    #     "data" : {
    #         "matches": matches
    #     },
    #     "error": {
    #         "message": "Multiple customers matched the supplied name"
    #     }
    # }


def get_validated_invoice(invoice_number: str):
    """Run an invoice through validation check"""
    invoice = get_invoice(invoice_number)

    return validate_invoice(invoice)



def create_customer_note(customer_id: str, note: str):
    return {
        "status": "success",
        "data": {
            "customer_id": customer_id,
            "note": note
        },
        "error": None
    }

#Response routing
tool_functions = {
    "get_customer_balance": get_customer_balance,
    "get_open_orders": get_open_orders,
    "get_invoice": get_invoice,
    "get_customer_id": get_customer_id,
    "create_customer_note": create_customer_note,
}
def resume_after_approval(pending_request: dict):

    trace = pending_request["trace"]
    tool_name = pending_request["tool_name"]
    args = pending_request["arguments"]
    call_id = pending_request["call_id"]
    response_id = pending_request["response_id"]
    trace.append({
        "event": "approval_granted",
        "tool_name": pending_request["tool_name"],
        "arguments": pending_request["arguments"]
    })

    tool_name = pending_request["tool_name"]
    args = pending_request["arguments"]

    if tool_name not in tool_functions:
        raise ValueError(f"Unknown tool requested: {tool_name}")

    try:
        result = tool_functions[tool_name](**args)
    except Exception as e:
        result ={
            tool_error(
                error_type=type(e).__name__,
                message=str(e)
            )
        }
    trace.append({
        "event": "tool_result",
        "tool_name": tool_name,
        "result": result
    })
    response = client.responses.create(
        model='gpt-5.6',
        tools=tools,
        previous_response_id=response_id,
        input=[
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result)
            }
        ]
    )
    trace.append({
        "event": "final_response",
        "output_text": response.output_text
    })

    return {
        "response":response,
        "trace": trace
        }


def get_requested_tools(trace):
    return [
        event["tool_name"]
        for event in trace
        if event["event"] == "tool_requested"
    ]
tests = [
    {
        "name": "balance lookup",
        "prompt": "What does customer 1003 owe?",
        "expected_tools": ["get_customer_balance"]
    },
    {
        "name": "open orders lookup",
        "prompt": "Does customer 1003 have open orders?",
        "expected_tools": ["get_open_orders"]
    },
    {
        "name": "parallel lookup",
        "prompt": "What does customer 1003 owe and do they have open orders?",
        "expected_tools": [
            "get_customer_balance",
            "get_open_orders"
        ]
    },
    {
        "name": "customer resolution",
        "prompt": "What does Prestonwood owe?",
        "expected_tools": [
            "get_customer_id",
            "get_customer_balance"
        ]
    },
    {
        "name": "no tool needed",
        "prompt": "Tell me a joke.",
        "expected_tools": []
    },
    {
        "name": "ambiguous customer",
        "prompt": "What does Acme owe?",
        "expected_tools": ["get_customer_id"]
    }    
]


#Main execution loop
import time
import json

def run_agent(user_prompt: str, max_steps: int = 5):

    trace = []
    start_time = time.perf_counter()

    response = client.responses.create(
        model="gpt-5.6",
        tools=tools,
        input=user_prompt
    )

    trace.append({
        "event": "initial_response",
        "response_id": response.id,
        "usage": response.usage.model_dump() if response.usage else None
    })

    for step in range(max_steps):

        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            elapsed = time.perf_counter() - start_time

            trace.append({
                "event": "final_response",
                "output_text": response.output_text,
                "elapsed_seconds": elapsed,
                "usage": response.usage.model_dump() if response.usage else None
            })

            return {
                "response": response,
                "trace": trace
            }

        tool_outputs = []

        for call in tool_calls:

            args = json.loads(call.arguments)

            trace.append({
                "event": "tool_requested",
                "tool_name": call.name,
                "arguments": args,
                "call_id": call.call_id
            })

            if call.name not in tool_functions:
                raise ValueError(
                    f"Unknown tool requested: {call.name}"
                )

            if call.name in WRITE_TOOLS:
                trace.append({
                    "event": "approval_required",
                    "tool_name": call.name,
                    "arguments": args
                })

                return {
                    "status": "approval_required",
                    "tool_name": call.name,
                    "arguments": args,
                    "call_id": call.call_id,
                    "response_id": response.id,
                    "trace": trace
                }

            try:
                result = tool_functions[call.name](**args)

            except Exception as e:
                result = tool_error(
                    error_type=type(e).__name__,
                    message=str(e)
                )

            trace.append({
                "event": "tool_result",
                "tool_name": call.name,
                "result": result
            })

            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result)
            })

        response = client.responses.create(
            model="gpt-5.6",
            tools=tools,
            previous_response_id=response.id,
            input=tool_outputs
        )

    raise RuntimeError(
        f"Agent exceeded maximum of {max_steps} tool rounds."
    )

for test in tests:
    result = run_agent(test["prompt"])

    actual_tools = get_requested_tools(
        result["trace"]
    )

    passed = actual_tools == test["expected_tools"]

    print(test["name"])
    print("Expected:", test["expected_tools"])
    print("Actual:  ", actual_tools)
    print("PASS" if passed else "FAIL")
    print()
# Output = run_agent("Please add a note to prestonwood saying 'follow up next tuesday'")
# if isinstance(Output, dict) and Output.get("status") == "approval_required":
#     print("approval required:")
#     print(Output.get("tool_name"))
# else:
#     print(Output.output_text)

# result = run_agent(
#     "What does customer 1003 owe, and do they have any open orders?"
# )

# print(result["response"].output_text)
# for event in result["trace"]:
#     #print(event)
#     print(json.dumps(result["trace"], indent=2, default=str))
# pending = run_agent("Add a note to customer 1003 saying call next week")
# print(pending["status"])
# print(pending["tool_name"])
# print(pending["arguments"])

# final = resume_after_approval(pending)

# print(final.output_text)
            

