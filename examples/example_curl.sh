#!/bin/bash
#
# Example of querying the streaming SSE endpoint using POST.
#

QUERY="Explain the 'process_array_core' function in CoreService.m"
TOP_K=3

curl -N -X POST "http://127.0.0.1:8000/v1/query/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d @- << EOF
{
  "query": "$QUERY",
  "top_k": $TOP_K
}
EOF
