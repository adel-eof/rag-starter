import sseclient, requests

url = "http://127.0.0.1:8000/v1/query/stream?q=show+the+processValue+function%3F&top_k=1"
with requests.get(url, stream=True) as r:
    client = sseclient.SSEClient(r)
    for event in client.events():
        print(event.data)
