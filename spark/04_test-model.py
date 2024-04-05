import requests

endpoint = '' # Set endpoint https://<mlendpoint>.cloudera.site/model
accessKey = '' # Set accesss key

r = requests.post(f"{endpoint}?accessKey={accessKey}", 
                  data='{"request":{"param":"value"}}', headers={'Content-Type': 'application/json'})


r.status_code
r.json()





