import requests

endpoint = '' # Set endpoint https://<mlendpoint>.cloudera.site/model
accessKey = 'mpn136d0760wec8orok6scpwj69xzh6r' # Set accesss key

r = requests.post(f"{endpoint}?accessKey={accessKey}", 
                  data='{"request":{"inputs":[[2000,15,75,40]]}}', headers={'Content-Type': 'application/json'})


r.status_code
r.json()






