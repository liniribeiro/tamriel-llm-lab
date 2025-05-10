import requests

def download_data():
    url = "https://huggingface.co/datasets/billxbf/sotu2023-qa/resolve/main/state_of_the_union.txt"
    res = requests.get(url)
    with open("data/The Alduin_Akatosh Dichotomy.txt", "w") as f:
        f.write(res.text)