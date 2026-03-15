import requests
from bs4 import BeautifulSoup
import pandas as pd

urls = {
    "egov": "https://e-gov.az/az/home/faq?name=egov",
    "asan_login": "https://e-gov.az/az/home/faq?name=asan_login",
    "mygov": "https://e-gov.az/az/home/faq?name=mygov"
}

rows = []

for service, url in urls.items():
    print(f"Scraping {service}...")

    r = requests.get(url)
    r.encoding = 'utf-8'  
    soup = BeautifulSoup(r.text, "html.parser")

    faq_items = soup.select("details.faq-item")

    for item in faq_items:
        question_elem = item.select_one(".faq-title")
        answer_elem = item.select_one(".faq-content")

        if question_elem and answer_elem:
            question = question_elem.get_text(separator=" ", strip=True)
            answer = answer_elem.get_text(separator=" ", strip=True)

            rows.append({
                "service": service,
                "question": question,
                "answer": answer
            })


df = pd.DataFrame(rows, columns=["service", "question", "answer"])

df.to_excel("data/faq_clean.xlsx", index=False, engine='openpyxl')

print("Scraping finished!")
print("Total FAQs:", len(df))