from retrieval import hybrid_retrieval
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """Sen e-gov.az portalının rəsmi chatbotusan.
- Yalnız e-gov.az portalı ilə bağlı suallara cavab ver.
- Cavabları həmişə təqdim edilən FAQ məlumatlarına əsasla.
- FAQ-da cavabı olmayan e-gov sualları üçün: "Bağışlayın, bu mövzu barədə məlumatım yoxdur. Ətraflı məlumat üçün e-gov.az saytına və ya 108 çağrı mərkəzinə müraciət edə bilərsiniz."
- E-gov ilə əlaqəsi olmayan suallara: "Bu sual e-gov.az xidmətləri ilə bağlı deyil. Mən yalnız e-gov.az portalı üzrə köməklik göstərə bilərəm. Portalla bağlı başqa sualınız varsa, məmnuniyyətlə cavablayaram."
- Cavablar qısa, aydın və praktik olsun.
- Söhbət tarixinə əsasən istifadəçinin əvvəlki suallarını və konteksti nəzərə al.
- Bütün cavabları maksimum dərəcədə nəzakətli, peşəkar və istifadəçiyə kömək etməyə yönəlmiş üslubda təqdim et.
"""

def generate_answer_with_context(user_query: str, chat_history=None, top_k: int = 4) -> str:
    messages = [{"role": "system", "content": system_prompt}]

    last_msgs = (chat_history[-5:] if chat_history else [])
    for h in last_msgs:
        messages.append({"role": "user", "content": h['question']})
        messages.append({"role": "assistant", "content": h['answer']})

    faq_results = hybrid_retrieval(user_query, top_k=top_k)

    seen = set()
    context_chunks = []
    for c in faq_results:
        key = c["question"] + c["answer"]
        if key not in seen:
            context_chunks.append(c)
            seen.add(key)

    faq_context = "\n\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in context_chunks])
    faq_section = f"\n\nFAQ-dan əlavə kontekst:\n{faq_context}" if faq_context else ""

    messages.append({"role": "user", "content": f"{user_query}{faq_section}"})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content