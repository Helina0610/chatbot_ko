import redis
import hashlib
import json


from fastapi import FastAPI, Request
app = FastAPI()

#@app.post("/chat")
#async def chat(request: Request):
#    body = await request.json()
#    prompt = body["prompt"]

#    result = get_ai_response(prompt)
#    return result

# Redis 연결
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# 캐시 key 생성 함수
def generate_cache_key(prompt: str) -> str:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    return f"prompt_cache:{prompt_hash}"

# 캐시 저장
def save_to_cache(prompt: str, response: dict, ttl_seconds: int = 600):
    key = generate_cache_key(prompt)
    r.setex(key, ttl_seconds, json.dumps(response))

# 캐시 조회
def load_from_cache(prompt: str):
    key = generate_cache_key(prompt)
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    return None


#def get_ai_response(prompt: str) -> dict:
#    # 1. 캐시에서 조회
#    cached_response = load_from_cache(prompt)
#    if cached_response:
#        print(f"✅ 캐시에서 가져옴 {cached_response}")
#        return cached_response

#    # 2. LLM 호출 (예시용)
#    #response = call_llm(prompt)

#    # 3. Redis에 저장
#    save_to_cache(prompt, response)

#    #return response


#save_to_cache("장미의 꽃말은 뭐야?" , {"content": "장미는 색깔별로 다양한 꽃말을 가지고 있습니다. 몇 가지 대표적인 꽃말은 다음과 같습니다"})
print(load_from_cache("장미의 꽃말은 뭐야?"))