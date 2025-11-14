"""
Bedrock 접근 권한 문제 해결 및 OpenSearch 통합
"""
import boto3
import json
from config import Config

def check_bedrock_access():
    """Bedrock 모델 접근 권한 확인"""
    try:
        bedrock_client = boto3.client('bedrock-runtime', region_name=Config.BEDROCK_REGION)
        
        # 간단한 텍스트 요청으로 테스트
        response = bedrock_client.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0',  # 더 접근하기 쉬운 모델
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}]
            })
        )
        print("✅ Bedrock 접근 가능")
        return True
    except Exception as e:
        print(f"❌ Bedrock 접근 실패: {e}")
        return False

def get_available_models():
    """사용 가능한 Bedrock 모델 확인"""
    try:
        bedrock_client = boto3.client('bedrock', region_name=Config.BEDROCK_REGION)
        response = bedrock_client.list_foundation_models()
        
        available_models = []
        for model in response['modelSummaries']:
            if 'anthropic' in model['modelId'].lower():
                available_models.append(model['modelId'])
        
        print("사용 가능한 Claude 모델:")
        for model in available_models:
            print(f"  - {model}")
        
        return available_models
    except Exception as e:
        print(f"❌ 모델 목록 조회 실패: {e}")
        return []

if __name__ == "__main__":
    print("🔧 Bedrock 접근 권한 확인 중...")
    check_bedrock_access()
    get_available_models()