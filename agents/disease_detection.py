"""
Disease Detection Agent using AWS Bedrock
"""
import json
import base64
import boto3
from typing import Dict, Any
from config import Config

class DiseaseDetectionAgent:
    """멀티모달 질병 진단 에이전트"""
    
    def __init__(self):
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=Config.BEDROCK_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
        )
        self.model_id = Config.BEDROCK_MODEL_ID
    
    def analyze_image(self, image_data: bytes, keywords: str) -> Dict[str, Any]:
        """이미지와 키워드를 분석하여 질병 진단"""
        
        # 이미지를 base64로 인코딩
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Claude Vision 프롬프트
        prompt = f"""
당신은 전문 의료진입니다. 제공된 의료 이미지를 분석하고 다음 키워드를 참고하여 진단해주세요.

키워드: {keywords}

다음 형식으로 응답해주세요:
1. 질병명: [구체적인 진단명]
2. 신뢰도: [0-100% 숫자만]
3. 주요 소견: [관찰된 이상 소견]
4. 해부학적 위치: [영향받은 부위]

정확하고 간결하게 답변해주세요.
"""
        
        # Bedrock 호출
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }]
                })
            )
            
            # 응답 파싱
            result = json.loads(response['body'].read())
            analysis_text = result['content'][0]['text']
            
            # 결과 파싱
            diagnosis_result = self._parse_analysis(analysis_text)
            
            return {
                'success': True,
                'diagnosis': diagnosis_result.get('disease_name', '진단 분석 중'),
                'confidence': diagnosis_result.get('confidence', '85%'),
                'findings': diagnosis_result.get('findings', '소견 분석 중'),
                'location': diagnosis_result.get('location', '위치 분석 중'),
                'raw_analysis': analysis_text
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'diagnosis': '분석 실패',
                'confidence': '0%'
            }
    
    def _parse_analysis(self, text: str) -> Dict[str, str]:
        """분석 결과 텍스트를 파싱"""
        result = {}
        
        lines = text.split('\n')
        for line in lines:
            if '질병명:' in line or '진단명:' in line:
                result['disease_name'] = line.split(':', 1)[1].strip()
            elif '신뢰도:' in line:
                confidence = line.split(':', 1)[1].strip()
                # 숫자만 추출
                import re
                numbers = re.findall(r'\d+', confidence)
                if numbers:
                    result['confidence'] = f"{numbers[0]}%"
            elif '주요 소견:' in line or '소견:' in line:
                result['findings'] = line.split(':', 1)[1].strip()
            elif '해부학적 위치:' in line or '위치:' in line:
                result['location'] = line.split(':', 1)[1].strip()
        
        return result