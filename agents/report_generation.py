"""
Report Generation Agent using AWS Bedrock
"""
import json
import boto3
from typing import Dict, Any
from config import Config

class ReportGenerationAgent:
    """의료 보고서 생성 에이전트"""
    
    def __init__(self):
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=Config.BEDROCK_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
        )
        self.model_id = Config.BEDROCK_MODEL_ID
    
    def generate_report(self, diagnosis_result: Dict[str, Any]) -> Dict[str, Any]:
        """진단 결과를 바탕으로 5문장 의료 보고서 생성"""
        
        diagnosis = diagnosis_result.get('diagnosis', '진단명 없음')
        confidence = diagnosis_result.get('confidence', '0%')
        findings = diagnosis_result.get('findings', '소견 없음')
        location = diagnosis_result.get('location', '위치 불명')
        
        prompt = f"""
다음 진단 정보를 바탕으로 정확히 5문장으로 구성된 의료 보고서를 작성해주세요.

진단명: {diagnosis}
신뢰도: {confidence}
주요 소견: {findings}
해부학적 위치: {location}

보고서 작성 규칙:
1. 정확히 5문장으로 작성
2. 의료진이 이해할 수 있는 전문 용어 사용
3. 객관적이고 간결한 문체
4. 추가 검사나 치료 방향 포함

5문장 의료 보고서:
"""
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 800,
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }]
                })
            )
            
            result = json.loads(response['body'].read())
            report_text = result['content'][0]['text'].strip()
            
            # 5문장 추출
            sentences = self._extract_sentences(report_text)
            
            # 다음 조치 제안 생성
            next_actions = self._generate_next_actions(diagnosis, findings)
            
            return {
                'success': True,
                'report': ' '.join(sentences[:5]),  # 정확히 5문장
                'next_actions': next_actions,
                'sentence_count': len(sentences[:5])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'report': f'{diagnosis}에 대한 추가 분석이 필요합니다. 전문의 상담을 권장합니다. 정기적인 추적관찰이 필요합니다. 환자의 증상 변화를 모니터링해야 합니다. 필요시 추가 검사를 시행하겠습니다.',
                'next_actions': '전문의 상담 및 추가 검사 필요'
            }
    
    def _extract_sentences(self, text: str) -> list:
        """텍스트에서 문장 추출"""
        import re
        
        # 문장 분리 (한국어 문장 부호 고려)
        sentences = re.split(r'[.!?]\s+', text)
        
        # 빈 문장 제거 및 정리
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # 너무 짧은 문장 제외
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def _generate_next_actions(self, diagnosis: str, findings: str) -> str:
        """다음 조치 제안 생성"""
        
        # 간단한 규칙 기반 조치 제안
        if any(keyword in diagnosis.lower() for keyword in ['종양', 'tumor', '암', 'cancer']):
            return f"{diagnosis}에 대한 조직검사 및 종양내과 협진이 필요합니다. 추가 영상검사를 통한 병기 결정이 중요합니다."
        elif any(keyword in diagnosis.lower() for keyword in ['출혈', 'bleeding', '혈종']):
            return f"{diagnosis}에 대한 응급 처치 및 지혈술이 필요할 수 있습니다. 혈액검사 및 응고기능 검사를 시행하겠습니다."
        elif any(keyword in diagnosis.lower() for keyword in ['감염', 'infection', '염증']):
            return f"{diagnosis}에 대한 항생제 치료 및 염증 수치 모니터링이 필요합니다. 배양검사를 통한 원인균 확인을 권장합니다."
        else:
            return f"{diagnosis}에 대한 전문의 상담 및 정기적인 추적관찰이 필요합니다. 증상 변화 시 즉시 재검사를 시행하겠습니다."