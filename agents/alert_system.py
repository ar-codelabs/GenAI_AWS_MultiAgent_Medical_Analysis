"""
Alert System Agent for emergency notifications
"""
import json
import boto3
from typing import Dict, Any
from datetime import datetime
from config import Config

class AlertSystemAgent:
    """응급 알림 시스템 에이전트"""
    
    def __init__(self):
        if Config.ENABLE_EMAIL_ALERTS:
            self.ses_client = boto3.client(
                'ses',
                region_name=Config.AWS_REGION,
                aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
            )
        else:
            self.ses_client = None
    
    def evaluate_alert_need(self, diagnosis_result: Dict[str, Any]) -> Dict[str, Any]:
        """진단 결과를 바탕으로 응급 알림 필요성 판단"""
        
        diagnosis = diagnosis_result.get('diagnosis', '').lower()
        confidence = diagnosis_result.get('confidence', '0%')
        findings = diagnosis_result.get('findings', '').lower()
        
        # 응급 상황 키워드 정의
        emergency_keywords = [
            '출혈', 'bleeding', '혈종', 'hematoma',
            '뇌졸중', 'stroke', '경색', 'infarction',
            '종양', 'tumor', '암', 'cancer', 'malignant',
            '파열', 'rupture', '천공', 'perforation',
            '응급', 'emergency', '위험', 'critical'
        ]
        
        # 신뢰도 추출
        confidence_num = self._extract_confidence_number(confidence)
        
        # 응급도 판단 로직
        alert_needed = False
        alert_reason = []
        
        # 1. 응급 키워드 검사
        for keyword in emergency_keywords:
            if keyword in diagnosis or keyword in findings:
                alert_needed = True
                alert_reason.append(f"응급 키워드 감지: {keyword}")
        
        # 2. 높은 신뢰도 + 심각한 진단
        if confidence_num >= 80:
            serious_keywords = ['종양', 'tumor', '출혈', 'bleeding', '뇌졸중', 'stroke']
            for keyword in serious_keywords:
                if keyword in diagnosis:
                    alert_needed = True
                    alert_reason.append(f"고신뢰도 심각 진단: {keyword} ({confidence})")
        
        # 알림 결과
        alert_result = {
            'alert_needed': 'yes' if alert_needed else 'no',
            'alert_reason': '; '.join(alert_reason) if alert_reason else '정상 범위',
            'confidence_threshold': confidence_num,
            'timestamp': datetime.now().isoformat()
        }
        
        # 이메일 발송 (설정된 경우)
        if alert_needed and Config.ENABLE_EMAIL_ALERTS and Config.DOCTOR_EMAIL:
            email_result = self._send_alert_email(diagnosis_result, alert_reason)
            alert_result['email_sent'] = email_result['success']
            alert_result['email_message_id'] = email_result.get('message_id', '')
        else:
            alert_result['email_sent'] = False
            alert_result['email_message_id'] = ''
        
        return alert_result
    
    def _extract_confidence_number(self, confidence_str: str) -> int:
        """신뢰도 문자열에서 숫자 추출"""
        import re
        numbers = re.findall(r'\d+', confidence_str)
        return int(numbers[0]) if numbers else 0
    
    def _send_alert_email(self, diagnosis_result: Dict[str, Any], alert_reasons: list) -> Dict[str, Any]:
        """응급 알림 이메일 발송"""
        
        if not self.ses_client:
            return {'success': False, 'error': 'SES client not configured'}
        
        try:
            diagnosis = diagnosis_result.get('diagnosis', '진단명 없음')
            confidence = diagnosis_result.get('confidence', '0%')
            findings = diagnosis_result.get('findings', '소견 없음')
            
            subject = f"🚨 응급 의료 알림 - {diagnosis}"
            
            body_text = f"""
응급 의료 상황이 감지되었습니다.

=== 진단 정보 ===
진단명: {diagnosis}
신뢰도: {confidence}
주요 소견: {findings}

=== 알림 사유 ===
{chr(10).join(f"• {reason}" for reason in alert_reasons)}

=== 시스템 정보 ===
분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
시스템: Medical AI Analysis System

즉시 확인 및 조치가 필요합니다.

---
Medical AI Analysis System
Powered by AWS Bedrock + LangGraph
            """
            
            response = self.ses_client.send_email(
                Source=Config.SES_SENDER_EMAIL,
                Destination={'ToAddresses': [Config.DOCTOR_EMAIL]},
                Message={
                    'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                    'Body': {'Text': {'Data': body_text, 'Charset': 'UTF-8'}}
                }
            )
            
            return {
                'success': True,
                'message_id': response['MessageId'],
                'recipient': Config.DOCTOR_EMAIL
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }