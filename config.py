"""
Configuration management for Medical AI Analysis System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """System configuration"""
    
    # AWS Settings
    AWS_REGION = os.getenv('AWS_REGION', 'us-west-2')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    # S3 Settings
    S3_BUCKET = os.getenv('S3_BUCKET')
    
    # Bedrock Settings
    BEDROCK_MODEL_ID = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
    BEDROCK_REGION = os.getenv('BEDROCK_REGION', 'us-east-1')
    
    # Email Settings
    DOCTOR_EMAIL = os.getenv('DOCTOR_EMAIL', '')
    SES_SENDER_EMAIL = os.getenv('SES_SENDER_EMAIL', '')
    ENABLE_EMAIL_ALERTS = os.getenv('ENABLE_EMAIL_ALERTS', 'false').lower() == 'true'
    
    # System Settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    MAX_IMAGE_SIZE_MB = int(os.getenv('MAX_IMAGE_SIZE_MB', '10'))
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required_fields = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY', 
            'S3_BUCKET',
            'OPENSEARCH_ENDPOINT'
        ]
        
        missing_fields = []
        for field in required_fields:
            if not getattr(cls, field):
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")
        
        return True