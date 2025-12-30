# 🏥 Medical AI Analysis System

An intelligent medical image analysis system powered by AWS Bedrock and LangGraph multi-agent architecture.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure AWS
aws configure

# 3. Set up environment variables
cp .env.example .env
# Edit .env file with your configuration

# 4. Run Jupyter Notebook
jupyter notebook medical_analysis_v1.0.ipynb
```

## 🏗️ System Architecture

### LangGraph Multi-Agent System
- **Disease Detection Agent**: Medical image analysis using Bedrock Claude Vision
- **Similar Search Agent**: Find similar cases using OpenSearch
- **Report Generation Agent**: Generate automated 5-sentence medical reports
- **Alert Agent**: Assess urgency and send notifications

### Analysis Output
- **Disease Detection**: Disease name and confidence score
- **Similar Cases**: Visual similarity graph
- **Next Actions**: Recommended medical procedures
- **Medical Report**: Automated 5-sentence summary
- **Alert System**: Emergency notification (yes/no)

## ⚙️ Configuration Guide

### 1. AWS Account Setup
```bash
# Install and configure AWS CLI
aws configure
# Access Key ID: YOUR_ACCESS_KEY
# Secret Access Key: YOUR_SECRET_KEY
# Default region: us-east-1
```

### 2. Environment Variables
Configure the following values in your `.env` file:

```env
# OpenSearch Configuration
OPENSEARCH_ENDPOINT=https://your-opensearch-endpoint.amazonaws.com
OPENSEARCH_INDEX=medical-images

# AWS Configuration
AWS_REGION=us-east-1
S3_BUCKET=your-medical-bucket

# Email Notification (Optional)
DOCTOR_EMAIL=doctor@hospital.com
SES_SENDER_EMAIL=system@hospital.com

# Bedrock Model Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
```

### 3. AWS Permissions
Required AWS service permissions:
- **Bedrock**: Model invocation access
- **S3**: Image upload/download permissions
- **SES**: Email sending permissions (optional)
- **OpenSearch**: Search and indexing access

## 📁 Project Structure

```
medical-ai-analysis/
├── medical_analysis_v1.0.ipynb     # Main Jupyter Notebook
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── config.py                        # Configuration management
├── agents/                          # LangGraph agents
│   ├── disease_detection.py
│   ├── report_generation.py
│   ├── similar_search.py
│   └── alert_system.py
├── sample_images/                   # Sample test images
│   └── MPX1007.png
└── README.md
```

## 🎯 Usage

1. **Launch Jupyter Notebook**: Open `medical_analysis_v1.0.ipynb`
2. **Upload Medical Image**: Select your medical image file
3. **Enter Keywords**: Input symptoms or areas of interest
4. **Run Analysis**: Execute the LangGraph workflow
5. **Review Results**: Check diagnosis, report, and alerts

## 🔒 Privacy & Security

- Medical images are processed securely through AWS services
- No patient data is stored permanently
- All communications are encrypted
- Compliant with healthcare data protection standards

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

This system is for research and educational purposes only. Always consult with qualified medical professionals for actual medical diagnosis and treatment decisions.
