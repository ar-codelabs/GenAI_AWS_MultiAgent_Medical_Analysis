"""
Similar Search Agent using OpenSearch
"""
import json
from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from opensearch_multimodal import opensearch_multimodal

class SimilarSearchAgent:
    """유사사례 검색 에이전트"""
    
    def __init__(self):
        self.opensearch = opensearch_multimodal
    
    def search_similar_cases(self, diagnosis_result: Dict[str, Any], keywords: str, image_data: bytes = None) -> Dict[str, Any]:
        """진단 결과를 바탕으로 유사사례 검색 (5-1 진단결과 포함)"""
        
        try:
            # 5-1 진단결과에서 모든 정보 추출
            diagnosis = diagnosis_result.get('diagnosis', '')
            findings = diagnosis_result.get('findings', '')
            location = diagnosis_result.get('location', '')
            
            # 진단명 + 소견 + 위치 + 사용자 키워드를 모두 조합
            search_components = []
            if diagnosis:
                search_components.append(diagnosis)
            if findings:
                search_components.append(findings)
            if location:
                search_components.append(location)
            if keywords:
                search_components.append(keywords)
            
            search_query = ', '.join(search_components).strip(', ')
            
            print(f"🔍 [OpenSearch 검색쿼리] '{search_query}'")
            
            # OpenSearch에서 유사사례 검색
            similar_cases = self.opensearch.search_similar_cases(
                query_text=search_query,
                query_image=image_data,
                top_k=5
            )
            
            # 검색 결과 포맷팅
            formatted_cases = []
            for case in similar_cases:
                formatted_case = {
                    'case_id': case.get('u_id', 'N/A'),
                    'diagnosis': case.get('diagnosis', 'Unknown'),
                    'description': case.get('description', ''),
                    'similarity_score': case.get('similarity_score', 0),
                    'patient_info': {
                        'age': case.get('age', 'N/A'),
                        'sex': case.get('sex', 'N/A')
                    }
                }
                formatted_cases.append(formatted_case)
            
            # 통계 정보 생성
            if formatted_cases:
                avg_similarity = sum(case['similarity_score'] for case in formatted_cases) / len(formatted_cases)
                most_common_diagnosis = max(set(case['diagnosis'] for case in formatted_cases), 
                                          key=lambda x: sum(1 for case in formatted_cases if case['diagnosis'] == x))
            else:
                avg_similarity = 0
                most_common_diagnosis = 'No similar cases found'
            
            return {
                'success': True,
                'similar_cases': formatted_cases,
                'total_found': len(formatted_cases),
                'average_similarity': round(avg_similarity, 3),
                'most_common_diagnosis': most_common_diagnosis,
                'search_query': search_query
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'similar_cases': [],
                'total_found': 0,
                'average_similarity': 0,
                'most_common_diagnosis': 'Search failed'
            }