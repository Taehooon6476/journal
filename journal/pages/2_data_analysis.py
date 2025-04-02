import streamlit as st
import boto3
import json
from PIL import Image
import io

def get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name='us-east-1'
    )

def process_image_for_bedrock(image):
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return img_bytes

def invoke_model(client, prompt, image_b64=None, model_id="us.anthropic.claude-3-sonnet-20240229-v1:0"):
    try:
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name='us-east-1'
        )

        system_prompt = f"""[분석 요청]
당신은 데이터 분석 전문가입니다. 주어진 텍스트에서 핵심 키워드를 추출하고 내용을 요약해주세요."""

        content = [{"text": prompt}]
        
        if image_b64:
            content.append({
                "image": {
                    "format": "jpeg",
                    "source": {
                        "bytes": image_b64
                    }
                }
            })

        response = bedrock_runtime.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{
                "role": "user",
                "content": content
            }],
            inferenceConfig={
                "maxTokens": 3000,
                "temperature": 0.3,
            }
        )

        try:
            response_text = response['output']['content'][0]['text']
        except:
            try:
                response_text = response['output']['content']['text']
            except:
                try:
                    response_text = response['output']['message']['content'][0]['text']
                except:
                    try:
                        response_text = response['output']['message']['text']
                    except:
                        response_text = str(response)

        return response_text
            
    except Exception as e:
        st.error(f"모델 호출 중 오류 발생: {str(e)}")
        print(f"상세 오류: {str(e)}")
        return None

def analyze_content(text, image_b64=None):
    client = get_bedrock_client()
    prompt = f"""
    다음 텍스트를 분석하여 핵심 키워드를 추출하고 내용을 요약해주세요:
    
    {text}
    
    [분석 형식]
    1. 핵심 키워드 (중요도 순):
    - 키워드1: (관련 문맥)
    - 키워드2: (관련 문맥)
    - 키워드3: (관련 문맥)
    
    2. 주요 주제:
    - (주제 1)
    - (주제 2)
    
    3. 내용 요약:
    (300자 이내로 핵심 내용 요약)
    
    4. 추가 분석:
    - 글의 톤과 스타일:
    - 주요 논점:
    - 데이터/통계 정보:
    """
    return invoke_model(client, prompt, image_b64)

def main():
    st.set_page_config(page_title="데이터 분석", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .stTabs {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }
        .stButton button {
            background-color: #4C7BF4;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 2rem;
        }
        .word-counter {
            color: #6c757d;
            text-align: right;
            font-size: 0.9em;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    # 상단 탭 (현재 탭 활성화)
    tabs = ["기사 작성", "팩트 체크", "데이터 분석", "맞춤법 교정", "SEO 제목"]
    selected_tab = st.radio("메뉴", tabs, index=2, horizontal=True, label_visibility="collapsed")
    
    # 다른 탭 클릭 시 해당 페이지로 이동
    if selected_tab == "기사 작성":
        st.switch_page("app")
    elif selected_tab == "팩트 체크":
        st.switch_page("pages/fact_check")
    elif selected_tab == "맞춤법 교정":
        st.switch_page("app")

    # 서브 메뉴
    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox("모델", ["Nova-Pro"], label_visibility="collapsed")
    with col2:
        style = st.selectbox("스타일", ["데이터 인사이트 추출"], label_visibility="collapsed")

    # 메인 콘텐츠 영역
    col_left, col_right = st.columns(2)
    
    with col_left:
        text_input = st.text_area(
            "분석할 내용을 입력하세요",
            value=st.session_state.current_text,
            height=400,
            key="text_input"
        )
        
        if st.button("분석하기", use_container_width=True):
            if text_input.strip():
                with st.spinner('처리 중...'):
                    result = analyze_content(text_input)
                    if result:
                        st.session_state.analysis_result = result
                        st.rerun()
    
    with col_right:
        if st.session_state.analysis_result:
            st.write(st.session_state.analysis_result)

    # 글자 수 카운터
    current_chars = len(text_input)
    st.markdown(f'<p class="word-counter">{current_chars}자/3,000자</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
