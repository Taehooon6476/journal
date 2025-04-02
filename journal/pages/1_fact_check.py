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

def invoke_model(client, prompt, image_b64=None, model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0"):
    try:
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name='us-east-1'
        )

        system_prompt = f"""[분석 요청]
당신은 팩트체크 전문가입니다. 주어진 내용의 사실관계를 철저히 검증해주세요."""

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

def check_facts(text, image_b64=None):
    client = get_bedrock_client()
    prompt = f"""
    다음 텍스트의 사실 관계를 검증하고 신뢰할 수 있는 정보와 검증이 필요한 정보를 구분해서 분석해주세요:
    
    {text}
    
    [분석 형식]
    1. 신뢰할 수 있는 정보:
    - (정보 1)
    - (정보 2)
    
    2. 검증이 필요한 정보:
    - (정보 1): (검증 필요 이유)
    - (정보 2): (검증 필요 이유)
    """
    return invoke_model(client, prompt, image_b64)

def main():
    st.set_page_config(page_title="팩트 체크", layout="wide")
    
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
    if 'fact_check_result' not in st.session_state:
        st.session_state.fact_check_result = None

    # 상단 탭 (현재 탭 활성화)
    tabs = ["기사 작성", "팩트 체크", "데이터 분석", "맞춤법 교정", "SEO 제목"]
    selected_tab = st.radio("메뉴", tabs, index=1, horizontal=True, label_visibility="collapsed")
    
    # 다른 탭 클릭 시 해당 페이지로 이동
    if selected_tab == "기사 작성":
        st.switch_page("app.py")
    elif selected_tab == "데이터 분석":
        st.switch_page("app.py")
    elif selected_tab == "맞춤법 교정":
        st.switch_page("app.py")

    # 서브 메뉴
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("모델", ["Nova-Pro"], label_visibility="collapsed")

    # 메인 콘텐츠 영역
    col_left, col_right = st.columns(2)
    
    with col_left:
        text_input = st.text_area(
            "팩트체크할 내용을 입력하세요",
            value=st.session_state.current_text,
            height=400,
            key="text_input"
        )
        
        if st.button("분석하기", use_container_width=True):
            if text_input.strip():
                with st.spinner('처리 중...'):
                    result = check_facts(text_input)
                    if result:
                        st.session_state.fact_check_result = result
                        st.rerun()
    
    with col_right:
        if st.session_state.fact_check_result:
            st.markdown("### 검증 가능성: 0")
            st.write(st.session_state.fact_check_result)
            st.markdown("### 검증 가능성 이유:")
            st.markdown("### 가능한 이유:")
            st.write("해당 문장은 객관적으로 평가할 사실이나 주장이 있음.")
            st.markdown("### 불가능한 이유:")
            st.write("검증이 필요한 구체적인 정보나 주장이 포함되어 있음.")
            st.markdown("### 수집된 증거:")
            st.write("관련 데이터 및 참고자료")
            st.markdown("### 분야별 고려사항:")
            st.write("각 분야별 전문성이 필요한 부분 분석")

    # 글자 수 카운터
    current_chars = len(text_input)
    st.markdown(f'<p class="word-counter">{current_chars}자/3,000자</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
